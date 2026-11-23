// See profile.h. Windows only; elsewhere this is a no-op so the runtime keeps
// the same shape on every host.
#include "profile.h"
#include "arm64_context.h"

#if defined(_WIN32)

#include <windows.h>
#include <tlhelp32.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <map>
#include <numeric>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace arc {
namespace {

struct Symbol {
  uint64_t rva;
  std::string name;
};

struct ModuleRange {
  uint64_t base;
  uint64_t size;
  std::string name;
};

struct State {
  std::vector<Symbol> symbols;  // sorted by rva
  std::vector<uint32_t> counts; // one per symbol
  std::vector<uint32_t> main_counts;  // same, restricted to the boot thread
  uint64_t image_base = 0;
  uint64_t image_size = 0;
  std::map<std::string, uint32_t> external;       // module+64K bucket
  std::map<std::string, uint32_t> main_external;  // boot thread only
  std::map<uint64_t, uint32_t> main_guest;  // boot thread's last guest entry
  std::unordered_map<uint64_t, std::string> page_module;  // rip page -> module
  std::unordered_map<DWORD, bool> dead_tids;
  std::vector<DWORD> tids;
  DWORD main_tid = 0;
  uint64_t interval_ms = 5;
  uint32_t samples = 0, in_image = 0, in_external = 0, unknown = 0;
  uint32_t main_samples = 0, main_in_image = 0, main_in_external = 0,
           main_unknown = 0;
  // The aggregate histogram cannot say which thread is doing the work: every
  // thread is sampled once per pass, so the one busy worker is diluted by the
  // forty parked ones. GetThreadTimes does separate them, and the per-thread
  // histograms then name what the busy one is sitting in.
  struct TidStat {
    uint64_t cpu_delta = 0;   // 100 ns of CPU used since the last report
    uint64_t last_cpu = 0;    // absolute CPU time at the previous sample
    uint64_t last_rsp = 0;    // stack pointer at the previous sample
    uint32_t samples = 0;
    std::map<uint64_t, uint32_t> host;    // host symbol rva -> samples
    std::map<std::string, uint32_t> ext;  // external module bucket -> samples
  };
  std::unordered_map<DWORD, TidStat> per_tid;
  std::atomic<bool> stop{false};
  std::thread thread;
};

State g;

bool IsHex(const char* s) {
  if (!s || !*s) return false;
  for (; *s; ++s) {
    const char c = *s;
    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') ||
          (c >= 'A' && c <= 'F')))
      return false;
  }
  return true;
}

// The map is a text dump of the linker's symbol table. Only the "Publics by
// Value" lines are wanted: `0001:00001000  fn0_...  0000000140002000 f obj`.
// Section header lines also start with a section:offset pair, but their third
// token is `.text` rather than an address, which is the cheap discriminator.
void LoadMap(const char* path) {
  std::FILE* f = std::fopen(path, "rb");
  if (!f) {
    std::fprintf(stderr, "[sample] map %s not readable; offsets only\n", path);
    return;
  }
  char line[1024];
  uint64_t preferred = UINT64_C(0x140000000);
  while (std::fgets(line, sizeof line, f)) {
    const char* load = std::strstr(line, "Preferred load address is");
    if (load) {
      unsigned long long value = 0;
      if (std::sscanf(load + 25, "%llx", &value) == 1 && value)
        preferred = value;
      continue;
    }
    char at[64], name[512], va[64];
    if (std::sscanf(line, " %63s %511s %63s", at, name, va) != 3) continue;
    if (!std::strchr(at, ':')) continue;
    unsigned section = 0;
    if (std::sscanf(at, "%x:", &section) != 1 || section != 1) continue;
    if (!IsHex(va)) continue;  // ".text", "CODE", and the length column
    if (name[0] == '$' || name[0] == '.') continue;  // unwind/data labels
    char* end = nullptr;
    const unsigned long long value = std::strtoull(va, &end, 16);
    if (!end || *end || value < preferred) continue;
    Symbol symbol;
    symbol.rva = static_cast<uint64_t>(value) - preferred;
    symbol.name = name;
    g.symbols.push_back(std::move(symbol));
  }
  std::fclose(f);
  std::sort(g.symbols.begin(), g.symbols.end(),
            [](const Symbol& a, const Symbol& b) { return a.rva < b.rva; });
  g.symbols.erase(std::unique(g.symbols.begin(), g.symbols.end(),
                              [](const Symbol& a, const Symbol& b) {
                                return a.rva == b.rva;
                              }),
                  g.symbols.end());
  std::fprintf(stderr, "[sample] %zu symbols from %s (preferred %#llx)\n",
               g.symbols.size(), path,
               static_cast<unsigned long long>(preferred));
}

void LoadImageRange() {
  const HMODULE module = GetModuleHandleW(nullptr);
  g.image_base = reinterpret_cast<uint64_t>(module);
  const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(module);
  const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS*>(
      reinterpret_cast<const unsigned char*>(module) + dos->e_lfanew);
  g.image_size = nt->OptionalHeader.SizeOfImage;
  g.counts.assign(g.symbols.size(), 0);
}

const std::string& ModuleFor(uint64_t address) {
  // One cache entry per 4K page: a page never spans two modules, and the exact
  // offset is appended by DescribeAddress so a coarse bucket cannot hide which
  // adjacent entry point the thread is in.
  const uint64_t page = address >> 12;
  auto it = g.page_module.find(page);
  if (it != g.page_module.end()) return it->second;
  std::string name = "?";
  HMODULE module = nullptr;
  if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                             GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                         reinterpret_cast<LPCWSTR>(static_cast<uintptr_t>(address)),
                         &module) &&
      module) {
    wchar_t path[MAX_PATH];
    if (GetModuleFileNameW(module, path, MAX_PATH)) {
      const wchar_t* slash = std::wcsrchr(path, L'\\');
      const wchar_t* base = slash ? slash + 1 : path;
      char narrow[MAX_PATH];
      if (WideCharToMultiByte(CP_UTF8, 0, base, -1, narrow, MAX_PATH, nullptr,
                              nullptr))
        name = narrow;
    }
  }
  return g.page_module.emplace(page, name).first->second;
}

// module+0xoffset with the real offset. The sampler used to round this to 256
// bytes, which put glGetError and glGetDoublev in the same bucket.
void DescribeAddress(uint64_t address, char* out, size_t size) {
  const std::string& module = ModuleFor(address);
  if (module == "?") {
    std::snprintf(out, size, "?+%#llx",
                  static_cast<unsigned long long>(address));
    return;
  }
  uint64_t off = 0;
  HMODULE base = nullptr;
  if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                             GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                         reinterpret_cast<LPCWSTR>(static_cast<uintptr_t>(address)),
                         &base) &&
      base)
    off = address - reinterpret_cast<uint64_t>(base);
  std::snprintf(out, size, "%s+%#llx", module.c_str(),
                static_cast<unsigned long long>(off));
}

const char* HitSymbol(uint64_t rva, uint64_t* symbol_rva) {
  auto it = std::upper_bound(
      g.symbols.begin(), g.symbols.end(), rva,
      [](uint64_t value, const Symbol& symbol) { return value < symbol.rva; });
  if (it == g.symbols.begin()) return nullptr;
  --it;
  const uint64_t gap = rva - it->rva;
  if (gap > UINT64_C(0x40000)) return nullptr;  // a hole, not a function body
  *symbol_rva = it->rva;
  return it->name.c_str();
}

const Symbol* FindSymbol(uint64_t rva) {
  auto it = std::lower_bound(
      g.symbols.begin(), g.symbols.end(), rva,
      [](const Symbol& symbol, uint64_t value) { return symbol.rva < value; });
  if (it != g.symbols.end() && it->rva == rva) return &*it;
  return nullptr;
}

// A raw stack read, not a StackWalk64 unwind: every frame here is well-formed
// enough that the return addresses sit at 8-byte slots, and reading them
// directly keeps this free of dbghelp and its symbol-path delays. Return
// addresses are what the caller chain is made of anyway.
void DumpStack(DWORD tid, uint64_t rsp) {
  if (!rsp) return;
  uint64_t words[64];
  SIZE_T got = 0;
  if (!ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<LPCVOID>(rsp),
                         words, sizeof words, &got) ||
      got < sizeof(uint64_t))
    return;
  const size_t count = got / sizeof(uint64_t);
  std::fprintf(stderr, "[sample-stack %lu] rsp=%#llx words=%u\n",
               static_cast<unsigned long>(tid),
               static_cast<unsigned long long>(rsp),
               static_cast<unsigned>(count));
  for (size_t i = 0; i < count; ++i) {
    const uint64_t word = words[i];
    if (word >= g.image_base && word - g.image_base < g.image_size) {
      const uint64_t rva = word - g.image_base;
      uint64_t symbol_rva = 0;
      const char* name = HitSymbol(rva, &symbol_rva);
      if (name)
        std::fprintf(stderr, "[sample-stack %lu]   +%#04x  host+%#llx  %s\n",
                     static_cast<unsigned long>(tid),
                     static_cast<unsigned>(i * 8),
                     static_cast<unsigned long long>(symbol_rva), name);
      else
        std::fprintf(stderr, "[sample-stack %lu]   +%#04x  host+%#llx\n",
                     static_cast<unsigned long>(tid),
                     static_cast<unsigned>(i * 8),
                     static_cast<unsigned long long>(rva));
      continue;
    }
    if (word < 0x10000) continue;
    char desc[160];
    DescribeAddress(word, desc, sizeof desc);
    if (desc[0] != '?')
      std::fprintf(stderr, "[sample-stack %lu]   +%#04x  %s\n",
                   static_cast<unsigned long>(tid),
                   static_cast<unsigned>(i * 8), desc);
  }
}

void Record(uint64_t rip, DWORD tid, uint64_t guest_entry) {
  const bool main = tid != 0 && tid == g.main_tid;
  ++g.samples;
  State::TidStat& ts = g.per_tid[tid];
  ++ts.samples;
  if (main) {
    ++g.main_samples;
    // The entered-function ring is the only thing that says which guest body
    // asked for the wait the host is sitting in. It is read before the thread
    // is resumed, so the value belongs to the sampled instruction pointer.
    if (guest_entry) ++g.main_guest[guest_entry];
  }
  if (rip >= g.image_base && rip - g.image_base < g.image_size) {
    const uint64_t rva = rip - g.image_base;
    uint64_t symbol_rva = 0;
    const char* name = HitSymbol(rva, &symbol_rva);
    if (name) {
      const size_t index = static_cast<size_t>(
          std::lower_bound(g.symbols.begin(), g.symbols.end(), symbol_rva,
                           [](const Symbol& symbol, uint64_t value) {
                             return symbol.rva < value;
                           }) -
          g.symbols.begin());
      if (index < g.counts.size()) ++g.counts[index];
      if (main && index < g.main_counts.size()) ++g.main_counts[index];
      if (main) ++g.main_in_image;
      ++g.in_image;
      ++ts.host[symbol_rva];
    } else {
      ++g.unknown;
      if (main) ++g.main_unknown;
    }
    return;
  }
  char desc[160];
  DescribeAddress(rip, desc, sizeof desc);
  if (desc[0] == '?') {
    ++g.unknown;
    if (main) ++g.main_unknown;
  } else {
    ++g.external[desc];
    ++g.in_external;
    ++ts.ext[desc];
    if (main) {
      ++g.main_external[desc];
      ++g.main_in_external;
    }
  }
}

void RefreshThreads() {
  HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
  if (snapshot == INVALID_HANDLE_VALUE) return;
  THREADENTRY32 entry;
  entry.dwSize = sizeof entry;
  const DWORD self = GetCurrentThreadId();
  const DWORD process = GetCurrentProcessId();
  g.tids.clear();
  if (Thread32First(snapshot, &entry)) {
    do {
      if (entry.th32OwnerProcessID == process && entry.th32ThreadID != self &&
          entry.th32ThreadID != 0)
        g.tids.push_back(entry.th32ThreadID);
    } while (Thread32Next(snapshot, &entry));
  }
  CloseHandle(snapshot);
}

void SampleOne(DWORD tid) {
  HANDLE thread = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT |
                                 THREAD_QUERY_INFORMATION,
                             FALSE, tid);
  if (!thread) return;
  if (SuspendThread(thread) == static_cast<DWORD>(-1)) {
    CloseHandle(thread);
    return;
  }
  CONTEXT context;
  std::memset(&context, 0, sizeof context);
  context.ContextFlags = CONTEXT_CONTROL;
  uint64_t guest_entry = 0;
  const bool got = GetThreadContext(thread, &context) != 0;
  const uint64_t sampled_rsp = got ? context.Rsp : 0;
  if (got && tid == g.main_tid) guest_entry = arc_main_frame_at(0);
  // Resume before recording. Recording allocates, and a suspended thread may
  // hold the CRT heap lock; waiting for that lock while holding the thread
  // stopped is a deadlock the process cannot recover from.
  ResumeThread(thread);
  // Thread CPU time is the one number that distinguishes the busy worker from
  // the parked ones. Read it after the resume for the same heap-lock reason as
  // the recording below: the map insert may allocate.
  FILETIME created, exited, kernel, user;
  if (GetThreadTimes(thread, &created, &exited, &kernel, &user)) {
    const uint64_t total =
        (static_cast<uint64_t>(kernel.dwHighDateTime) << 32 |
         kernel.dwLowDateTime) +
        (static_cast<uint64_t>(user.dwHighDateTime) << 32 | user.dwLowDateTime);
    State::TidStat& ts = g.per_tid[tid];
    if (ts.last_cpu && total > ts.last_cpu) ts.cpu_delta += total - ts.last_cpu;
    ts.last_cpu = total;
    if (sampled_rsp) ts.last_rsp = sampled_rsp;
  }
  CloseHandle(thread);
  if (got) Record(context.Rip, tid, guest_entry);
}

void Report() {
  const uint32_t total = g.samples;
  if (!total) return;
  std::fprintf(stderr,
               "[sample] %u samples  image=%u external=%u unknown=%u\n", total,
               g.in_image, g.in_external, g.unknown);
  std::vector<uint32_t> order(g.symbols.size());
  std::iota(order.begin(), order.end(), 0);
  const size_t top = std::min<size_t>(16, order.size());
  std::partial_sort(order.begin(), order.begin() + top, order.end(),
                    [](uint32_t a, uint32_t b) {
                      return g.counts[a] > g.counts[b];
                    });
  for (size_t i = 0; i < top; ++i) {
    const uint32_t count = g.counts[order[i]];
    if (!count) break;
    const Symbol& symbol = g.symbols[order[i]];
    std::fprintf(stderr, "[sample]  %5.1f%%  %s+%#llx  %s\n",
                 100.0 * count / total, "image",
                 static_cast<unsigned long long>(symbol.rva),
                 symbol.name.c_str());
  }
  std::vector<std::pair<uint32_t, std::string>> external;
  external.reserve(g.external.size());
  for (const auto& item : g.external) external.push_back({item.second, item.first});
  std::sort(external.begin(), external.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
  for (size_t i = 0; i < external.size() && i < 10; ++i)
    std::fprintf(stderr, "[sample]  %5.1f%%  %s\n",
                 100.0 * external[i].first / total,
                 external[i].second.c_str());
  if (g.main_samples) {
    std::fprintf(stderr,
                 "[sample-main] %u samples  image=%u external=%u unknown=%u\n",
                 g.main_samples, g.main_in_image, g.main_in_external,
                 g.main_unknown);
    std::vector<uint32_t> main_order(g.symbols.size());
    std::iota(main_order.begin(), main_order.end(), 0);
    const size_t main_top = std::min<size_t>(6, main_order.size());
    std::partial_sort(main_order.begin(), main_order.begin() + main_top,
                      main_order.end(),
                      [](uint32_t a, uint32_t b) {
                        return g.main_counts[a] > g.main_counts[b];
                      });
    for (size_t i = 0; i < main_top; ++i) {
      const uint32_t count = g.main_counts[main_order[i]];
      if (!count) break;
      const Symbol& symbol = g.symbols[main_order[i]];
      std::fprintf(stderr, "[sample-main]  %5.1f%%  host+%#llx  %s\n",
                   100.0 * count / g.main_samples,
                   static_cast<unsigned long long>(symbol.rva),
                   symbol.name.c_str());
    }
    std::vector<std::pair<uint32_t, std::string>> main_external;
    for (const auto& item : g.main_external)
      main_external.push_back({item.second, item.first});
    std::sort(main_external.begin(), main_external.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    for (size_t i = 0; i < main_external.size() && i < 4; ++i)
      std::fprintf(stderr, "[sample-main]  %5.1f%%  %s\n",
                   100.0 * main_external[i].first / g.main_samples,
                   main_external[i].second.c_str());
    std::vector<std::pair<uint32_t, uint64_t>> main_guest;
    for (const auto& item : g.main_guest)
      main_guest.push_back({item.second, item.first});
    std::sort(main_guest.begin(), main_guest.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    for (size_t i = 0; i < main_guest.size() && i < 6; ++i) {
      const uint64_t packed = main_guest[i].second;
      std::fprintf(stderr, "[sample-main]  %5.1f%%  guest image=%u +%#llx\n",
                   100.0 * main_guest[i].first / g.main_samples,
                   static_cast<unsigned>(packed >> 56),
                   static_cast<unsigned long long>(
                       packed & UINT64_C(0xFFFFFFFFFFFFFF)));
    }
  }
  // Which thread is actually burning the CPU, and what is it sitting in? The
  // aggregate report above cannot answer that, because it counts every thread
  // once per pass regardless of whether the thread is running.
  std::vector<std::pair<uint64_t, DWORD>> busy;
  busy.reserve(g.per_tid.size());
  for (const auto& item : g.per_tid)
    if (item.second.cpu_delta) busy.push_back({item.second.cpu_delta, item.first});
  std::sort(busy.begin(), busy.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
  for (size_t i = 0; i < busy.size() && i < 3; ++i) {
    State::TidStat& ts = g.per_tid[busy[i].second];
    const uint32_t denom = ts.samples ? ts.samples : 1;
    std::fprintf(stderr, "[sample-tid %lu] cpu %.1f ms  samples=%u\n",
                 static_cast<unsigned long>(busy[i].second),
                 busy[i].first / 10000.0, ts.samples);
    std::vector<std::pair<uint32_t, uint64_t>> top_host;
    top_host.reserve(ts.host.size());
    for (const auto& hit : ts.host) top_host.push_back({hit.second, hit.first});
    std::sort(top_host.begin(), top_host.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    for (size_t k = 0; k < top_host.size() && k < 8; ++k) {
      const Symbol* symbol = FindSymbol(top_host[k].second);
      std::fprintf(stderr, "[sample-tid %lu]   %5.1f%%  host+%#llx  %s\n",
                   static_cast<unsigned long>(busy[i].second),
                   100.0 * top_host[k].first / denom,
                   static_cast<unsigned long long>(top_host[k].second),
                   symbol ? symbol->name.c_str() : "?");
    }
    std::vector<std::pair<uint32_t, std::string>> top_ext;
    top_ext.reserve(ts.ext.size());
    for (const auto& hit : ts.ext) top_ext.push_back({hit.second, hit.first});
    std::sort(top_ext.begin(), top_ext.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    for (size_t k = 0; k < top_ext.size() && k < 5; ++k)
      std::fprintf(stderr, "[sample-tid %lu]   %5.1f%%  %s\n",
                   static_cast<unsigned long>(busy[i].second),
                   100.0 * top_ext[k].first / denom,
                   top_ext[k].second.c_str());
    if (i < 2) DumpStack(busy[i].second, ts.last_rsp);
  }
  std::fill(g.counts.begin(), g.counts.end(), 0);
  std::fill(g.main_counts.begin(), g.main_counts.end(), 0);
  g.external.clear();
  g.main_external.clear();
  g.main_guest.clear();
  for (auto& item : g.per_tid) {
    item.second.cpu_delta = 0;
    item.second.samples = 0;
    item.second.host.clear();
    item.second.ext.clear();
  }
  g.samples = g.in_image = g.in_external = g.unknown = 0;
  g.main_samples = g.main_in_image = g.main_in_external = g.main_unknown = 0;
}

void Loop() {
  SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
  auto next_threads = std::chrono::steady_clock::now();
  auto next_report = next_threads + std::chrono::seconds(1);
  while (!g.stop.load(std::memory_order_relaxed)) {
    const auto now = std::chrono::steady_clock::now();
    if (now >= next_threads) {
      RefreshThreads();
      next_threads = now + std::chrono::milliseconds(500);
    }
    for (DWORD tid : g.tids) SampleOne(tid);
    if (std::chrono::steady_clock::now() >= next_report) {
      Report();
      next_report += std::chrono::seconds(1);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(g.interval_ms));
  }
}

}  // namespace

void StartSampler() {
  const char* interval = std::getenv("ARC_SAMPLE");
  if (!interval || !*interval) return;
  const int value = std::atoi(interval);
  if (value > 0) g.interval_ms = static_cast<uint64_t>(value);
  LoadImageRange();
  g.main_tid = GetCurrentThreadId();
  const char* map = std::getenv("ARC_SAMPLE_MAP");
  if (map && *map) {
    LoadMap(map);
    g.counts.assign(g.symbols.size(), 0);
  }
  g.main_counts.assign(g.symbols.size(), 0);
  g.stop.store(false);
  g.thread = std::thread(Loop);
  std::fprintf(stderr, "[sample] every %llu ms, image %#llx+%#llx\n",
               static_cast<unsigned long long>(g.interval_ms),
               static_cast<unsigned long long>(g.image_base),
               static_cast<unsigned long long>(g.image_size));
}

void StopSampler() {
  if (!g.thread.joinable()) return;
  g.stop.store(true);
  g.thread.join();
  Report();
}

}  // namespace arc

#else

namespace arc {
void StartSampler() {}
void StopSampler() {}
}  // namespace arc

#endif
