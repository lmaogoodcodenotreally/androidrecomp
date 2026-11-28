#if defined(_WIN32)
#include <malloc.h>
#else
#include <malloc.h>
#endif

#include "shim.h"

#include "arm64_context.h"

#include <atomic>
#include <cerrno>
#include <deque>
#include <string>
#include <mutex>
#include <vector>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#if defined(ARC_HAVE_ZLIB)
#include <zlib.h>
#endif

namespace arc {
namespace {

// --- layer 1: Android-only entry points ------------------------------------
// liblog. The engine is chatty; routing this to stderr on the first run is how
// you find out what it is unhappy about.

int AndroidLogVPrint(int prio, const char* tag, const char* fmt, va_list ap) {
  fprintf(stderr, "[%d] %s: ", prio, tag ? tag : "?");
  int n = vfprintf(stderr, fmt, ap);
  fputc('\n', stderr);
  return n;
}

int AndroidLogPrint(int prio, const char* tag, const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  int n = AndroidLogVPrint(prio, tag, fmt, ap);
  va_end(ap);
  return n;
}

int AndroidLogWrite(int prio, const char* tag, const char* text) {
  return fprintf(stderr, "[%d] %s: %s\n", prio, tag ? tag : "?",
                 text ? text : "");
}

// The failed condition and the tag come through the first two registers, which
// the thunk marshals correctly. The message after them is variadic and is not
// read: it would need the context, and an assertion that reports which check
// failed has already said the useful part.
[[noreturn]] void AndroidLogAssert(const char* cond, const char* tag,
                                   const char* fmt) {
  fprintf(stderr, "assertion failed: %s (%s) %s\n", cond ? cond : "?",
          tag ? tag : "?", fmt ? fmt : "");
  arc_trap(nullptr, "the guest failed an assertion");
  abort();
}

// Bionic's FORTIFY_SOURCE variants. Each takes the destination buffer's known
// size as an extra argument and traps on overflow. We are not trying to
// reproduce the diagnostics, only the semantics, so each forwards to the
// unchecked function -- which is exactly what Bionic does when the size is not
// known at compile time.
void* MemcpyChk(void* d, const void* s, size_t n, size_t) { return memcpy(d, s, n); }
void* MemmoveChk(void* d, const void* s, size_t n, size_t) { return memmove(d, s, n); }
char* StrcpyChk(char* d, const char* s, size_t) { return strcpy(d, s); }
char* StrcatChk(char* d, const char* s, size_t) { return strcat(d, s); }
size_t StrlenChk(const char* s, size_t) { return strlen(s); }
char* StrchrChk(const char* s, int c, size_t) {
  return const_cast<char*>(strchr(s, c));
}
int VsnprintfChk(char* d, size_t n, int, size_t, const char* fmt, va_list ap) {
  return vsnprintf(d, n, fmt, ap);
}

[[noreturn]] void Assert2(const char* file, int line, const char* func,
                          const char* msg) {
  fprintf(stderr, "%s:%d: %s: assertion failed: %s\n", file, line,
          func ? func : "?", msg ? msg : "");
  abort();
}

[[noreturn]] void StackChkFail() {
  fprintf(stderr, "stack protector: canary overwritten\n");
  abort();
}

// A data symbol, not a function -- the resolver hands back the address of real
// storage and the engine reads the canary out of it.
uintptr_t g_stack_chk_guard = 0;

int* Errno() { return &errno; }

// The engine registers destructors for its 1,527 static objects here. We never
// unload the image, so there is nothing to run them at, and recording them
// would be bookkeeping with no reader.
int CxaAtexit(void (*)(void*), void*, void*) { return 0; }
void CxaFinalize(void*) {}

[[noreturn]] void GuestAbort() {
  arc_trap(nullptr, "the guest called abort()");
  abort();
}

// Audio, declined politely.
//
// The APK ships OpenAL and the title imports thirty-two of its functions. The
// library is real ARM code and is not lifted, so a call into it is a branch
// the dispatcher cannot route -- reported as a miss and, when the render
// thread makes it, the end of the run.
//
// A host with no audio backend has a truthful answer available: there is no
// device. OpenAL is specified for exactly this -- alcOpenDevice returns null
// when it cannot open one, and a caller that handles null never asks for a
// context, a buffer or a source. So this is the device and context lifecycle
// and nothing else; the twenty-odd AL calls are unreachable behind a null
// device, and writing them before anything asks for them would be inventing
// behaviour rather than declining it.
void* AlcOpenDevice(const char*) { return nullptr; }
int AlcCloseDevice(void*) { return 1; }  // ALC_TRUE: nothing to fail
void* AlcCreateContext(void*, const int*) { return nullptr; }
int AlcMakeContextCurrent(void*) { return 1; }
void* AlcGetCurrentContext() { return nullptr; }
void* AlcGetContextsDevice(void*) { return nullptr; }
void AlcDestroyContext(void*) {}
void AlcProcessContext(void*) {}
void AlcSuspendContext(void*) {}
int AlcGetError(void*) { return 0; }  // ALC_NO_ERROR
// AL_NO_ERROR. A caller polls this after every call it makes and an invented
// error is worse than none: it turns "there is no sound" into "sound is
// broken", which a title is entitled to treat as fatal.
int AlGetError() { return 0; }
// Never null. A caller measures what this returns.
const char* AlGetString(int) { return ""; }
const char* AlcGetString(void*, int) { return ""; }

// And the rest of the title's audio surface, because it does not check.
//
// A null device is supposed to be the end of it, and the title asks for one,
// gets null, and then sets the distance model anyway -- so "unreachable
// behind a null device" was wrong about this engine. These are the thirty-two
// names it imports and no more: not a guess at OpenAL, an answer to the list.
//
// The queries are where care is needed. A setter that does nothing is
// invisible; a getter that leaves its output untouched hands back whatever
// was on the caller's stack, and a getter that answers "still playing"
// forever is a loading screen that never ends. So state reads as stopped and
// counts read as zero.
constexpr int kAlPosition = 0x1004;
constexpr int kAlDirection = 0x1005;
constexpr int kAlVelocity = 0x1006;
constexpr int kAlSourceState = 0x1010;
constexpr int kAlStopped = 0x1014;

int AlFloatCount(int param) {
  return (param == kAlPosition || param == kAlDirection ||
          param == kAlVelocity)
             ? 3
             : 1;
}

// Ids are handed out rather than zeroed: zero is "no object" in OpenAL, and a
// caller that is given it either retries or reports a failure.
unsigned AlNextName() {
  static std::atomic<unsigned> next{1};
  return next++;
}

void AlGenBuffers(int n, unsigned* out) {
  for (int i = 0; out && i < n; ++i) out[i] = AlNextName();
}
void AlGenSources(int n, unsigned* out) {
  for (int i = 0; out && i < n; ++i) out[i] = AlNextName();
}
void AlDeleteBuffers(int, const unsigned*) {}
void AlDeleteSources(int, const unsigned*) {}
void AlBufferData(unsigned, int, const void*, int, int) {}
void AlDistanceModel(int) {}
void AlListener3f(int, float, float, float) {}
void AlListenerf(int, float) {}
void AlListenerfv(int, const float*) {}
void AlGetListener3f(int, float* a, float* b, float* c) {
  if (a) *a = 0.0f;
  if (b) *b = 0.0f;
  if (c) *c = 0.0f;
}
void AlSource3f(unsigned, int, float, float, float) {}
void AlSourcef(unsigned, int, float) {}
void AlSourcei(unsigned, int, int) {}
void AlGetSourcef(unsigned, int, float* out) {
  if (out) *out = 0.0f;
}
void AlGetSourcefv(unsigned, int param, float* out) {
  for (int i = 0; out && i < AlFloatCount(param); ++i) out[i] = 0.0f;
}
void AlGetSourcei(unsigned, int param, int* out) {
  if (out) *out = param == kAlSourceState ? kAlStopped : 0;
}
void AlSourcePlay(unsigned) {}
void AlSourcePause(unsigned) {}
void AlSourceStop(unsigned) {}
void AlSourceRewind(unsigned) {}
void AlSourceQueueBuffers(unsigned, int, const unsigned*) {}
void AlSourceUnqueueBuffers(unsigned, int, unsigned*) {}

struct Entry {
  const char* name;
  void* fn;
};


// free, so that reuse can be delayed.
//
// A recompiled program is the same program, so a lifetime bug it has on the
// device it shipped on it has here too -- but "has" and "shows" are different
// things. The device's allocator hands a freed block back on its own schedule;
// this one hands it back immediately and to whoever asks next, which turns a
// dangling pointer that used to read stale-but-plausible fields into a read of
// somebody else's string. ARC_LEAK_FREE stops recycling entirely, which is not
// a fix and is not meant to be: if a fault goes away under it, the fault was a
// use-after-free, and that is worth knowing in one run rather than five.
//
// ponytail: leak-everything, because it answers the question. If a real
// quarantine is wanted -- hold N megabytes of freed blocks aside and release
// the oldest -- that is the upgrade, and it belongs here.
// Leaking everything is affordable for a boot and not for a session. Measured
// on one: about 875 MB a minute once the game is being played, 9 GB after ten
// minutes, at which point the machine is paging and every frame takes long
// enough that input looks dead. The lifetime bug it covers for is in the DLC
// mount walk, which is over before the first frame -- so the leak can be
// switched off once that is behind us, and ShimStopLeaking is how the host
// says so. Still not a fix; just one that does not have to be paid for as long
// as you play.
std::atomic<int> g_leak_state{-1};  // -1 undecided, 1 leaking, 0 recycling

// A free of a block the heap no longer knows about is a double free, and the
// heap's own reaction to one -- 0xc0000374, which ends the process -- arrives
// after the fact: it cannot say which block was freed twice or who asked. The
// question "is this still an allocated block" can be asked first instead.
// `malloc` here is the CRT's, and the CRT's heap is the process heap, so
// HeapValidate answers it in about ten nanoseconds for a block that is still
// allocated (only the bogus case costs real time, and that one happens once
// per bug). A block it rejects is skipped and named.
//
// This does not fix the double free; it makes it visible, survivable, and
// cheap. ARC_FREE_GUARD=0 turns it off.

// The exception object the last transferred throw landed with. Published so a
// free of that block can be named rather than guessed at from a heap report.
extern "C" uint64_t arc_eh_object(void);

namespace {

bool FreeGuardEnabled() {
  static const bool on = [] {
    const char* s = getenv("ARC_FREE_GUARD");
    return !(s && s[0] == '0' && s[1] == 0);
  }();
  return on;
}

bool BlockStillAllocated(void* p) {
  return HeapValidate(GetProcessHeap(), 0, p) != 0;
}

bool IsCaughtExceptionBlock(void* p) {
  const uint64_t object = arc_eh_object();
  if (!object) return false;
  const uint64_t here = reinterpret_cast<uint64_t>(p);
  return here + 0x100 >= object && here <= object;
}

// The free that matters is the one that came first, and the guard only ever
// sees the second. So the last frees are kept -- pointer, caller, thread -- and
// when the guard rejects a block the history is searched for earlier frees of
// it. One atomic increment and three stores per free, no lock, no allocation:
// a run with ARC_FREE_HISTORY=1 measures the same as one without.
// ARC_FREE_HISTORY=1 turns it on.
struct FreeEvent {
  void* block;
  void* from;
  unsigned long tid;
};

constexpr size_t kFreeHistory = 1u << 16;
FreeEvent g_free_history[kFreeHistory];
std::atomic<uint64_t> g_free_count{0};

bool FreeHistoryEnabled() {
  static const bool on = getenv("ARC_FREE_HISTORY") != nullptr;
  return on;
}

void FreeHistoryRecord(void* p) {
  const uint64_t n = g_free_count.fetch_add(1, std::memory_order_relaxed);
  FreeEvent& e = g_free_history[n & (kFreeHistory - 1)];
  e.block = p;
  e.from = _ReturnAddress();
  e.tid = GetCurrentThreadId();
}

// "mkt_host.exe+0x1234" for an address inside this image, the module and offset
// for another one, and the raw address when neither answers.
void DescribeFreeCaller(void* address, char* out, size_t size) {
  HMODULE mod = nullptr;
  if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                              GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                          static_cast<LPCSTR>(address), &mod) ||
      !mod) {
    _snprintf_s(out, size, _TRUNCATE, "%p", address);
    return;
  }
  char path[MAX_PATH] = {0};
  GetModuleFileNameA(mod, path, MAX_PATH);
  const char* name = strrchr(path, '\\');
  name = name ? name + 1 : path;
  _snprintf_s(out, size, _TRUNCATE, "%s+%#llx", name[0] ? name : "?",
              static_cast<unsigned long long>(
                  reinterpret_cast<uintptr_t>(address) -
                  reinterpret_cast<uintptr_t>(mod)));
}

void ReportFreeHistory(void* p) {
  if (!FreeHistoryEnabled()) return;
  const uint64_t seen = g_free_count.load(std::memory_order_relaxed);
  const uint64_t first = seen > kFreeHistory ? seen - kFreeHistory : 0;
  unsigned found = 0;
  for (uint64_t i = first; i < seen; ++i) {
    const FreeEvent& e = g_free_history[i & (kFreeHistory - 1)];
    if (e.block != p || e.from == nullptr) continue;
    char caller[160];
    DescribeFreeCaller(e.from, caller, sizeof caller);
    fprintf(stderr, "[free] earlier free #%llu by tid %lu from %s\n",
            static_cast<unsigned long long>(i), e.tid, caller);
    if (++found >= 8) break;
  }
  if (!found) fprintf(stderr, "[free] no earlier free in the last %u\n",
                      static_cast<unsigned>(kFreeHistory));
  fflush(stderr);
}

}  // namespace

void ArcFree(void* p) {
  int state = g_leak_state.load(std::memory_order_relaxed);
  if (state < 0) {
    state = getenv("ARC_LEAK_FREE") != nullptr ? 1 : 0;
    g_leak_state.store(state, std::memory_order_relaxed);
  }
  if (state == 1) return;
  if (FreeHistoryEnabled()) FreeHistoryRecord(p);
  if (FreeGuardEnabled() && p && !BlockStillAllocated(p)) {
    static std::atomic<int> reported{0};
    if (reported.fetch_add(1, std::memory_order_relaxed) < 8) {
      fprintf(stderr, "[free] block not allocated: %p%s\n", p,
              IsCaughtExceptionBlock(p) ? " (the caught exception)" : "");
      // The heap's bookkeeping for a live block starts at p-0x10. A header that
      // still says "in use, this size" means the block was freed through some
      // path that bypassed this one; a header full of something else means it
      // was written past.
      const uint64_t* header = reinterpret_cast<const uint64_t*>(p) - 4;
      fprintf(stderr, "[free] header %p:", static_cast<const void*>(header));
      for (int i = 0; i < 6; ++i)
        fprintf(stderr, " %016llx",
                static_cast<unsigned long long>(header[i]));
      fprintf(stderr, "\n");
      ReportFreeHistory(p);
      fflush(stderr);
    }
    return;
  }
  {
    static const bool trace = getenv("ARC_FREE_TRACE") != nullptr;
    if (trace && IsCaughtExceptionBlock(p)) {
      fprintf(stderr, "[free] exception block %p (object %p)\n", p,
              reinterpret_cast<void*>(arc_eh_object()));
      fflush(stderr);
    }
  }
  free(p);
}

// malloc, so that what it hands back can be zeroed.
//
// Bionic grows its heap by mapping pages, and a fresh page is zero. A program
// that never shrinks -- a game loading its content -- therefore sees zeroed
// memory from most of its allocations, and shipped code can depend on that
// without anybody noticing, because on the device it is true. This heap hands
// back whatever the last owner left. ARC_ZERO_HEAP makes the two agree.
//
// ponytail: zeroes everything rather than working out who needs it. The cost
// is one memset per allocation; if that ever shows up in a profile, the answer
// is to find the type that depends on it, not to make this cleverer.
void* ArcMalloc(size_t n) {
  static const bool zero = getenv("ARC_ZERO_HEAP") != nullptr;
  return zero ? calloc(1, n) : malloc(n);
}

void* ArcRealloc(void* p, size_t n) {
  static const bool zero = getenv("ARC_ZERO_HEAP") != nullptr;
  if (!zero) return realloc(p, n);
  // Only the part past the old contents is new, and only that part may be
  // zeroed: realloc's contract is that everything up to the old size survives.
#if defined(_WIN32)
  const size_t had = p ? _msize(p) : 0;
#else
  const size_t had = p ? malloc_usable_size(p) : 0;
#endif
  void* q = realloc(p, n);
  if (q && n > had) memset(static_cast<char*>(q) + had, 0, n - had);
  return q;
}

// A pure virtual call, answered instead of aborted.
//
// libc++ prints "Pure virtual function called!" and aborts, which is the right
// thing when the program is at fault. Here the program is a shipped binary and
// the fault is ours: an object it still has listed has had its derived
// destructor run, so its vptr is its abstract base's and slot after slot is
// this handler. The engine hits it while walking its consumable registry,
// calling getName() on 1525 entries to find one by name -- and eighteen of
// those entries are dead.
//
// Aborting there ends the boot. Answering with an empty string does not: the
// caller strcmp()s the name it got against the one it wants, does not match,
// and moves on to the next entry. The one it wants is still in the list.
//
// ponytail: an empty string is the right answer for the getName() this is
// actually hit from, and a plausible-but-wrong one for any other pure virtual
// -- a caller expecting a number gets a pointer-sized one. So it is off unless
// ARC_PURE_VIRTUAL=skip asks for it, and it says how many times it answered,
// because a boot that needed this a thousand times is not a boot that worked.
// The fix is to stop the engine destroying registered objects; this is what
// makes the rest of the boot reachable while that is still true.
const char* PureVirtual() {
  static std::atomic<long> answered{0};
  const long n = ++answered;
  if (n <= 8 || n % 1000 == 0) {
    // Which call site, because the answer is only right for one of them.
    // An empty string suits the getName() walk this was written for and is a
    // pointer-shaped lie to anything else -- and the run that ended three
    // calls after these appeared was tearing a town down, not walking the
    // consumable registry. Naming the caller is what tells those apart.
    fprintf(stderr,
            "[pure] a pure virtual was called on a dead object; answered with"
            " an empty string (%ld so far)\n",
            n);
    const size_t depth = arc_frame_count();
    for (size_t i = 0; i < depth && i < 4; ++i)
      fprintf(stderr, "[pure]     from %#llx\n",
              static_cast<unsigned long long>(arc_frame_at(i)));
  }
  return "";
}

// strrchr and atoi, given a name that is not there.
//
// FetchMTXItems builds a store request by taking each item's id and reading
// the number after its last dot: atoi(strrchr(id, '.') + 1), falling back to
// atoi(id) when there is no dot. One of this catalogue's items has no id at
// all, so both calls get a null pointer -- and Bionic's versions would fault
// on that too, which says the item is meant to have an id and ours does not.
//
// The store is not on the way into the town, so the honest options are to
// crash on the way past it or to let it produce a zero. These produce the
// zero: strrchr answers "no dot", which is a case the caller already handles,
// and atoi answers 0, which is what it answers for any string without digits.
//
// ponytail: two functions, not a null-safe libc. Off unless
// ARC_NULL_SAFE_STR is set, and each one says so the first time, because a
// null name is a missing catalogue and that is worth fixing rather than
// tolerating forever.
bool NullSafeStrings() {
  static const bool on = getenv("ARC_NULL_SAFE_STR") != nullptr;
  return on;
}

void SaidNull(const char* fn) {
  static std::mutex lock;
  static std::vector<const char*> said;
  std::lock_guard<std::mutex> held(lock);
  for (const char* p : said)
    if (strcmp(p, fn) == 0) return;
  said.push_back(fn);
  fprintf(stderr, "[str] %s was given a null pointer; answering as if the"
                  " string were empty\n", fn);
}

char* StrrchrSafe(const char* s, int c) {
  if (!s) {
    if (!NullSafeStrings()) return const_cast<char*>(strrchr(s, c));
    SaidNull("strrchr");
    return nullptr;
  }
  return const_cast<char*>(strrchr(s, c));
}

int AtoiSafe(const char* s) {
  if (!s) {
    if (!NullSafeStrings()) return atoi(s);
    SaidNull("atoi");
    return 0;
  }
  return atoi(s);
}

#if defined(ARC_HAVE_ZLIB)
uint64_t ZlibCrc32(uint64_t crc, const void* buf, uint32_t len) {
  return crc32(static_cast<uLong>(crc), static_cast<const Bytef*>(buf), len);
}
#endif

// The guest reads `environ` as a data symbol, so the import needs an address
// rather than a function. `environ` is a variable of type char**, not the
// array itself: the GOT slot holds its address, the first load reads the
// array pointer, and the second reads the first string. Exporting the array
// directly made the guest treat the first element as the array and dereference
// null -- an empty environment must still have the extra indirection.
char* g_environ_entries[] = {nullptr};
char** g_environ = g_environ_entries;

// The guest is a Linux target, and it is not only libc that believes so: the
// managed runtime's `Path` uses Unix rules. A host path such as
// `C:\Temp` does not start with '/', so `Path.Combine` treats it as relative
// and appends it -- which turned a temp directory into a component of a path
// that already had two other roots. Answer the environment questions that
// feed those paths with the guest's own virtual root, and let the file shim
// translate it. Everything else goes to the host unchanged, because a title
// asks for its own variables by name and a table here would be guessing.
const char* ShimGetenv(const char* name) {
  if (!name) return nullptr;
  if (strcmp(name, "TMPDIR") == 0 || strcmp(name, "TEMP") == 0 ||
      strcmp(name, "TMP") == 0)
    return "/androidrecomp/tmp";
  if (strcmp(name, "HOME") == 0 || strcmp(name, "USERPROFILE") == 0)
    return "/androidrecomp/storage";
  if (strcmp(name, "PWD") == 0) return "/androidrecomp/cwd";
  if (strcmp(name, "LANG") == 0 || strcmp(name, "LC_ALL") == 0)
    return "en_US.UTF-8";
  const char* v = getenv(name);
  // This is a hot import: asking the CRT for the trace flag on every lookup
  // doubled the cost of every guest getenv. Read both flags once.
  static const bool trace = getenv("ARC_TRACE_ENV") != nullptr;
  static const bool trace_all = getenv("ARC_TRACE_ENV_ALL") != nullptr;
  if (trace_all || (trace && strncmp(name, "UNITY_", 6) == 0)) {
    static int shown;
    if (!trace_all || shown++ < 5000)
      fprintf(stderr, "[env] %s -> %s\n", name, v ? v : "(null)");
  }
  return v;
}

const Entry kExplicit[] = {
    {"environ", reinterpret_cast<void*>(&g_environ)},
    {"getenv", reinterpret_cast<void*>(&ShimGetenv)},
    {"strrchr", reinterpret_cast<void*>(&StrrchrSafe)},
    {"atoi", reinterpret_cast<void*>(&AtoiSafe)},
    {"free", reinterpret_cast<void*>(&ArcFree)},
    {"malloc", reinterpret_cast<void*>(&ArcMalloc)},
    {"realloc", reinterpret_cast<void*>(&ArcRealloc)},
    {"alcOpenDevice", reinterpret_cast<void*>(&AlcOpenDevice)},
    {"alcCloseDevice", reinterpret_cast<void*>(&AlcCloseDevice)},
    {"alcCreateContext", reinterpret_cast<void*>(&AlcCreateContext)},
    {"alcMakeContextCurrent", reinterpret_cast<void*>(&AlcMakeContextCurrent)},
    {"alcGetCurrentContext", reinterpret_cast<void*>(&AlcGetCurrentContext)},
    {"alcGetContextsDevice", reinterpret_cast<void*>(&AlcGetContextsDevice)},
    {"alcDestroyContext", reinterpret_cast<void*>(&AlcDestroyContext)},
    {"alcProcessContext", reinterpret_cast<void*>(&AlcProcessContext)},
    {"alcSuspendContext", reinterpret_cast<void*>(&AlcSuspendContext)},
    {"alcGetError", reinterpret_cast<void*>(&AlcGetError)},
    {"alcGetString", reinterpret_cast<void*>(&AlcGetString)},
    {"alGetError", reinterpret_cast<void*>(&AlGetError)},
    {"alGetString", reinterpret_cast<void*>(&AlGetString)},
    {"alGenBuffers", reinterpret_cast<void*>(&AlGenBuffers)},
    {"alGenSources", reinterpret_cast<void*>(&AlGenSources)},
    {"alDeleteBuffers", reinterpret_cast<void*>(&AlDeleteBuffers)},
    {"alDeleteSources", reinterpret_cast<void*>(&AlDeleteSources)},
    {"alBufferData", reinterpret_cast<void*>(&AlBufferData)},
    {"alDistanceModel", reinterpret_cast<void*>(&AlDistanceModel)},
    {"alListener3f", reinterpret_cast<void*>(&AlListener3f)},
    {"alListenerf", reinterpret_cast<void*>(&AlListenerf)},
    {"alListenerfv", reinterpret_cast<void*>(&AlListenerfv)},
    {"alGetListener3f", reinterpret_cast<void*>(&AlGetListener3f)},
    {"alSource3f", reinterpret_cast<void*>(&AlSource3f)},
    {"alSourcef", reinterpret_cast<void*>(&AlSourcef)},
    {"alSourcei", reinterpret_cast<void*>(&AlSourcei)},
    {"alGetSourcef", reinterpret_cast<void*>(&AlGetSourcef)},
    {"alGetSourcefv", reinterpret_cast<void*>(&AlGetSourcefv)},
    {"alGetSourcei", reinterpret_cast<void*>(&AlGetSourcei)},
    {"alSourcePlay", reinterpret_cast<void*>(&AlSourcePlay)},
    {"alSourcePause", reinterpret_cast<void*>(&AlSourcePause)},
    {"alSourceStop", reinterpret_cast<void*>(&AlSourceStop)},
    {"alSourceRewind", reinterpret_cast<void*>(&AlSourceRewind)},
    {"alSourceQueueBuffers", reinterpret_cast<void*>(&AlSourceQueueBuffers)},
    {"alSourceUnqueueBuffers",
     reinterpret_cast<void*>(&AlSourceUnqueueBuffers)},

    {"__android_log_print", reinterpret_cast<void*>(&AndroidLogPrint)},
    {"__android_log_vprint", reinterpret_cast<void*>(&AndroidLogVPrint)},
    {"__android_log_write", reinterpret_cast<void*>(&AndroidLogWrite)},
    {"__android_log_assert", reinterpret_cast<void*>(&AndroidLogAssert)},
    {"__memcpy_chk", reinterpret_cast<void*>(&MemcpyChk)},
    {"__memmove_chk", reinterpret_cast<void*>(&MemmoveChk)},
    {"__strcpy_chk", reinterpret_cast<void*>(&StrcpyChk)},
    {"__strcat_chk", reinterpret_cast<void*>(&StrcatChk)},
    {"__strlen_chk", reinterpret_cast<void*>(&StrlenChk)},
    {"__strchr_chk", reinterpret_cast<void*>(&StrchrChk)},
    {"__vsnprintf_chk", reinterpret_cast<void*>(&VsnprintfChk)},
    {"__assert2", reinterpret_cast<void*>(&Assert2)},
    {"__stack_chk_fail", reinterpret_cast<void*>(&StackChkFail)},
    {"__stack_chk_guard", reinterpret_cast<void*>(&g_stack_chk_guard)},
    {"__errno", reinterpret_cast<void*>(&Errno)},
    {"__cxa_atexit", reinterpret_cast<void*>(&CxaAtexit)},
    {"__cxa_finalize", reinterpret_cast<void*>(&CxaFinalize)},
    {"exit", reinterpret_cast<void*>(&GuestExit)},
    {"_exit", reinterpret_cast<void*>(&GuestExit)},
    {"_Exit", reinterpret_cast<void*>(&GuestExit)},
    {"abort", reinterpret_cast<void*>(&GuestAbort)},

#if defined(ARC_HAVE_ZLIB)
    // Statically linked libraries bind by explicit address: they are inside
    // our own executable, so a by-name runtime lookup cannot see them. This is
    // the pattern OpenAL and the GL loader follow too.
    //
    // crc32 is wrapped rather than bound directly: Bionic's uLong is 64 bits
    // and the host's is 32, so the return value needs zero-extension before
    // the guest reads the full register.
    {"zlibVersion", reinterpret_cast<void*>(&zlibVersion)},
    {"crc32", reinterpret_cast<void*>(&ZlibCrc32)},
#endif
};

// --- layer 2: same function, different name on this host --------------------
// Every entry here must be ABI-identical, not merely similar. `mkdir` is the
// cautionary case: Bionic takes (path, mode), the Microsoft CRT's `_mkdir`
// takes (path). Aliasing those would compile, link, run, and corrupt the
// stack. Anything whose signature differs is deliberately left unresolved so
// it shows up in the work list and gets a real wrapper.
struct Alias {
  const char* from;
  const char* to;
};

const Alias kAliases[] = {
#if defined(_WIN32)
    {"strdup", "_strdup"},
    {"strcasecmp", "_stricmp"},
    {"strncasecmp", "_strnicmp"},
    {"fileno", "_fileno"},
    {"isatty", "_isatty"},
    {"unlink", "_unlink"},
    {"putenv", "_putenv"},
    {"fdopen", "_fdopen"},
    // off_t is 64-bit on Android, which is what _fseeki64 takes.
    {"fseeko", "_fseeki64"},
    {"ftello", "_ftelli64"},
#endif
};

// --- layer 3: the host C runtime -------------------------------------------

void* HostLookup(const char* name) {
#if defined(_WIN32)
  // The UCRT is where the standard C library actually lives; the
  // api-ms-win-crt-* names are forwarders onto it.
  static HMODULE ucrt = GetModuleHandleA("ucrtbase.dll")
                            ? GetModuleHandleA("ucrtbase.dll")
                            : LoadLibraryA("ucrtbase.dll");
  return ucrt ? reinterpret_cast<void*>(GetProcAddress(ucrt, name)) : nullptr;
#else
  return dlsym(RTLD_DEFAULT, name);
#endif
}

}  // namespace

// The guest ending the process is a result, not a crash. Left to resolve
// through the host C runtime, a title that decides initialisation has failed
// simply terminates us -- no message, no fault, nothing to distinguish it from
// a lifter bug. Intercepting it turns "the process died" into "the engine gave
// up, and here is the code it gave up with".
//
// Outside the anonymous namespace because the syscall dispatcher answers
// `exit_group` with this too: libil2cpp issues that syscall directly in places,
// and that path used to end a run with nothing said at all.
[[noreturn]] void GuestExit(int status) {
  char msg[64];
  snprintf(msg, sizeof(msg), "the guest called exit(%d)", status);
  arc_trap(nullptr, msg);
  abort();  // only reached if no recovery point is armed
}

// Names the host must answer even though a lifted dependency defines them.
//
// The resolver prefers lifted code for everything, and rightly: a guest's own
// implementation is the one the guest was built against. The exception is a
// function whose whole job is to end the process.
void ShimStopLeaking() {
  // Blocks freed while leaking stay leaked: they may still be pointed at by
  // whatever the walk left dangling, and handing them back now is the exact
  // hazard the knob existed to avoid. Only frees from here on are honoured.
  if (g_leak_state.exchange(0, std::memory_order_relaxed) == 1)
    fprintf(stderr, "[shim] recycling freed memory from now on\n");
}

// A C++ exception, at the throw, because nothing can catch it.
//
// No frame here unwinds. Lifted code carries no exception tables and no
// personality routine, so once libc++ hands an exception to
// _Unwind_RaiseException there is nowhere to resume and it calls
// std::terminate. The terminate handler is worse than useless: it is a word
// the guest's own runtime never managed to install, so instead of an abort
// message the boot branches to whatever that word happened to hold. The
// report then says "indirect branch to 0x1b2, neither lifted nor a known
// import", which names neither the throw nor the type.
//
// These two names are the ones whose whole job is to end the process, so the
// host answers them. A throw is named where it happens -- type, object, and
// the lifted frames that led there -- and terminate is caught on the way out
// for the paths that reach it without a preceding throw.
namespace {

bool CanRead(uint64_t address, size_t size) {
#if defined(_WIN32)
  MEMORY_BASIC_INFORMATION mbi;
  if (!address || !size ||
      !VirtualQuery(reinterpret_cast<const void*>(address), &mbi, sizeof mbi))
    return false;
  if (mbi.State != MEM_COMMIT || (mbi.Protect & PAGE_GUARD)) return false;
  return (mbi.Protect &
          (PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY |
           PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE |
           PAGE_EXECUTE_WRITECOPY)) != 0;
#else
  return address != 0 && size != 0;
#endif
}

const char* TypeName(uint64_t type) {
  if (!type || !CanRead(type + 8, 8)) return 0;
  const uint64_t p = arc_ld64(type + 8);
  if (!p || !CanRead(p, 1)) return 0;
  return reinterpret_cast<const char*>(p);
}

void LiftedFrames(const char* tag) {
  const size_t depth = arc_frame_count();
  for (size_t i = 0; i < depth && i < 16; ++i)
    fprintf(stderr, "[cxa]     %s from %#llx\n", tag,
            static_cast<unsigned long long>(arc_frame_at(i)));
}

void ReportCxaThrow(Arm64Ctx* c) {
  const uint64_t object = c->x[0];
  const uint64_t type = c->x[1];
  const char* name = TypeName(type);
  fprintf(stderr,
          "[cxa] uncaught C++ exception: type=%s object=%#llx type_info=%#llx\n",
          name ? name : "?", static_cast<unsigned long long>(object),
          static_cast<unsigned long long>(type));
  LiftedFrames("#");
  arc_trap(c, "uncaught C++ exception");
}

void ReportTerminate(Arm64Ctx* c) {
  fprintf(stderr, "[cxa] std::terminate reached\n");
  LiftedFrames("#");
  arc_trap(c, "std::terminate");
}

}  // namespace

uint64_t ShimOverride(const char* name) {
  static const char* const how = getenv("ARC_PURE_VIRTUAL");
  if (how && strcmp(how, "skip") == 0 &&
      strcmp(name, "__cxa_pure_virtual") == 0)
    return reinterpret_cast<uint64_t>(&PureVirtual);
  // C++ exceptions cannot be caught here, so the two ends of that path are
  // answerable names rather than a branch into whatever the handler held.
  if (strcmp(name, "__cxa_throw") == 0) {
    const uint64_t at = reinterpret_cast<uint64_t>(&ReportCxaThrow);
    arc_register_ctx_native(at, "__cxa_throw", &ReportCxaThrow);
    return at;
  }
  if (strcmp(name, "_ZSt9terminatev") == 0) {
    const uint64_t at = reinterpret_cast<uint64_t>(&ReportTerminate);
    arc_register_ctx_native(at, "_ZSt9terminatev", &ReportTerminate);
    return at;
  }
  return 0;
}

uint64_t ShimResolve(const char* name) {
  // First: anything taking a guest va_list. Several of these also exist as
  // plain forwarders elsewhere in the shim, and forwarding is the wrong answer
  // -- the argument layouts do not match.
  // Before all of them: the imports that take or return a float. Those
  // cannot go through the integer bridge the rest of these use, so they are
  // bound as context-taking natives and have to win over any plain forwarder
  // of the same name elsewhere in the shim.
  if (uint64_t a = ShimResolveMath(name)) return a;
  if (uint64_t a = ShimResolveVarargs(name)) return a;
  if (uint64_t a = ShimResolvePthread(name)) return a;
  if (uint64_t a = ShimResolvePosix(name)) return a;
  if (uint64_t a = ShimResolveGL(name)) return a;
  if (uint64_t a = ShimResolveEGL(name)) return a;
  if (uint64_t a = ShimResolveZlib(name)) return a;
  if (uint64_t a = ShimResolveAsset(name)) return a;
  if (uint64_t a = ShimResolveFile(name)) return a;
  if (uint64_t a = ShimResolveSys(name)) return a;

  for (const Entry& e : kExplicit)
    if (strcmp(e.name, name) == 0) return reinterpret_cast<uint64_t>(e.fn);

  for (const Alias& a : kAliases)
    if (strcmp(a.from, name) == 0)
      return reinterpret_cast<uint64_t>(HostLookup(a.to));

  return reinterpret_cast<uint64_t>(HostLookup(name));
}

size_t ShimExplicitCount() { return sizeof(kExplicit) / sizeof(kExplicit[0]); }

// --- cached readability probes ---------------------------------------------
//
// Measured on this port with the title parked on the download screen: one
// thread spent 0.92 of a core inside NtQueryVirtualMemory, and the same was
// true of the load phase before it. Two condition-variable waits and one futex
// wait each asked about memory that had not moved since the previous call.
//
// Region granularity is the point. VirtualQuery answers for a whole committed
// allocation, so a worker alternating between its condition variable, the
// frame record beside it and the futex word inside it hits the same two or
// three entries forever. Four per thread covers that; replacement is
// round-robin, and a miss simply asks the kernel again.
namespace {

struct GuestProbe {
  uintptr_t base;
  uintptr_t end;
  uint32_t generation;
};

thread_local GuestProbe g_probes[8];
thread_local unsigned g_probe_next;

std::atomic<uint32_t> g_memory_generation{1};

}  // namespace

void GuestMemoryReleased() {
  g_memory_generation.fetch_add(1, std::memory_order_relaxed);
}

bool GuestRangeReadable(uint64_t address, size_t size) {
#if defined(_WIN32)
  if (!address || !size) return false;
  const uint64_t end = address + size;
  if (end < address) return false;
  const uint32_t generation = g_memory_generation.load(std::memory_order_relaxed);
  for (const GuestProbe& p : g_probes)
    if (p.generation == generation && address >= p.base && end <= p.end)
      return true;
  MEMORY_BASIC_INFORMATION mbi;
  if (!VirtualQuery(reinterpret_cast<const void*>(address), &mbi, sizeof mbi))
    return false;
  if (mbi.State != MEM_COMMIT || (mbi.Protect & PAGE_GUARD)) return false;
  if (!(mbi.Protect & (PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY |
                       PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE |
                       PAGE_EXECUTE_WRITECOPY)))
    return false;
  const uintptr_t base = reinterpret_cast<uintptr_t>(mbi.BaseAddress);
  const uintptr_t limit = base + mbi.RegionSize;
  if (end > limit) return false;
  g_probes[g_probe_next++ & 7] = GuestProbe{base, limit, generation};
  return true;
#else
  return address != 0 && size != 0;
#endif
}

uint64_t ShimHandOut(uint64_t address, const char* name) {
  if (!address) return 0;
  // The dispatcher keeps the name pointer rather than a copy of the string,
  // and the caller's is the guest's own memory, so it is interned here. A
  // deque and not a vector: a vector reallocates, and a short string keeps its
  // characters inside the object itself, so every pointer handed out earlier
  // would be left dangling. GL entry point names are all short.
  static std::mutex lock;
  static std::deque<std::string> interned;
  const char* kept = "handed out at run time";
  if (name && *name) {
    std::lock_guard<std::mutex> held(lock);
    interned.emplace_back(name);
    kept = interned.back().c_str();
  }
  arc_register_native(address, kept);
  return address;
}

}  // namespace arc
