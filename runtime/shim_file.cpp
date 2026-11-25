// File I/O, directories and memory mapping.
//
// This is the layer where an Android ABI actually has to be reproduced rather
// than forwarded. The host has `open`, `stat` and `readdir` too, but they are
// not the same functions: Bionic's `O_CREAT` is 0100 and the Microsoft CRT's is
// 0x100, and Bionic's `struct stat` and `struct dirent` have layouts of their
// own. The engine allocated those structs and reads the fields back at fixed
// offsets, so we fill them field by field at Bionic's offsets instead of
// letting the host write its own shape into them.
//
// The struct layouts below are the Linux arm64 kernel ABI, which Bionic uses
// unchanged. They are stable and public, but they are also the highest-risk
// thing in this file: if file operations misbehave in a way that looks like
// garbage sizes or modes, `GuestStat` is the first place to look.
//
// Descriptors are the host's own, because every call that consumes one routes
// through this file.

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <random>
#include <set>
#include <string>

#include <map>
#include "shim.h"
#include "arm64_context.h"

extern "C" int arc_image_owns(uint64_t address);

#if defined(_WIN32)
#include <winsock2.h>
#include <windows.h>
#include <direct.h>
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#else
#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace arc {
namespace {

// --- Bionic / Linux arm64 ABI ----------------------------------------------

struct GuestTimespec {
  int64_t tv_sec;
  int64_t tv_nsec;
};

struct GuestStat {
  uint64_t st_dev;
  uint64_t st_ino;
  uint32_t st_mode;
  uint32_t st_nlink;
  uint32_t st_uid;
  uint32_t st_gid;
  uint64_t st_rdev;
  uint64_t __pad1;
  int64_t st_size;
  int32_t st_blksize;
  int32_t __pad2;
  int64_t st_blocks;
  GuestTimespec st_atim;
  GuestTimespec st_mtim;
  GuestTimespec st_ctim;
  uint32_t __unused4;
  uint32_t __unused5;
};
static_assert(sizeof(GuestStat) == 128, "Linux arm64 struct stat is 128 bytes");

struct GuestDirent {
  uint64_t d_ino;
  int64_t d_off;
  uint16_t d_reclen;
  uint8_t d_type;
  char d_name[256];
};

constexpr uint32_t kGuestIfmt = 0170000;
constexpr uint32_t kGuestIfreg = 0100000;
constexpr uint32_t kGuestIfdir = 0040000;
constexpr uint8_t kGuestDtDir = 4;
constexpr uint8_t kGuestDtReg = 8;

// Bionic's open flags are octal Linux values and share only the low two bits
// with the Microsoft CRT's. Translating them is not optional: passing Bionic's
// O_CREAT (64) straight to _open sets _O_APPEND|... and silently does the wrong
// thing rather than failing.
constexpr int kGuestOCreat = 0100;
constexpr int kGuestOExcl = 0200;
constexpr int kGuestOTrunc = 01000;
constexpr int kGuestOAppend = 02000;

int TranslateOpenFlags(int guest_flags) {
#if defined(_WIN32)
  int host = guest_flags & 3;  // RDONLY/WRONLY/RDWR agree
  if (guest_flags & kGuestOCreat) host |= _O_CREAT;
  if (guest_flags & kGuestOExcl) host |= _O_EXCL;
  if (guest_flags & kGuestOTrunc) host |= _O_TRUNC;
  if (guest_flags & kGuestOAppend) host |= _O_APPEND;
  // Android has no text mode; a translated newline would corrupt assets.
  return host | _O_BINARY;
#else
  return guest_flags;
#endif
}

#if defined(_WIN32)
void FillStat(const struct __stat64& in, GuestStat* out) {
  memset(out, 0, sizeof(*out));
  out->st_dev = in.st_dev;
  out->st_ino = in.st_ino;
  out->st_nlink = in.st_nlink;
  out->st_uid = in.st_uid;
  out->st_gid = in.st_gid;
  out->st_rdev = in.st_rdev;
  out->st_size = in.st_size;
  out->st_blksize = 4096;
  out->st_blocks = (in.st_size + 511) / 512;
  out->st_atim.tv_sec = in.st_atime;
  out->st_mtim.tv_sec = in.st_mtime;
  out->st_ctim.tv_sec = in.st_ctime;
  // The CRT reports _S_IFDIR/_S_IFREG, which happen to match Linux's values,
  // but the permission bits do not survive: synthesise something sane.
  out->st_mode = (in.st_mode & _S_IFDIR) ? (kGuestIfdir | 0755)
                                         : (kGuestIfreg | 0644);
}
#endif

// Which paths the engine actually asks for, and whether they were there. A
// title that cannot find its own files says so in its own words much later, if
// at all -- and by then the name it looked for is the one thing missing from
// the account.
bool TracingFiles() {
  static const bool on = getenv("ARC_TRACE_FILES") != nullptr;
  return on;
}

void TraceFile(const char* what, const char* path, bool found) {
  if (!TracingFiles()) return;
  fprintf(stderr, "[file] %-8s %-5s %s\n", what, found ? "ok" : "MISS",
          path ? path : "(null)");
  // The case-sensitivity probe is the first place a guest-visible path is
  // built out of more than one root. When it fails, the interesting thing is
  // which managed method asked: the answer names the subsystem, and the call
  // chain is what the shadow stack was added for.
  if (path && strstr(path, "CASESENSITIVETEST")) {
    const size_t depth = arc_main_shadow_depth();
    fprintf(stderr, "[file] shadow depth=%zu\n", depth);
    for (size_t i = 0; i < depth && i < 24; ++i)
      fprintf(stderr, "  f[%zu] %#llx\n", i,
              static_cast<unsigned long long>(arc_main_shadow_at(i)));
  }
}

// Which file a stream is, so that closing it can be traced by name.
//
// A trace of opens with no closes answers "did it find the file" and nothing
// about lifetime, which is the question when a fault looks like something
// reading through a handle another part of the engine has already let go.
// Only kept while tracing: otherwise it is a map insertion per fopen for
// nobody's benefit.
std::mutex g_open_lock;
std::map<FILE*, std::string> g_open_paths;

void RememberOpen(FILE* f, const char* path) {
  if (!TracingFiles() || !f) return;
  std::lock_guard<std::mutex> held(g_open_lock);
  g_open_paths[f] = path ? path : "(null)";
}

void TraceCloseImpl(FILE* f) {
  if (!TracingFiles()) return;
  std::string path = "(never opened here)";
  {
    std::lock_guard<std::mutex> held(g_open_lock);
    auto it = g_open_paths.find(f);
    if (it != g_open_paths.end()) {
      path = it->second;
      g_open_paths.erase(it);
    }
  }
  fprintf(stderr, "[file] %-8s %-5s %s\n", "fclose", "ok", path.c_str());
}

}  // namespace

// fclose lives in the posix half of the shim; the map of open paths lives
// here, next to fopen that fills it.
void ShimTraceClose(void* stream) {
  TraceCloseImpl(static_cast<FILE*>(stream));
}

namespace {

std::string HostPath(const char* path);

int StatPath(const char* path, GuestStat* out) {
  const std::string host = HostPath(path);
#if defined(_WIN32)
  struct __stat64 st;
  if (_stat64(host.c_str(), &st) != 0) return -1;
  FillStat(st, out);
  return 0;
#else
  struct stat st;
  if (::stat(host.c_str(), &st) != 0) return -1;
  memset(out, 0, sizeof(*out));
  out->st_mode = st.st_mode;
  out->st_size = st.st_size;
  out->st_ino = st.st_ino;
  out->st_nlink = st.st_nlink;
  out->st_blksize = st.st_blksize;
  out->st_blocks = st.st_blocks;
  out->st_mtim.tv_sec = st.st_mtime;
  return 0;
#endif
}

int Stat(const char* path, GuestStat* out) {
  const int rc = StatPath(path, out);
  TraceFile("stat", path, rc == 0);
  return rc;
}
int Lstat(const char* path, GuestStat* out) { return Stat(path, out); }

int Fstat(int fd, GuestStat* out) {
#if defined(_WIN32)
  struct __stat64 st;
  if (_fstat64(fd, &st) != 0) return -1;
  FillStat(st, out);
  return 0;
#else
  struct stat st;
  if (::fstat(fd, &st) != 0) return -1;
  memset(out, 0, sizeof(*out));
  out->st_mode = st.st_mode;
  out->st_size = st.st_size;
  out->st_blksize = st.st_blksize;
  return 0;
#endif
}

// `/dev/urandom` is not a file to be translated; it is the interface to the
// system's entropy, and the host has its own. Without it the C++ runtime's
// std::random_device throws out of a *static initialiser* -- so the engine
// terminates before anything runs, and the only clue is an abort message
// several frames into the terminate handler. Worth an explicit answer.
constexpr int kRandomFd = 0x7000000;

bool IsRandomDevice(const char* path) {
  return path && (strcmp(path, "/dev/urandom") == 0 ||
                  strcmp(path, "/dev/random") == 0);
}

void FillRandom(void* buf, size_t n) {
  static std::mutex lock;
  static std::random_device source;
  std::lock_guard<std::mutex> held(lock);
  auto* out = static_cast<unsigned char*>(buf);
  size_t i = 0;
  while (i < n) {
    const unsigned int bits = source();
    for (size_t b = 0; b < sizeof(bits) && i < n; ++b, ++i)
      out[i] = static_cast<unsigned char>(bits >> (b * 8));
  }
}

// Bionic accepts mode letters the UCRT does not. "re" is the one that matters
// here: the e is Android's close-on-exec flag, which has no Windows meaning,
// and passing it through makes the UCRT reject the whole mode. Its invalid
// parameter handler then fast-fails the process -- an abort with no trace and
// no guest frame, which looks like a fault in the caller.
//
// Keep only the parts both C libraries agree on: r/w/a, an optional '+', and
// binary. Everything else is dropped rather than forwarded.
void HostFileMode(const char* mode, char out[8]) {
  char first = 'r';
  if (mode && (*mode == 'r' || *mode == 'w' || *mode == 'a')) first = *mode;
  size_t n = 0;
  out[n++] = first;
  if (mode && strchr(mode, '+')) out[n++] = '+';
  out[n++] = 'b';
  out[n] = 0;
}

// Android exposes processor capabilities through procfs and sysfs. A Windows
// Unity sizes its worker pools from the processor lines in /proc/cpuinfo and
// the present/online masks, not from sysconf. A single processor line leaves
// the asset pipeline with one worker and lets a job starve the thread that
// would complete it, so the file has to describe the same cores CoreCount()
// admits to.
static const std::string& CpuInfoText() {
  static const std::string text = [] {
    std::string s;
    const int64_t cores = arc::HostCoreCount();
    for (int64_t i = 0; i < cores; ++i) {
      s += "processor\t: " + std::to_string(i) + "\n";
      s += "BogoMIPS\t: 38.40\n";
      s += "Features\t: fp asimd evtstrm\n";
      s += "CPU implementer\t: 0x41\n";
      s += "CPU architecture: 8\n";
      s += "CPU variant\t: 0x0\n";
      s += "CPU part\t: 0xd0c\n";
      s += "CPU revision\t: 1\n\n";
    }
    s += "Hardware\t: AArch64\n";
    return s;
  }();
  return text;
}

static const std::string& CpuRangeText() {
  static const std::string text = [] {
    const int64_t cores = arc::HostCoreCount();
    if (cores <= 1) return std::string("0\n");
    return std::string("0-") + std::to_string(cores - 1) + "\n";
  }();
  return text;
}

// The per-core files under /sys/devices/system/cpu. Only cpu0 used to exist, so
// a title that walks the cores -- which the device tier check does, several
// times a second -- found a file for the first core and nothing for the rest,
// and asked again on the next pass. A host has no sysfs, so every core gets the
// same small honest answer; what matters is that the answer is the same shape
// Android's is, for every core the mask above admits to.
static const char* CpuSysfsText(const char* path) {
  constexpr char kPrefix[] = "/sys/devices/system/cpu/cpu";
  constexpr size_t kPrefixLength = sizeof kPrefix - 1;
  if (!path || strncmp(path, kPrefix, kPrefixLength) != 0) return nullptr;
  const char* at = path + kPrefixLength;
  long cpu = 0;
  const char* digits = at;
  while (*at >= '0' && *at <= '9') cpu = cpu * 10 + (*at++ - '0');
  if (at == digits || cpu >= arc::HostCoreCount()) return nullptr;
  if (strcmp(at, "/cpu_capacity") == 0) return "1024\n";
  if (strcmp(at, "/cpufreq/cpuinfo_max_freq") == 0 ||
      strcmp(at, "/cpufreq/scaling_cur_freq") == 0 ||
      strcmp(at, "/cpufreq/scaling_max_freq") == 0) {
    return "3000000\n";
  }
  return nullptr;
}

// host has neither, so these opens return null and the engine's CPU and memory
// probes read "no file" as "no information". Give them small, honest answers:
// the register file and FP/SIMD are what the lifted code itself relies on, and
// deliberately not advertising crypto extensions keeps the guest off paths
// whose instructions this lifter does not execute.
const char* VirtualFileText(const char* path) {
  if (strcmp(path, "/proc/cpuinfo") == 0) return CpuInfoText().c_str();
  if (strcmp(path, "/proc/meminfo") == 0) {
    return
        "MemTotal:        8192000 kB\n"
        "MemFree:         4096000 kB\n"
        "MemAvailable:    6144000 kB\n"
        "Buffers:               0 kB\n"
        "Cached:                0 kB\n"
        "SwapCached:            0 kB\n"
        "SwapTotal:             0 kB\n"
        "SwapFree:              0 kB\n";
  }
  if (strcmp(path, "/proc/version") == 0) {
    return "Linux version 5.10.0-android (recomp) #1 SMP PREEMPT\n";
  }
  // il2cpp's crash reporter and its thread bookkeeping read these before it
  // initialises. A missing /proc/self/stat is treated as a fatal error and the
  // guest calls abort(); the fields are read by index, so the line has to have
  // all 52 of them even though most are zero here.
  if (strcmp(path, "/proc/self/stat") == 0) {
    return
        "1234 (mktrecomp) R 1 1234 1234 0 -1 4194304 0 0 0 0 0 0 0 0 20 0 1 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 17 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0\n";
  }
  if (strcmp(path, "/proc/self/status") == 0) {
    return
        "Name:\tmktrecomp\n"
        "State:\tR (running)\n"
        "Tgid:\t1234\n"
        "Pid:\t1234\n"
        "PPid:\t1\n"
        "Threads:\t1\n"
        "VmSize:\t 2097152 kB\n"
        "VmRSS:\t  524288 kB\n";
  }
  if (strcmp(path, "/sys/devices/system/cpu/present") == 0 ||
      strcmp(path, "/sys/devices/system/cpu/possible") == 0 ||
      strcmp(path, "/sys/devices/system/cpu/online") == 0) {
    return CpuRangeText().c_str();
  }
  if (const char* text = CpuSysfsText(path)) return text;
  return nullptr;
}

// Guest paths are relative to the content root, not to the process directory.
// Unity's il2cpp bootstrap reads "assets/bin/Data/unity_app_guid" and the
// engine's own data files use "bin/Data/..."; the host was started from a
// build directory, so both resolved to nothing. Rewrite those prefixes under
// the asset root. A relative path that already exists in the process
// directory is left alone -- a title may write logs or preferences there.
//
// The guest also sees a small virtual tree rooted at /androidrecomp. It is
// there because the target is Android: the managed runtime's Path uses Unix
// rules, and a Windows absolute path handed to it is "relative" -- so
// Path.Combine appends it instead of replacing, and a path built from two
// independent roots comes out with both of them in it. Keeping every path
// the guest can see under a Unix root keeps its own path semantics intact;
// this function is the only place that knows where those names live on the
// host.
std::string HostPath(const char* path) {
  if (!path) return {};
  std::string p(path);
  const char* root = arc::ShimAssetRoot();
  if (!root || !*root) return p;
  auto under_root = [&](size_t skip) {
    return (std::filesystem::path(root) / p.substr(skip)).generic_string();
  };
  if (p.compare(0, 7, "assets/") == 0) return under_root(7);
  if (p.compare(0, 7, "assets\\") == 0) return under_root(7);
  // A native loader that builds its path from the APK root asks for the
  // asset directory with its leading slash still attached. Treat that the
  // same as the relative form instead of handing it to the host unchanged,
  // where it names a directory outside the content tree.
  if (p.compare(0, 8, "/assets/") == 0 ||
      p.compare(0, 8, "\\assets\\") == 0)
    return under_root(8);
  if (p == "/assets" || p == "\\assets") return root;
  if (p.compare(0, 10, "/bin/Data/") == 0 ||
      p.compare(0, 10, "\\bin\\Data\\") == 0 ||
      p.compare(0, 9, "/Managed/") == 0 ||
      p.compare(0, 9, "\\Managed\\") == 0)
    return under_root(0);
  if (p.compare(0, 9, "bin/Data/") == 0 ||
      p.compare(0, 9, "bin\\Data\\") == 0 ||
      p.compare(0, 8, "Managed/") == 0 ||
      p.compare(0, 8, "Managed\\") == 0)
    return under_root(0);
  const std::filesystem::path host_root(root);
  auto match = [&](const char* prefix) {
    const size_t n = strlen(prefix);
    return p.compare(0, n, prefix) == 0 && (p.size() == n || p[n] == '/');
  };
  auto map = [&](const char* prefix, const std::filesystem::path& base) {
    std::string rest = p.substr(strlen(prefix));
    while (!rest.empty() && rest[0] == '/') rest.erase(0, 1);
    return (base / rest).generic_string();
  };
  // The temporary directory is the one the managed side builds paths from
  // on its own, without asking JNI, so it gets its own writable directory
  // rather than the content root: a probe that creates a directory there
  // should not write into the shipped tree. Android guarantees this
  // directory exists before any code runs; here the first thing that asks
  // for it is a case-sensitivity probe that stats it and treats a miss as
  // fatal, so make it once, on first use.
  if (match("/androidrecomp/tmp")) {
    static std::once_flag once;
    std::call_once(once, [&] {
      std::error_code ec;
      std::filesystem::create_directories(host_root / "tmp", ec);
    });
    return map("/androidrecomp/tmp", host_root / "tmp");
  }
  // The current directory is the content root's parent, which is where the
  // host process started. A title that builds a path from its working
  // directory and then a relative one still lands somewhere writable.
  if (match("/androidrecomp/cwd"))
    return map("/androidrecomp/cwd", host_root.parent_path());
  if (match("/androidrecomp/storage"))
    return map("/androidrecomp/storage", host_root);
  if (match("/androidrecomp/files"))
    return map("/androidrecomp/files", host_root);
  if (match("/androidrecomp/cache"))
    return map("/androidrecomp/cache", host_root);
  if (match("/androidrecomp/obb"))
    return map("/androidrecomp/obb", host_root);
  if (match("/androidrecomp/app"))
    return map("/androidrecomp/app", host_root);
  if (match("/androidrecomp/assets"))
    return map("/androidrecomp/assets", host_root);
  if (match("/androidrecomp"))
    return map("/androidrecomp", host_root);
  const bool relative = p.size() > 1 && p[0] != '/' && p[0] != '\\' &&
                        p[1] != ':';
  if (relative) {
    std::error_code ec;
    if (!std::filesystem::exists(p, ec)) {
      std::filesystem::path candidate = std::filesystem::path(root) / p;
      if (std::filesystem::exists(candidate, ec))
        return candidate.generic_string();
    }
  }
  return p;
}

FILE* VirtualFopen(const char* path, const char* mode) {
  if (!path) return nullptr;
  const bool reading = !mode || mode[0] == 'r';
  if (!reading) return nullptr;
  const char* text = VirtualFileText(path);
  if (!text) return nullptr;
  if (TracingFiles()) fprintf(stderr, "[fopen] virtual text %s\n", path);
  FILE* f = tmpfile();
  if (!f) return nullptr;
  fputs(text, f);
  rewind(f);
  return f;
}

// The same virtual files have to be reachable through open()/read(), not just
// fopen(). il2cpp's thread code opens /proc/self/stat with open() and treats a
// failed descriptor as fatal, so hand back a duplicate of a temporary file
// seeded with the text. The duplicate outlives the FILE* and shares its offset.
int VirtualOpen(const char* path, int flags) {
  if (!path) return -1;
  const int access = flags & 3;  // Bionic O_RDONLY is 0, O_WRONLY 1, O_RDWR 2.
  if (access != 0) return -1;
  const char* text = VirtualFileText(path);
  if (!text) return -1;
  if (TracingFiles()) fprintf(stderr, "[open] virtual text %s\n", path);
  FILE* f = tmpfile();
  if (!f) return -1;
  fputs(text, f);
  rewind(f);
#if defined(_WIN32)
  const int fd = _dup(_fileno(f));
#else
  const int fd = dup(fileno(f));
#endif
  fclose(f);
  return fd;
}

FILE* Fopen(const char* path, const char* mode) {
  char binary[8];
  HostFileMode(mode, binary);
  FILE* f = VirtualFopen(path, mode);
  const std::string host = HostPath(path);
  if (!f) f = fopen(host.c_str(), binary);
  TraceFile("fopen", host.c_str(), f != nullptr);
  RememberOpen(f, host.c_str());
  return f;
}

FILE* Freopen(const char* path, const char* mode, FILE* stream) {
  char binary[8];
  HostFileMode(mode, binary);
  const std::string host = HostPath(path);
  FILE* f = freopen(host.c_str(), binary, stream);
  TraceFile("freopen", host.c_str(), f != nullptr);
  return f;
}

int Open(const char* path, int flags, ...) {
  if (IsRandomDevice(path)) return kRandomFd;
  const int virtual_fd = VirtualOpen(path, flags);
  if (virtual_fd >= 0) return virtual_fd;
  const std::string host = HostPath(path);
#if defined(_WIN32)
  const int fd = _open(host.c_str(), TranslateOpenFlags(flags),
                       _S_IREAD | _S_IWRITE);
#else
  const int fd = ::open(host.c_str(), flags, 0644);
#endif
  TraceFile("open", host.c_str(), fd >= 0);
  return fd;
}
int Open2(const char* path, int flags) { return Open(path, flags); }

#if defined(_WIN32)
// On Linux a socket is a file descriptor and read()/write()/close() work on it.
// On Windows a SOCKET is a different namespace entirely, and the CRT's _read
// on one fails with EBADF -- so an HTTP client that opens a connection, sends
// its request and reads the reply gets a connection it can never read from.
// Nothing reports that as a network error; the caller just times out.
//
// Asking Winsock for the socket type is the cheap way to tell the two apart:
// it answers for a socket and fails for anything else.
//
// ponytail: this assumes a CRT descriptor is never also a valid SOCKET value.
// They are separate namespaces and small descriptors are not socket handles in
// practice, but the two could collide. Keep a set of the descriptors the socket
// shim handed out if that ever bites.
// Sockets the guest has asked to be non-blocking; see Fcntl below.
std::mutex g_nonblock_lock;
std::set<int> g_nonblock;

// The host socket behind a descriptor, or INVALID_SOCKET when it is a file.
//
// A socket the guest opened is not a Winsock handle here: the socket shim
// hands out small guest numbers and keeps the handle to itself, because a
// SOCKET value does not fit in a Bionic fd_set. So the mapping is asked first,
// and the getsockopt probe is only for handles that reached the guest some
// other way.
SOCKET HostSocket(int fd) {
  if (const uint64_t host = arc::ShimHostSocket(fd))
    return static_cast<SOCKET>(host);
  int type = 0;
  int len = sizeof type;
  if (::getsockopt(static_cast<SOCKET>(fd), SOL_SOCKET, SO_TYPE,
                   reinterpret_cast<char*>(&type), &len) != 0)
    return INVALID_SOCKET;
  return static_cast<SOCKET>(fd);
}

bool IsSocket(int fd) { return HostSocket(fd) != INVALID_SOCKET; }
#endif

int Close(int fd) {
  if (fd == kRandomFd) return 0;
  if (arc::ShimCloseEpoll(fd) == 0) return 0;
#if defined(_WIN32)
  if (IsSocket(fd)) {
    {
      std::lock_guard<std::mutex> g(g_nonblock_lock);
      g_nonblock.erase(fd);
    }
    const int rc = ::closesocket(HostSocket(fd));
    arc::ShimTraceSocketIo("close", fd, 0, 0, nullptr);
    arc::ShimForgetGuestSocket(fd);
    return rc;
  }
  return _close(fd);
#else
  return ::close(fd);
#endif
}
int64_t Read(int fd, void* buf, uint64_t n) {
  if (fd == kRandomFd) {
    FillRandom(buf, static_cast<size_t>(n));
    return static_cast<int64_t>(n);
  }
#if defined(_WIN32)
  if (const SOCKET socket = HostSocket(fd); socket != INVALID_SOCKET) {
    const int rc = ::recv(socket, static_cast<char*>(buf),
                          static_cast<int>(n), 0);
    if (rc < 0) ShimNetErrno();
    arc::ShimTraceSocketIo("read", fd, static_cast<int64_t>(n), rc, buf);
    return rc;
  }
  return _read(fd, buf, static_cast<unsigned>(n));
#else
  return ::read(fd, buf, n);
#endif
}
int64_t ReadChk(int fd, void* buf, uint64_t n, uint64_t) {
  return Read(fd, buf, n);
}
int64_t Write(int fd, const void* buf, uint64_t n) {
#if defined(_WIN32)
  if (const SOCKET socket = HostSocket(fd); socket != INVALID_SOCKET) {
    const int rc = ::send(socket, static_cast<const char*>(buf),
                          static_cast<int>(n), 0);
    if (rc < 0) ShimNetErrno();
    arc::ShimTraceSocketIo("write", fd, static_cast<int64_t>(n), rc, buf);
    return rc;
  }
  return _write(fd, buf, static_cast<unsigned>(n));
#else
  return ::write(fd, buf, n);
#endif
}
int64_t Lseek(int fd, int64_t off, int whence) {
#if defined(_WIN32)
  return _lseeki64(fd, off, whence);
#else
  return ::lseek(fd, off, whence);
#endif
}

int Access(const char* path, int mode) {
  const std::string host = HostPath(path);
#if defined(_WIN32)
  // Bionic's X_OK (1) has no Windows meaning and _access rejects it.
  return _access(host.c_str(), mode == 1 ? 0 : mode);
#else
  return ::access(host.c_str(), mode);
#endif
}

// Every component, not just the last one. Both the CRT's _mkdir and POSIX's
// mkdir fail when a parent is missing, and a title asks for the whole path at
// once -- on Android its data directory is built by the framework before any
// native code runs, so the engine has never had to make the parents itself.
// Here nothing built them, so the single-level call fails, the engine believes
// the directory exists because we said the request succeeded, and the first
// file it writes there cannot be opened.
int Mkdir(const char* path, uint32_t) {
  if (!path || !*path) return -1;
  const std::string host = HostPath(path);
  std::error_code ec;
  // Already there is success: mkdir -p semantics, because the caller wants the
  // directory to exist rather than to have been the one that made it.
  if (std::filesystem::is_directory(host, ec)) {
    TraceFile("mkdir", path, true);
    return 0;
  }
  std::filesystem::create_directories(host, ec);
  TraceFile("mkdir", path, !ec);
  return ec ? -1 : 0;
}
int Rmdir(const char* path) {
  const std::string host = HostPath(path);
#if defined(_WIN32)
  return _rmdir(host.c_str());
#else
  return ::rmdir(host.c_str());
#endif
}
int Chdir(const char* path) {
  const std::string host = HostPath(path);
#if defined(_WIN32)
  return _chdir(host.c_str());
#else
  return ::chdir(host.c_str());
#endif
}
char* Getcwd(char* buf, uint64_t n) {
  // The guest's Path class speaks Unix. Handing it the host's drive-letter
  // directory makes every later Combine treat that root as a relative piece,
  // so the same root arrives a second time inside the result. The virtual tree
  // in HostPath already names this directory; report that name instead.
  static const char kCwd[] = "/androidrecomp/cwd";
  if (!buf || n < sizeof kCwd) {
    errno = n ? ERANGE : EINVAL;
    return nullptr;
  }
  memcpy(buf, kCwd, sizeof kCwd);
  return buf;
}
int Ftruncate(int fd, int64_t len) {
#if defined(_WIN32)
  return _chsize_s(fd, len);
#else
  return ::ftruncate(fd, len);
#endif
}
int Truncate(const char* path, int64_t len) {
  int fd = Open(path, 1 /* O_WRONLY */);
  if (fd < 0) return -1;
  int r = Ftruncate(fd, len);
  Close(fd);
  return r;
}
int Dup2(int a, int b) {
#if defined(_WIN32)
  return _dup2(a, b);
#else
  return ::dup2(a, b);
#endif
}
int Fsync(int fd) {
#if defined(_WIN32)
  return _commit(fd);
#else
  return ::fsync(fd);
#endif
}
char* Realpath(const char* path, char* resolved) {
#if defined(_WIN32)
  return _fullpath(resolved, path, 260);
#else
  return ::realpath(path, resolved);
#endif
}

// Permission and ownership have no Windows counterpart worth emulating, and
// nothing in a game port depends on them succeeding.
int Chmod(const char*, uint32_t) { return 0; }
int Fchmod(int, uint32_t) { return 0; }
int Fchmodat(int, const char*, uint32_t, int) { return 0; }
int Fchown(int, uint32_t, uint32_t) { return 0; }
// fcntl was a stub returning success for everything, and that is the one
// answer it must not give. An HTTP stack opens a socket, sets O_NONBLOCK, and
// then connects expecting -1/EINPROGRESS and a writability wait. Reporting
// success while leaving the socket blocking makes connect return 0 instead,
// the stack sees a state its non-blocking path does not have, and it closes
// the socket and tries the next address -- forever, without ever sending a
// byte. From outside it looks exactly like a network that is down.
//
// Only the file-status flags are real, and only for sockets, because Winsock
// is the only thing here with a switch to throw. Windows cannot read the flag
// back, so F_GETFL answers from what we were last told.
constexpr int kGuestFGetfd = 1;
constexpr int kGuestFSetfd = 2;
constexpr int kGuestFGetfl = 3;
constexpr int kGuestFSetfl = 4;
constexpr int kGuestONonblock = 0x800;
constexpr int kGuestORdwr = 2;

bool SetNonblocking(int fd, bool on) {
#if defined(_WIN32)
  u_long mode = on ? 1 : 0;
  if (::ioctlsocket(HostSocket(fd), FIONBIO, &mode) != 0) {
    ShimNetErrno();
    return false;
  }
  std::lock_guard<std::mutex> g(g_nonblock_lock);
  if (on) g_nonblock.insert(fd);
  else g_nonblock.erase(fd);
  return true;
#else
  const int flags = ::fcntl(fd, F_GETFL, 0);
  return ::fcntl(fd, F_SETFL, on ? (flags | O_NONBLOCK)
                                 : (flags & ~O_NONBLOCK)) == 0;
#endif
}

int Fcntl(int fd, int cmd, ...) {
#if defined(_WIN32)
  if (cmd == kGuestFGetfl || cmd == kGuestFSetfl) {
    if (!IsSocket(fd)) return cmd == kGuestFGetfl ? kGuestORdwr : 0;
    if (cmd == kGuestFGetfl) {
      std::lock_guard<std::mutex> g(g_nonblock_lock);
      return kGuestORdwr | (g_nonblock.count(fd) ? kGuestONonblock : 0);
    }
    va_list ap;
    va_start(ap, cmd);
    const int flags = va_arg(ap, int);
    va_end(ap);
    return SetNonblocking(fd, (flags & kGuestONonblock) != 0) ? 0 : -1;
  }
  // F_GETFD/F_SETFD are close-on-exec, and nothing here execs.
  if (cmd == kGuestFGetfd || cmd == kGuestFSetfd) return 0;
#else
  (void)fd;
  (void)cmd;
#endif
  return 0;
}
int Utimes(const char*, const void*) { return 0; }
int Utimensat(int, const char*, const void*, int) { return 0; }
int64_t Pathconf(const char*, int) { return -1; }
int Statvfs(const char*, void* buf) {
  if (buf) memset(buf, 0, 112);  // Linux arm64 struct statvfs
  return 0;
}
int Link(const char*, const char*) { return -1; }
int Symlink(const char*, const char*) { return -1; }
int64_t Readlink(const char*, char*, uint64_t) { return -1; }

// rename(2) replaces the destination; the CRT's rename refuses to. That is not
// a detail: a downloader writes to `name.tmp` and renames into place, and a
// title that checks its own state by looking for the finished file treats the
// refusal as "the download never finished". The host CRT was answering this by
// fallback because the file shim never claimed the name, so every install that
// had to replace an existing file quietly did nothing and left its temporary
// behind -- which is exactly what a stalled "Downloading the latest data"
// looks like from the outside.
static void SetErrnoFromWin32(unsigned long err) {
  switch (err) {
    case ERROR_FILE_NOT_FOUND:
    case ERROR_PATH_NOT_FOUND:
    case ERROR_INVALID_DRIVE:
      errno = ENOENT;
      break;
    case ERROR_ACCESS_DENIED:
    case ERROR_SHARING_VIOLATION:
    case ERROR_LOCK_VIOLATION:
      errno = EACCES;
      break;
    case ERROR_ALREADY_EXISTS:
    case ERROR_FILE_EXISTS:
      errno = EEXIST;
      break;
    case ERROR_DISK_FULL:
    case ERROR_HANDLE_DISK_FULL:
      errno = ENOSPC;
      break;
    default:
      errno = EIO;
      break;
  }
}

int Rename(const char* from, const char* to) {
  if (!from || !*from || !to || !*to) {
    errno = EINVAL;
    return -1;
  }
  const std::string a = HostPath(from);
  const std::string b = HostPath(to);
#if defined(_WIN32)
  // MOVEFILE_COPY_ALLOWED so a rename across volumes still lands, which POSIX
  // permits when the guest does not care about atomicity.
  if (MoveFileExA(a.c_str(), b.c_str(),
                  MOVEFILE_REPLACE_EXISTING | MOVEFILE_COPY_ALLOWED)) {
    TraceFile("rename", to, true);
    return 0;
  }
  SetErrnoFromWin32(GetLastError());
  TraceFile("rename", to, false);
  return -1;
#else
  const int r = ::rename(a.c_str(), b.c_str());
  TraceFile("rename", to, r == 0);
  return r;
#endif
}

int Pipe(int*) { return -1; }

// --- directories -----------------------------------------------------------
// The guest gets an opaque DIR* and a dirent laid out its way, so the host's
// own dirent never reaches it.

struct GuestDir {
#if defined(_WIN32)
  HANDLE find = INVALID_HANDLE_VALUE;
  WIN32_FIND_DATAA data{};
  bool pending = false;
#else
  DIR* dir = nullptr;
#endif
  GuestDirent entry{};
};

void* Opendir(const char* path) {
  if (!path || !*path) return nullptr;
  const std::string host = HostPath(path);
  auto* d = new GuestDir();
#if defined(_WIN32)
  std::string pattern(host);
  while (!pattern.empty() &&
         (pattern.back() == '/' || pattern.back() == '\\'))
    pattern.pop_back();
  pattern += "\\*";
  d->find = FindFirstFileA(pattern.c_str(), &d->data);
  if (d->find == INVALID_HANDLE_VALUE) {
    TraceFile("opendir", path, false);
    delete d;
    return nullptr;
  }
  d->pending = true;
  TraceFile("opendir", path, true);
#else
  d->dir = ::opendir(host.c_str());
  if (!d->dir) {
    delete d;
    return nullptr;
  }
#endif
  return d;
}

void* Readdir(void* handle) {
  auto* d = static_cast<GuestDir*>(handle);
  if (!d) return nullptr;
#if defined(_WIN32)
  if (d->pending)
    d->pending = false;
  else if (!FindNextFileA(d->find, &d->data))
    return nullptr;
  const char* name = d->data.cFileName;
  d->entry.d_type = (d->data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
                        ? kGuestDtDir
                        : kGuestDtReg;
#else
  struct dirent* e = ::readdir(d->dir);
  if (!e) return nullptr;
  const char* name = e->d_name;
  d->entry.d_type = e->d_type;
#endif
  size_t n = strlen(name);
  if (n >= sizeof(d->entry.d_name)) n = sizeof(d->entry.d_name) - 1;
  memcpy(d->entry.d_name, name, n);
  d->entry.d_name[n] = '\0';
  d->entry.d_reclen = static_cast<uint16_t>(sizeof(GuestDirent));
  return &d->entry;
}

int Closedir(void* handle) {
  auto* d = static_cast<GuestDir*>(handle);
  if (!d) return -1;
#if defined(_WIN32)
  if (d->find != INVALID_HANDLE_VALUE) FindClose(d->find);
#else
  if (d->dir) ::closedir(d->dir);
#endif
  delete d;
  return 0;
}

// --- memory ----------------------------------------------------------------
// ponytail: a file mapping is read into freshly allocated memory rather than
// mapped. Correct for the read-only asset loading this is used for, and wrong
// for MAP_SHARED writeback, which nothing here does. Revisit if a title
// actually shares a mapping.

constexpr int kGuestMapAnonymous = 0x20;
void* const kMapFailed = reinterpret_cast<void*>(-1);

void RememberMap(void* p, size_t length);

void* Mmap(void* /*addr*/, uint64_t length, int prot, int flags, int fd,
           int64_t offset) {
#if defined(_WIN32)
  // Android's GC reserves large PROT_NONE spans and then reads them without a
  // matching mprotect; committing the span up front is what keeps those reads
  // inside a valid page. It is commit charge, not resident memory: the GC still
  // touches only the slices it actually uses.
  void* p = VirtualAlloc(nullptr, length, MEM_RESERVE | MEM_COMMIT,
                         PAGE_READWRITE);
#else
  void* p = mmap(nullptr, length, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) p = nullptr;
#endif
  if (!p) return kMapFailed;
  RememberMap(p, static_cast<size_t>(length));
  if (getenv("ARC_TRACE_MMAP"))
    fprintf(stderr, "[mmap] %p %#llx prot=%#x flags=%#x fd=%d\n", p,
            static_cast<unsigned long long>(length), prot, flags, fd);
  if (!(flags & kGuestMapAnonymous) && fd >= 0) {
    Lseek(fd, offset, 0 /* SEEK_SET */);
    uint64_t got = 0;
    while (got < length) {
      int64_t n = Read(fd, static_cast<char*>(p) + got, length - got);
      if (n <= 0) break;
      got += static_cast<uint64_t>(n);
    }
  }
  return p;
}

// A guest mmap is allowed to be released in pieces. VirtualFree with
// MEM_RELEASE only accepts the base of the original reservation, so treating
// every munmap as a release tears down the whole mapping -- including the
// ranges the caller explicitly kept. That is not a leak or a stale pointer in
// the usual sense: the unused half becomes free address space, and the next
// dereference faults in code that was correct on Android. Keep the original
// ranges and decommit only the requested pages, releasing the reservation when
// its base is unmapped in full.
std::mutex g_map_regions_lock;
std::map<uintptr_t, size_t> g_map_regions;

void RememberMap(void* p, size_t length) {
  if (!p || !length) return;
  std::lock_guard<std::mutex> held(g_map_regions_lock);
  g_map_regions[reinterpret_cast<uintptr_t>(p)] = length;
}

int Munmap(void* addr, uint64_t length) {
  if (getenv("ARC_TRACE_MMAP"))
    fprintf(stderr, "[munmap] %p %#llx\n", addr,
            static_cast<unsigned long long>(length));
  // Whatever this releases, a cached "readable" answer about it is now a lie.
  arc::GuestMemoryReleased();
#if defined(_WIN32)
  if (!addr) return -1;
  const uintptr_t at = reinterpret_cast<uintptr_t>(addr);
  {
    std::lock_guard<std::mutex> held(g_map_regions_lock);
    auto it = g_map_regions.upper_bound(at);
    if (it != g_map_regions.begin()) {
      --it;
      const uintptr_t base = it->first;
      const size_t span = it->second;
      if (at >= base && at < base + span) {
        if (at == base && length >= span) {
          VirtualFree(reinterpret_cast<void*>(base), 0, MEM_RELEASE);
          g_map_regions.erase(it);
          return 0;
        }
        // Decommit only the requested part. The reservation, and every page
        // outside the range, stay alive for the caller that still owns them.
        VirtualFree(addr, static_cast<size_t>(length), MEM_DECOMMIT);
        return 0;
      }
    }
  }
  if (arc_image_owns(at) ||
      (length && arc_image_owns(at + length - 1)))
    fprintf(stderr, "[munmap-image] %p %#llx\n", addr,
            static_cast<unsigned long long>(length));
  return VirtualFree(addr, 0, MEM_RELEASE) ? 0 : -1;
#else
  return munmap(addr, static_cast<size_t>(length));
#endif
}

int Mprotect(void* addr, uint64_t length, int) {
  arc::GuestMemoryReleased();
#if defined(_WIN32)
  if (!addr || !length) return 0;
  // mmap may have reserved the range inaccessible; mprotect is how the guest
  // makes it usable again. Committing an already committed page succeeds.
  VirtualAlloc(addr, static_cast<size_t>(length), MEM_COMMIT, PAGE_READWRITE);
#endif
  return 0;
}
void* Mremap(void*, uint64_t, uint64_t, int, ...) { return kMapFailed; }
int Madvise(void*, uint64_t, int) { return 0; }
int Mlock(const void*, uint64_t) { return 0; }
int Getpagesize() { return 4096; }

int PosixMemalign(void** out, uint64_t alignment, uint64_t size) {
#if defined(_WIN32)
  *out = _aligned_malloc(size, alignment);
#else
  if (posix_memalign(out, alignment, size) != 0) *out = nullptr;
#endif
  return *out ? 0 : 12 /* ENOMEM */;
}

struct Entry {
  const char* name;
  void* fn;
};

#define E(sym, fn) {sym, reinterpret_cast<void*>(&fn)}
const Entry kTable[] = {
    E("fopen", Fopen),        E("fopen64", Fopen),
    E("freopen", Freopen),    E("freopen64", Freopen),
    E("open", Open),          E("__open_2", Open2),
    E("close", Close),        E("read", Read),
    E("__read_chk", ReadChk), E("write", Write),
    E("lseek", Lseek),        E("lseek64", Lseek),
    E("stat", Stat),          E("lstat", Lstat),
    E("fstat", Fstat),        E("access", Access),
    E("mkdir", Mkdir),        E("rmdir", Rmdir),
    E("chdir", Chdir),        E("getcwd", Getcwd),
    E("ftruncate", Ftruncate), E("truncate", Truncate),
    E("dup2", Dup2),          E("fsync", Fsync),
    E("realpath", Realpath),  E("chmod", Chmod),
    E("fchmod", Fchmod),      E("fchmodat", Fchmodat),
    E("fchown", Fchown),      E("fcntl", Fcntl),
    E("utimes", Utimes),      E("utimensat", Utimensat),
    E("pathconf", Pathconf),  E("statvfs", Statvfs),
    E("link", Link),          E("symlink", Symlink),
    E("readlink", Readlink),  E("pipe", Pipe),
    E("rename", Rename),

    E("opendir", Opendir),    E("readdir", Readdir),
    E("closedir", Closedir),

    E("mmap", Mmap),          E("munmap", Munmap),
    E("mprotect", Mprotect),  E("mremap", Mremap),
    E("madvise", Madvise),    E("mlock", Mlock),
    E("getpagesize", Getpagesize),
    E("posix_memalign", PosixMemalign),
};
#undef E

}  // namespace

// The same entropy the /dev/urandom stand-in uses, for callers outside this
// file: the syscall shim needs it for getrandom by number.
void ShimFillRandom(void* buf, size_t n) { FillRandom(buf, n); }

uint64_t ShimResolveFile(const char* name) {
  for (const Entry& e : kTable)
    if (strcmp(e.name, name) == 0) return reinterpret_cast<uint64_t>(e.fn);
  return 0;
}

size_t ShimFileCount() { return sizeof(kTable) / sizeof(kTable[0]); }

// The host name for a guest path, by the same rules every file entry point
// uses. A port answering a call that carries a path -- a Java method that copies
// a file, a native entry point that writes beside the app -- has a guest path
// and needs the host one, and re-deriving the virtual tree outside this file is
// how two mappings of the same tree drift apart.
bool ShimGuestPath(const char* guest, char* out, size_t size) {
  if (!guest || !out || !size) return false;
  const std::string host = HostPath(guest);
  if (host.size() >= size) return false;
  memcpy(out, host.c_str(), host.size() + 1);
  return true;
}

}  // namespace arc
