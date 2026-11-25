// Linux syscalls issued directly by guest code.
//
// Most Android libraries call libc, which the shim already answers by name.
// libil2cpp statically links a handful of bionic wrappers that issue `svc #0`
// themselves instead of calling out, so those calls never reach the import
// table. The lifted instruction leaves the syscall number in x8 and the
// arguments in x0..x5, exactly as the architecture's syscall ABI does; this is
// the kernel side of those calls, implemented over the same shims.
//
// The return convention is the raw one: a non-negative result on success and
// -errno on failure. The bionic wrappers around these instructions inspect the
// sign bit and move the negated value into errno themselves, so returning -1
// here (as the host shims do) would be the wrong layer's convention.

#include "arm64_context.h"
#include "arc_signal.h"
#include "shim.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <mutex>

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace {

using OpenFn = int (*)(const char*, int, ...);
using CloseFn = int (*)(int);
using ReadFn = int64_t (*)(int, void*, uint64_t);
using WriteFn = int64_t (*)(int, const void*, uint64_t);
using LseekFn = int64_t (*)(int, int64_t, int);
using StatFn = int (*)(const char*, void*);
using FstatFn = int (*)(int, void*);
using TimeFn = int (*)(void*, void*);

// The host shims return -1 for failure and set host errno. The guest's raw
// syscall contract wants the negated errno, so translate at this boundary.
int64_t NegErrno() {
  const int e = errno;
  return e ? -static_cast<int64_t>(e) : -5;  // EIO when nothing was set
}

template <typename Fn>
Fn Lookup(const char* name) {
  return reinterpret_cast<Fn>(arc::ShimResolve(name));
}

struct GuestTimespec {
  int64_t tv_sec;
  int64_t tv_nsec;
};

// Bionic's condition variables, and the ones Unity links into libunity, are
// implemented on top of the futex syscall. The lift used to answer every futex
// with success, which is harmless for a pure `FUTEX_WAKE` but turns a timed
// wait into a busy loop: the caller re-checks its word forever. These buckets
// are deliberately a hash rather than an exact address map. Futex explicitly
// permits spurious wakeups, so two words landing in one bucket only costs an
// extra re-check, and the table never has to allocate while holding a lock.
// A wait is a node on the stack of the thread that made it, linked into the
// bucket its word hashes to. Waking by address rather than by bucket is what
// makes FUTEX_WAKE(uaddr, 1) mean what the caller thinks it means: a semaphore
// hands out one permit per post, and a notify spent on an unrelated word that
// happens to share the bucket leaves the real waiter parked until the next
// post -- often a whole frame later.
struct FutexWaitNode;

struct FutexBucket {
  std::mutex lock;
  FutexWaitNode* waiters = nullptr;
  std::atomic<int> wait_count{0};
  // Diagnostics for ARC_FUTEX_TRACE: the last guest caller that woke this
  // word. A wait that outlives a frame is either a timeout or a late wake, and
  // only the waker's address says which of the two sides was late.
  std::atomic<uint64_t> last_waker_lr{0};
  std::atomic<uint64_t> last_waker_sp{0};
};

struct FutexWaitNode {
  uint64_t address = 0;
  std::condition_variable cv;
  bool woken = false;
  FutexWaitNode* next = nullptr;
};

// The waker is recorded on the bucket, so the waiter has to carry the answer
// back to its own trace line.
thread_local uint64_t t_woken_by_lr = 0;
thread_local uint64_t t_woken_by_sp = 0;

constexpr size_t kFutexBuckets = 256;
FutexBucket g_futex[kFutexBuckets];

// How long a blocked wait stays asleep before looking for a guest signal. A
// wake always returns at once; only cooperative signal delivery waits for the
// slice to end, and nothing in this build relies on sub-frame signal latency.
constexpr int kWaitSliceMs = 20;

FutexBucket& FutexBucketFor(uint64_t address) {
  const uint64_t mixed = (address >> 2) ^ (address >> 11) ^ (address >> 21);
  return g_futex[mixed % kFutexBuckets];
}

uint64_t FutexNowNs(bool realtime) {
  if (realtime) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
  }
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

// Return the remaining wait in nanoseconds, a negative value for "no timeout",
// and zero when the deadline has already passed.
int64_t FutexTimeoutNs(uint64_t address, bool absolute, bool realtime) {
  if (!address) return -1;
  const auto* ts = reinterpret_cast<const GuestTimespec*>(address);
  const uint64_t seconds = static_cast<uint64_t>(ts->tv_sec);
  const uint64_t nanos = static_cast<uint64_t>(ts->tv_nsec);
  const uint64_t deadline = seconds * UINT64_C(1000000000) + nanos;
  if (!absolute) return static_cast<int64_t>(deadline);
  const uint64_t now = FutexNowNs(realtime);
  return deadline > now ? static_cast<int64_t>(deadline - now) : 0;
}

bool GuestReadable(uint64_t address, size_t size) {
  // Every futex wait asks this, and a spinning wait asks it hundreds of
  // thousands of times a second about the same word. The shared probe keeps
  // the region it last validated, per thread.
  return arc::GuestRangeReadable(address, size);
}

int64_t FutexWait(uint64_t address, uint32_t expected, int64_t timeout_ns) {
  auto* word = reinterpret_cast<volatile uint32_t*>(address);
  // A guest null or unmapped futex word is a guest fault, not a host one. The
  // Linux ABI reports EFAULT from the syscall boundary, so do the same after
  // checking the page instead of dereferencing it here.
  if (!GuestReadable(address, sizeof(uint32_t))) return -14;  // EFAULT
  if (*word != expected) return -11;  // EAGAIN

  FutexBucket& bucket = FutexBucketFor(address);
  FutexWaitNode node;
  node.address = address;
  std::unique_lock<std::mutex> held(bucket.lock);
  // The guest's compare and the enqueue have to be one step from the waker's
  // point of view. It takes the same bucket lock in FutexWake, so checking
  // again after acquiring it closes the lost-wakeup window.
  if (*word != expected) return -11;
  if (timeout_ns == 0) return 0;
  const auto start = std::chrono::steady_clock::now();
  const auto deadline =
      timeout_ns < 0
          ? std::chrono::steady_clock::time_point::max()
          : start + std::chrono::nanoseconds(timeout_ns);
  node.next = bucket.waiters;
  bucket.waiters = &node;
  bucket.wait_count.fetch_add(1, std::memory_order_relaxed);
  arc_set_wait_reason("futex_wait");
  t_woken_by_lr = 0;
  t_woken_by_sp = 0;
  int64_t result = 0;
  for (;;) {
    if (node.woken) break;
    if (arc_signal_pending_current()) {
      held.unlock();
      arc_run_pending_signals();
      held.lock();
      continue;
    }
    const auto now = std::chrono::steady_clock::now();
    if (now >= deadline) {
      result = -110;  // ETIMEDOUT
      break;
    }
    // The slice bounds how late a cooperative guest signal can be delivered to
    // a blocked thread; it is not part of the wait. A short slice costs a wake
    // per slice per blocked thread, and with a dozen parked workers that is
    // real idle CPU, so keep it as long as the signal latency allows.
    auto slice = now + std::chrono::milliseconds(kWaitSliceMs);
    if (slice > deadline) slice = deadline;
    node.cv.wait_until(held, slice);
  }
  // The node lives on this stack, so it has to leave the list before the wait
  // returns. The bucket lock still excludes every waker, which is what makes
  // the unlink safe.
  for (FutexWaitNode** link = &bucket.waiters; *link; link = &(*link)->next) {
    if (*link == &node) {
      *link = node.next;
      break;
    }
  }
  bucket.wait_count.fetch_sub(1, std::memory_order_relaxed);
  if (result == 0) {
    t_woken_by_lr = bucket.last_waker_lr.load(std::memory_order_relaxed);
    t_woken_by_sp = bucket.last_waker_sp.load(std::memory_order_relaxed);
  }
  held.unlock();
  arc_set_wait_reason(nullptr);
  return result;
}

int64_t FutexWake(uint64_t address, uint32_t count) {
  if (!count) return 0;
  if (!GuestReadable(address, sizeof(uint32_t))) return -14;
  // Only the trace pays for naming the waker; the flag is read once.
  static const bool trace_wakers = getenv("ARC_FUTEX_TRACE") != nullptr;
  FutexBucket& bucket = FutexBucketFor(address);
  std::lock_guard<std::mutex> held(bucket.lock);
  if (trace_wakers) {
    const Arm64Ctx* w = arc_current_guest_context();
    bucket.last_waker_lr.store(w ? w->x[30] : 0, std::memory_order_relaxed);
    bucket.last_waker_sp.store(w ? w->sp : 0, std::memory_order_relaxed);
  }
  // Linux reports how many waiters were actually woken, and a wake for a word
  // that nobody is waiting on must not spend a permit on another word.
  uint32_t woken = 0;
  for (FutexWaitNode* n = bucket.waiters; n && woken < count; n = n->next) {
    if (n->address != address) continue;
    n->woken = true;
    n->cv.notify_one();
    ++woken;
  }
  return woken;
}

void FutexRequeue(uint64_t from, uint64_t to, uint32_t wake_count,
                  uint32_t requeue_count) {
  FutexWake(from, wake_count ? wake_count
                             : std::numeric_limits<uint32_t>::max());
  // There is no ownership transfer in this model. Waking the destination is a
  // permitted spurious wakeup, and the waiter re-reads its word and queues
  // again if it still has to wait.
  if (requeue_count)
    FutexWake(to, std::numeric_limits<uint32_t>::max());
}

}  // namespace

extern "C" int64_t arc_svc(Arm64Ctx* c) {
  const uint64_t nr = c->x[8];
  const uint64_t a0 = c->x[0], a1 = c->x[1], a2 = c->x[2], a3 = c->x[3];
  const uint64_t a4 = c->x[4], a5 = c->x[5];

  switch (nr) {
    case 56: {  // openat(dirfd, path, flags, mode)
      const auto fn = Lookup<OpenFn>("open");
      if (!fn) return -38;  // ENOSYS
      const int fd = fn(reinterpret_cast<const char*>(a1), static_cast<int>(a2),
                        static_cast<int>(a3));
      return fd < 0 ? NegErrno() : fd;
    }
    case 57: {  // close(fd)
      const auto fn = Lookup<CloseFn>("close");
      if (!fn) return -38;
      return fn(static_cast<int>(a0)) == 0 ? 0 : NegErrno();
    }
    case 62: {  // lseek(fd, offset, whence)
      const auto fn = Lookup<LseekFn>("lseek");
      if (!fn) return -38;
      const int64_t r = fn(static_cast<int>(a0), static_cast<int64_t>(a1),
                           static_cast<int>(a2));
      return r < 0 ? NegErrno() : r;
    }
    case 63: {  // read(fd, buf, count)
      const auto fn = Lookup<ReadFn>("read");
      if (!fn) return -38;
      const int64_t r =
          fn(static_cast<int>(a0), reinterpret_cast<void*>(a1), a2);
      return r < 0 ? NegErrno() : r;
    }
    case 64: {  // write(fd, buf, count)
      const auto fn = Lookup<WriteFn>("write");
      if (!fn) return -38;
      const int64_t r =
          fn(static_cast<int>(a0), reinterpret_cast<const void*>(a1), a2);
      return r < 0 ? NegErrno() : r;
    }
    case 79: {  // newfstatat(dirfd, path, buf, flags)
      const auto fn = Lookup<StatFn>("stat");
      if (!fn) return -38;
      const int r = fn(reinterpret_cast<const char*>(a1),
                       reinterpret_cast<void*>(a2));
      return r == 0 ? 0 : NegErrno();
    }
    case 80: {  // fstat(fd, buf)
      const auto fn = Lookup<FstatFn>("fstat");
      if (!fn) return -38;
      const int r = fn(static_cast<int>(a0), reinterpret_cast<void*>(a1));
      return r == 0 ? 0 : NegErrno();
    }
    case 98: {  // futex(uaddr, op, value, timeout, uaddr2, value3)
      if (!GuestReadable(a0, sizeof(uint32_t))) {
        fprintf(stderr,
                "[svc] futex invalid uaddr=%#llx op=%#llx value=%#llx "
                "timeout=%#llx lr=%#llx sp=%#llx x19=%#llx x20=%#llx "
                "x21=%#llx\n",
                static_cast<unsigned long long>(a0),
                static_cast<unsigned long long>(a1),
                static_cast<unsigned long long>(a2),
                static_cast<unsigned long long>(a3),
                static_cast<unsigned long long>(c->x[30]),
                static_cast<unsigned long long>(c->sp),
                static_cast<unsigned long long>(c->x[19]),
                static_cast<unsigned long long>(c->x[20]),
                static_cast<unsigned long long>(c->x[21]));
      }
      // The low seven bits are the command; the upper bits are modifiers
      // (FUTEX_PRIVATE_FLAG, FUTEX_CLOCK_REALTIME). A private futex only
      // promises not to wake a waiter in another address space, which a single
      // host process does not have, so both forms use the same buckets.
      constexpr uint32_t kCommandMask = 0x7f;
      constexpr uint32_t kClockRealtime = 0x100;
      constexpr uint32_t kWait = 0;
      constexpr uint32_t kWaitBitset = 9;
      constexpr uint32_t kWake = 1;
      constexpr uint32_t kRequeue = 3;
      constexpr uint32_t kCmpRequeue = 4;
      constexpr uint32_t kWakeOp = 5;
      constexpr uint32_t kWakeBitset = 10;
      const uint32_t command = static_cast<uint32_t>(a1) & kCommandMask;
      const uint32_t value = static_cast<uint32_t>(a2);
      static std::atomic<int> traced{0};
      if (getenv("ARC_FUTEX_TRACE") && traced.fetch_add(1) < 24)
        fprintf(stderr,
                "[futex] op=%#llx uaddr=%#llx val=%#llx timeout=%#llx "
                "uaddr2=%#llx value3=%#llx\n",
                static_cast<unsigned long long>(a1),
                static_cast<unsigned long long>(a0),
                static_cast<unsigned long long>(a2),
                static_cast<unsigned long long>(a3),
                static_cast<unsigned long long>(a4),
                static_cast<unsigned long long>(a5));
      const bool trace = getenv("ARC_FUTEX_TRACE") != nullptr;
      const auto started = trace ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point{};
      int64_t result = 0;
      // What the caller asked for, as opposed to what it got. A wait that lasts
      // its full deadline and one that is woken late look identical in the
      // duration alone.
      int64_t requested_ns = -1;
      switch (command) {
        case kWait:
        case kWaitBitset: {
          const bool absolute = command == kWaitBitset;
          const bool realtime = (static_cast<uint32_t>(a1) & kClockRealtime) != 0;
          requested_ns = FutexTimeoutNs(a3, absolute, realtime);
          result = FutexWait(a0, value, requested_ns);
          break;
        }
        case kWake:
        case kWakeBitset:
          result = FutexWake(a0, value);
          break;
        case kRequeue:
          FutexRequeue(a0, a4, value, static_cast<uint32_t>(a3));
          result = static_cast<int64_t>(value) + static_cast<int64_t>(a3);
          break;
        case kCmpRequeue: {
          auto* word = reinterpret_cast<volatile uint32_t*>(a0);
          if (!word) {
            result = -14;
            break;
          }
          if (*word != static_cast<uint32_t>(a5)) {
            result = -11;  // EAGAIN
            break;
          }
          FutexRequeue(a0, a4, value, static_cast<uint32_t>(a3));
          result = static_cast<int64_t>(value) + static_cast<int64_t>(a3);
          break;
        }
        case kWakeOp:
          // The value-3 operand also carries an atomic operation on uaddr2.
          // The operation is only an optimisation in glibc's mutex fast path;
          // waking both words and letting each waiter re-check its own word
          // preserves the futex contract (spurious wakeups are allowed).
          FutexRequeue(a0, a4, value, 0);
          result = value;
          break;
        default:
          // PI futexes (priority inheritance) are not used by the guest's
          // condition variables or job system. Report the unsupported command
          // so a future caller is visible instead of silently spinning.
          result = -38;  // ENOSYS
          break;
      }
      if (trace) {
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - started)
                              .count();
        // A wait that lasts longer than a frame, or one that gives up on its
        // timeout, is the interesting kind: it names the guest caller and the
        // word that nobody signalled.
        if (ms >= 20.0 || result == -110) {
          // Name the caller. A futex reached through the `syscall` import
          // arrives with a synthesised context whose lr and sp are zero, so
          // fall back to the thread's real guest context, which is the lifted
          // function that asked.
          const Arm64Ctx* caller = c;
          if (!caller->x[30]) caller = arc_current_guest_context();
          fprintf(stderr,
                  "[futex] op=%#llx cmd=%#x addr=%#llx val=%#llx timeout=%#llx"
                  " ret=%lld %.1fms want=%lldms lr=%#llx sp=%#llx"
                  " by=%#llx bysp=%#llx\n",
                  static_cast<unsigned long long>(a1), command,
                  static_cast<unsigned long long>(a0),
                  static_cast<unsigned long long>(value),
                  static_cast<unsigned long long>(a3),
                  static_cast<long long>(result), ms,
                  static_cast<long long>(requested_ns < 0 ? -1
                                                          : requested_ns / 1000000),
                  caller ? static_cast<unsigned long long>(caller->x[30]) : 0,
                  caller ? static_cast<unsigned long long>(caller->sp) : 0,
                  static_cast<unsigned long long>(t_woken_by_lr),
                  static_cast<unsigned long long>(t_woken_by_sp));
        }
      }
      return result;
    }
    case 113: {  // clock_gettime(clockid, timespec*)
      auto* ts = reinterpret_cast<GuestTimespec*>(a1);
      if (!ts) return -14;  // EFAULT
      const int clock = static_cast<int>(a0);
      int64_t ns;
      if (clock == 0) {
        ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                 std::chrono::system_clock::now().time_since_epoch())
                 .count();
      } else {
        ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                 std::chrono::steady_clock::now().time_since_epoch())
                 .count();
      }
      ts->tv_sec = ns / 1000000000;
      ts->tv_nsec = ns % 1000000000;
      return 0;
    }
    case 135:  // rt_sigprocmask
      return 0;
    case 169: {  // gettimeofday(timeval*, timezone*)
      const auto fn = Lookup<TimeFn>("gettimeofday");
      if (!fn) return -38;
      return fn(reinterpret_cast<void*>(a0), reinterpret_cast<void*>(a1)) == 0
                 ? 0
                 : NegErrno();
    }
    case 172:  // getpid
      return 1;
    case 178:  // gettid
#if defined(_WIN32)
      return static_cast<int64_t>(GetCurrentThreadId());
#else
      return static_cast<int64_t>(syscall(SYS_gettid));
#endif
    case 278:  // getrandom(buf, len, flags)
      arc::ShimFillRandom(reinterpret_cast<void*>(a0),
                          static_cast<size_t>(a1));
      return static_cast<int64_t>(a1);
    case 93:   // exit
    case 94:   // exit_group
      // Reported, not taken: this is the same act as calling `exit`, and a
      // title that gives up here used to end the run with nothing said about
      // why. A run that stops during initialisation with no message is this.
      arc::GuestExit(static_cast<int>(a0));
    default:
      return -38;  // ENOSYS
  }
}
