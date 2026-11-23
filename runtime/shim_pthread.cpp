// POSIX threads, semaphores and thread-local keys for the engine.
//
// The engine allocates `pthread_mutex_t`, `pthread_cond_t` and friends inline
// inside its own structures, sized by Bionic's headers at the time it was
// compiled. We cannot change those sizes, and guessing them wrong is silent
// memory corruption rather than a crash.
//
// So we never depend on them. Every pthread call routes through this file, so
// the guest never inspects the bytes itself -- which means the storage is
// opaque and we only have to *fit inside* it, not match its layout. Each
// object holds a single 32-bit id naming an entry in a registry here. The
// smallest of Bionic's types is `pthread_once_t` at four bytes, so a `uint32_t`
// fits every one of them with room to spare, on every host, with no NDK
// headers required to prove it.
//
// Zero means "not created yet". That is not an accident of ours: Bionic's
// PTHREAD_MUTEX_INITIALIZER is static zeroes, so a statically initialised mutex
// legitimately arrives here having never been through pthread_mutex_init.

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <csetjmp>
#include <cstdio>
#include <cstdlib>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <pthread.h>
#endif

#include "arm64_context.h"
#include "arc_signal.h"
#include "shim.h"

// The generated dispatcher owns the image table. libunity is image 1 in the
// host's load order (libil2cpp is 0), and its frame clock is one of the
// globals the lifted code addresses directly.
extern "C" uint64_t arc_image_bases[];
extern "C" int arc_image_owns(uint64_t address);

namespace arc {
namespace {

// ponytail: one lock over every registry. Each mutex_lock briefly serialises
// here, which is a real throughput ceiling for a game. Shard by id, or add a
// lock-free fast path for the already-created case, if profiling says so.
std::mutex g_registry;
uint32_t g_next_id = 1;

template <typename T>
struct Registry {
  // std::map, not unordered_map: references to elements stay valid across
  // inserts, so a holder can keep using its object while another thread
  // creates one.
  std::map<uint32_t, std::unique_ptr<T>> items;

  T* Obtain(uint32_t* slot) {
    std::lock_guard<std::mutex> g(g_registry);
    if (*slot == 0) {
      *slot = g_next_id++;
      items[*slot] = std::make_unique<T>();
    }
    auto it = items.find(*slot);
    return it == items.end() ? nullptr : it->second.get();
  }

  T* Find(uint32_t id) {
    std::lock_guard<std::mutex> g(g_registry);
    auto it = items.find(id);
    return it == items.end() ? nullptr : it->second.get();
  }

  void Release(uint32_t* slot) {
    std::lock_guard<std::mutex> g(g_registry);
    items.erase(*slot);
    *slot = 0;
  }
};

// --- mutexes ---------------------------------------------------------------
// ponytail: every mutex is recursive, whatever type the guest asked for.
// Recursive is strictly more permissive than normal -- code that is correct
// under PTHREAD_MUTEX_NORMAL is still correct here; it just no longer
// self-deadlocks. Split by type only if we ever want to reproduce a deadlock.
Registry<std::recursive_mutex> g_mutexes;
Registry<std::condition_variable_any> g_conds;
Registry<std::shared_mutex> g_rwlocks;

struct Semaphore {
  std::mutex m;
  std::condition_variable cv;
  unsigned count = 0;
};
Registry<Semaphore> g_semaphores;

// A guest thread is not a std::thread running a host function. Its entry point
// is a guest address, so it has to be dispatched with a context and a stack of
// its own -- and the host thread it runs on needs a large stack, because lifted
// functions call one another as ordinary C functions and the guest's call depth
// is the host's.
constexpr size_t kGuestThreadStack = 16 * 1024 * 1024;
constexpr size_t kGuestThreadHeadroom = 1 * 1024 * 1024;
constexpr size_t kHostThreadStack = 256u << 20;

// How many guest threads are alive, and the ceiling on that.
//
// A real system has one: pthread_create answers EAGAIN when a process asks
// for more than it may have. A host with no ceiling instead gives a runaway
// everything it asks for, and the first thing to notice is the machine
// rather than the port -- which is not a diagnosis anyone can act on.
//
// It is a backstop, not a policy, and the number is deliberately far above
// what any title needs: a guest that reaches it aborts, because the NDK's
// std::thread turns EAGAIN into an exception nothing catches, so a ceiling
// low enough to be tidy would kill runs that were working. Set
// ARC_MAX_THREADS lower to find out *which* function is asking, together
// with ARC_TRACE_THREADS; leave it alone otherwise.
std::atomic<long> g_live_threads{0};
std::atomic<long> g_threads_made{0};
const std::chrono::steady_clock::time_point g_first_thread =
    std::chrono::steady_clock::now();

long GuestThreadCeiling() {
  static const long limit = [] {
    const char* s = getenv("ARC_MAX_THREADS");
    const long v = s ? strtol(s, nullptr, 10) : 0;
    // Two thousand. Measured rather than chosen: this host stopped being able
    // to start threads at all somewhere past ten thousand, and by then every
    // other program on it was in trouble too -- including the tools looking
    // at the problem. A guest that aborts with the message above is a bug
    // report; a machine that cannot fork is not.
    return v > 0 ? v : 2048;
  }();
  return limit;
}


struct Thread {
#if defined(_WIN32)
  void* handle = nullptr;
  // Kept separately because a detached thread's handle is closed while the
  // thread is still alive, and pthread_kill must still be able to name it.
  uint32_t host_id = 0;
#else
  pthread_t handle{};
  bool started = false;
#endif
  void* result = nullptr;
  bool detached = false;
  // Which shim this thread is blocked in, for the diagnostics that have to
  // explain why a stop-the-world signal was not acknowledged.
  const char* wait_reason = nullptr;
};
Registry<Thread> g_threads;

struct Attr {
  int detach_state = 0;
  size_t stack_size = 0;
};
Registry<Attr> g_attrs;

// --- thread-local keys -----------------------------------------------------
struct KeyTable {
  std::vector<void*> values;
};
thread_local KeyTable t_keys;
std::atomic<uint32_t> g_next_key{1};

// pthread_key_create's destructor, by key. IL2CPP registers one to tear down
// the managed thread object, so a thread that ends has to run these or the
// runtime keeps a thread it can still name and the next one to reuse the slot
// trips over the corpse.
std::mutex g_key_lock;
std::vector<void (*)(void*)> g_key_destructors;

// Which rwlocks this thread holds for writing, so unlock knows which of
// shared_mutex's two unlock calls to make. pthread has one unlock for both.
thread_local std::vector<uint32_t> t_write_held;

// Bionic's timespec on LP64.
struct GuestTimespec {
  int64_t tv_sec;
  int64_t tv_nsec;
};

// The id the guest holds for the calling thread, so pthread_self() and
// pthread_kill() have something to name.
thread_local uint32_t t_self_id = 0;

// pthread_exit ends a thread, but a lifted guest frame cannot be unwound from
// inside: the C stack that would be unwound belongs to the lifter, not to the
// guest. So the thread's entry point arms a host setjmp before it enters guest
// code, and pthread_exit jumps back there. The value travels beside it because
// the jump lands in a different frame.
thread_local jmp_buf* t_exit_jmp = nullptr;
thread_local void* t_exit_value = nullptr;

// Bionic's pthread_self() never returns zero: it answers the address of the
// thread's own descriptor, and guest code is entitled to use zero as "nobody".
// IL2CPP's monitor does exactly that -- a monitor block stores the owning
// pthread_t and treats zero as "unlocked" -- so a main thread that answered
// zero would let any other thread win the acquire against a lock the main
// thread already holds, and the main thread's own Exit would then throw
// SynchronizationLockException. Every thread gets a real id, main included;
// the ones RunGuestThread creates are registered there, this covers the rest.
static uint32_t SelfId() {
  if (t_self_id == 0) {
    uint32_t slot = 0;
    Thread* t = g_threads.Obtain(&slot);
    t_self_id = slot;
    if (t) {
#if defined(_WIN32)
      t->host_id = GetCurrentThreadId();
#else
      t->handle = pthread_self();
      t->started = false;
#endif
    }
  }
  return t_self_id;
}

static Thread* SelfThread() {
  return g_threads.Find(SelfId());
}

static void SetWaitReason(const char* reason) {
  if (Thread* t = SelfThread()) t->wait_reason = reason;
}

extern "C" void arc_set_wait_reason(const char* reason) {
  SetWaitReason(reason);
}

// The id this thread presents to the guest, for the diagnostics that need to
// say which thread held a lock.
extern "C" uint32_t arc_guest_self_id(void) { return SelfId(); }

// --- entry points ----------------------------------------------------------

int MutexInit(uint32_t* m, const void*) {
  *m = 0;
  g_mutexes.Obtain(m);
  return 0;
}
int MutexDestroy(uint32_t* m) {
  g_mutexes.Release(m);
  return 0;
}
int MutexLock(uint32_t* m) {
  if (auto* p = g_mutexes.Obtain(m)) {
    SetWaitReason("mutex_lock");
    p->lock();
    SetWaitReason(nullptr);
  }
  return 0;
}
int MutexUnlock(uint32_t* m) {
  if (auto* p = g_mutexes.Obtain(m)) p->unlock();
  return 0;
}
int MutexTrylock(uint32_t* m) {
  auto* p = g_mutexes.Obtain(m);
  return (p && p->try_lock()) ? 0 : 16 /* EBUSY */;
}
int MutexTimedlock(uint32_t* m, int, const void*) {
  // The object is our own mutex, not a host pthread_mutex_t, so there is no
  // host timed lock to forward to. A blocking lock preserves mutual exclusion;
  // the timeout is advisory in the engine's use of it.
  return MutexLock(m);
}
int MutexattrInit(uint32_t* a) {
  *a = 0;
  return 0;
}
int MutexattrDestroy(uint32_t*) { return 0; }
int MutexattrSettype(uint32_t*, int) { return 0; }  // see the note above

int CondattrInit(void* attr) {
  if (attr) memset(attr, 0, 8);
  return 0;
}
int CondattrDestroy(void*) { return 0; }
int CondattrSetclock(void*, int) { return 0; }

void TraceCondOp(const char* op, uint32_t* c) {
  if (!getenv("ARC_TRACE_COND")) return;
  fprintf(stderr, "[cond] %s c=%p\n", op, (void*)c);
}

int CondInit(uint32_t* c, const void*) {
  *c = 0;
  g_conds.Obtain(c);
  TraceCondOp("init", c);
  return 0;
}
int CondDestroy(uint32_t* c) {
  g_conds.Release(c);
  return 0;
}
int CondSignal(uint32_t* c) {
  if (auto* p = g_conds.Obtain(c)) p->notify_one();
  TraceCondOp("signal", c);
  return 0;
}
int CondBroadcast(uint32_t* c) {
  if (auto* p = g_conds.Obtain(c)) p->notify_all();
  TraceCondOp("broadcast", c);
  return 0;
}
// The engine's `UnityChoreographer` is a native object that asks Android's
// Java Choreographer for a frame and then waits on a condition variable until
// the Java callback runs. A host with no Java side never gets that callback,
// so the main thread parks here before the first draw and every later frame
// waits forever for a vsync that cannot arrive.
//
// Android's callback writes the frame time into the 16-byte record at
// `this+0x58` and signals `this+0x88`. We can supply the same edge from the
// wait itself: the condition's owner is 0x88 bytes below it, and the record is
// recognisable by its second word being the one-time "armed" flag. This is
// deliberately gated by ARC_FAKE_VSYNC so a real Java or platform vsync source
// can take over without changing the default.
static uint64_t MonotonicNanosNow() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

bool FakeVsync(uint32_t* c) {
  static const bool enabled = getenv("ARC_FAKE_VSYNC") != nullptr;
  if (!enabled) return false;
  static int shown = 0;
  const bool trace = getenv("ARC_TRACE_COND") != nullptr;
  const uintptr_t owner = reinterpret_cast<uintptr_t>(c) - 0x88;
  MEMORY_BASIC_INFORMATION mbi;
  if (!VirtualQuery(reinterpret_cast<const void*>(owner), &mbi, sizeof mbi) ||
      mbi.State != MEM_COMMIT || (mbi.Protect & PAGE_GUARD) ||
      !(mbi.Protect & (PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY |
                       PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE |
                       PAGE_EXECUTE_WRITECOPY)))
    return false;
  uint64_t record = 0;
  memcpy(&record, reinterpret_cast<const void*>(owner + 0x58), sizeof record);
  if (trace && record && shown < 40) {
    ++shown;
    Arm64Ctx* ctx = arc_current_guest_context();
    fprintf(stderr,
            "[cond] c=%p owner=%p lr=%#llx sp=%#llx record=%#llx\n", (void*)c,
            (void*)owner, ctx ? (unsigned long long)ctx->x[30] : 0,
            ctx ? (unsigned long long)ctx->sp : 0,
            static_cast<unsigned long long>(record));
  }
  if (!record || (record & 7)) return false;
  if (!VirtualQuery(reinterpret_cast<const void*>(record), &mbi, sizeof mbi) ||
      mbi.State != MEM_COMMIT || (mbi.Protect & PAGE_GUARD) ||
      !(mbi.Protect & (PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY |
                       PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE |
                       PAGE_EXECUTE_WRITECOPY)))
    return false;
  if (record + 16 > reinterpret_cast<uintptr_t>(mbi.BaseAddress) +
                       mbi.RegionSize)
    return false;
  uint64_t frame[2] = {};
  memcpy(frame, reinterpret_cast<const void*>(record), sizeof frame);
  if (trace && shown <= 40)
    fprintf(stderr, "[cond] frame=%#llx,%#llx\n",
            static_cast<unsigned long long>(frame[0]),
            static_cast<unsigned long long>(frame[1]));
  // The engine stores the armed flag as a 32-bit 1 at +8, leaving the upper
  // half of that word untouched. Comparing the whole 64-bit word made this
  // depend on the allocator's leftover bytes, so ask only what the code wrote.
  if (frame[0] != 0 || (frame[1] & 0xffffffffu) != 1) return false;
  // The stamp is a frame *time*, not a flag: the engine measures the frame from
  // it, so a constant leaves every delta at zero and time stops advancing while
  // the render loop keeps drawing. A real monotonic timestamp -- the same kind
  // of value Android's Choreographer hands over -- restores the frame step.
  const uint64_t stamp = MonotonicNanosNow();
  frame[0] = stamp;
  memcpy(reinterpret_cast<void*>(record), frame, sizeof frame);
  if (auto* p = g_conds.Obtain(c)) p->notify_all();
  static bool said = false;
  if (!said) {
    said = true;
    Arm64Ctx* ctx = arc_current_guest_context();
    fprintf(stderr,
            "[vsync] synthesised frame for UnityChoreographer at %p "
            "stamp=%llu lr=%#llx sp=%#llx\n",
            reinterpret_cast<void*>(owner),
            static_cast<unsigned long long>(stamp),
            ctx ? static_cast<unsigned long long>(ctx->x[30]) : 0ull,
            ctx ? static_cast<unsigned long long>(ctx->sp) : 0ull);
  }
  return true;
}

// The engine also keeps a monotonic frame clock and waits until it reaches a
// requested value. The value is advanced by the Android choreographer, which
// this host does not have. The wait is recognisable from the guest side: the
// return address is the instruction after `pthread_cond_wait` in the clock
// loop, the condition sits 0x28 bytes above its mutex, and the requested value
// is in x19. Advancing the clock to that value is equivalent to a host vsync
// tick and lets the frame finish.
bool FakeUnityClock(uint32_t* c, uint32_t* m) {
  static const bool enabled = getenv("ARC_FAKE_VSYNC") != nullptr;
  if (!enabled) return false;
  Arm64Ctx* ctx = arc_current_guest_context();
  if (!ctx) return false;
  const uint64_t unity = arc_image_bases[1];
  const uint64_t lr = ctx->x[30];
  if (!unity || lr <= unity || lr - unity != UINT64_C(0x9ee648)) return false;
  if (reinterpret_cast<uintptr_t>(m) + 0x28 !=
      reinterpret_cast<uintptr_t>(c))
    return false;
  uint64_t* now = reinterpret_cast<uint64_t*>(
      reinterpret_cast<uintptr_t>(m) + 0x58);
  MEMORY_BASIC_INFORMATION mbi;
  if (!VirtualQuery(now, &mbi, sizeof mbi) || mbi.State != MEM_COMMIT ||
      (mbi.Protect & PAGE_GUARD) ||
      !(mbi.Protect & (PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY |
                       PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE |
                       PAGE_EXECUTE_WRITECOPY)))
    return false;
  const uint64_t target = ctx->x[19];
  // The engine's frame clock is an absolute time in seconds, and a frame step
  // is the difference between two frames. It is advanced by the Android vsync
  // callback, which this host does not have, so every step is zero and the
  // engine's own clock stops while the renderer keeps drawing: animations,
  // loading progress and UI fades all crawl at the minimum step the engine will
  // accept. Adding the elapsed host time here is what the callback does.
  static uint64_t last_clock_ns = 0;
  const uint64_t clock_ns = MonotonicNanosNow();
  if (last_clock_ns && clock_ns > last_clock_ns)
    *reinterpret_cast<double*>(unity + UINT64_C(0x1886740)) +=
        static_cast<double>(clock_ns - last_clock_ns) * 1e-9;
  last_clock_ns = clock_ns;
  // The engine computes this target as `last + ticks_per_frame`. A saturated
  // tick count makes the clock jump by 0x7fffffff and leaves deltaTime
  // meaningless, so the value is advanced to the target as before.
  if (*now < target) {
    static bool said = false;
    if (!said) {
      said = true;
      fprintf(stderr, "[clock] frame clock %#llx -> %#llx\n",
              static_cast<unsigned long long>(*now),
              static_cast<unsigned long long>(target));
    }
    *now = target;
  }
  if (auto* p = g_conds.Obtain(c)) p->notify_all();
  return true;
}

int CondWait(uint32_t* c, uint32_t* m) {
  auto* cv = g_conds.Obtain(c);
  auto* mu = g_mutexes.Obtain(m);
  if (!cv || !mu) return 22 /* EINVAL */;
  std::unique_lock<std::recursive_mutex> lk(*mu, std::adopt_lock);
  SetWaitReason("cond_wait");
  if (FakeVsync(c) || FakeUnityClock(c, m)) {
    SetWaitReason(nullptr);
    lk.release();  // the guest still owns the lock on return
    return 0;
  }
  for (;;) {
    if (arc_signal_pending_current()) {
      lk.unlock();
      arc_run_pending_signals();
      lk.lock();
      continue;
    }
    if (cv->wait_for(lk, std::chrono::milliseconds(5)) ==
        std::cv_status::no_timeout)
      break;
  }
  SetWaitReason(nullptr);
  lk.release();  // the guest still owns the lock on return
  return 0;
}
int CondTimedwait(uint32_t* c, uint32_t* m, const GuestTimespec* abstime) {
  auto* cv = g_conds.Obtain(c);
  auto* mu = g_mutexes.Obtain(m);
  if (!cv || !mu) return 22;
  std::unique_lock<std::recursive_mutex> lk(*mu, std::adopt_lock);
  // pthread's timeout is absolute against CLOCK_REALTIME. system_clock does
  // not tick in nanoseconds on every host, so the deadline is cast to whatever
  // resolution it does use.
  const auto since_epoch = std::chrono::seconds(abstime->tv_sec) +
                           std::chrono::nanoseconds(abstime->tv_nsec);
  const auto deadline = std::chrono::system_clock::time_point(
      std::chrono::duration_cast<std::chrono::system_clock::duration>(
          since_epoch));
  SetWaitReason("cond_timedwait");
  if (FakeVsync(c) || FakeUnityClock(c, m)) {
    SetWaitReason(nullptr);
    lk.release();
    return 0;
  }
  for (;;) {
    if (arc_signal_pending_current()) {
      lk.unlock();
      arc_run_pending_signals();
      lk.lock();
      continue;
    }
    const auto now = std::chrono::system_clock::now();
    if (now >= deadline) {
      SetWaitReason(nullptr);
      lk.release();
      return 110 /* ETIMEDOUT */;
    }
    auto slice = now + std::chrono::milliseconds(5);
    if (slice > deadline) slice = deadline;
    if (cv->wait_until(lk, slice) == std::cv_status::no_timeout) {
      SetWaitReason(nullptr);
      lk.release();
      return 0;
    }
  }
}

int RwlockInit(uint32_t* l, const void*) {
  *l = 0;
  g_rwlocks.Obtain(l);
  return 0;
}
int RwlockDestroy(uint32_t* l) {
  g_rwlocks.Release(l);
  return 0;
}
int RwlockRdlock(uint32_t* l) {
  if (auto* p = g_rwlocks.Obtain(l)) p->lock_shared();
  return 0;
}
int RwlockWrlock(uint32_t* l) {
  if (auto* p = g_rwlocks.Obtain(l)) {
    p->lock();
    t_write_held.push_back(*l);
  }
  return 0;
}
int RwlockUnlock(uint32_t* l) {
  auto* p = g_rwlocks.Obtain(l);
  if (!p) return 22;
  auto it = std::find(t_write_held.begin(), t_write_held.end(), *l);
  if (it != t_write_held.end()) {
    t_write_held.erase(it);
    p->unlock();
  } else {
    p->unlock_shared();
  }
  return 0;
}

// pthread_once_t is a bare int, so it holds the state machine directly rather
// than a registry id: 0 = untouched, 1 = running, 2 = done.
// The initialiser is guest code, so it is dispatched rather than called. A
// guest function pointer invoked as a host one executes ARM instructions on an
// x86 processor, which arrives as an access violation *on execute* -- a fault
// that names an address inside the game's own image and looks for all the
// world like a bad pointer in the guest, rather than like the host calling it
// wrongly. Every shim entry point taking a callback has this hazard; a thread's
// entry point already dispatches, and this is the same thing.
int Once(uint32_t* control, uint64_t fn) {
  uint32_t expected = 0;
  auto* state = reinterpret_cast<std::atomic<uint32_t>*>(control);
  if (state->compare_exchange_strong(expected, 1)) {
    arc_call_guest(fn, nullptr, 0);
    state->store(2);
    return 0;
  }
  while (state->load() != 2) std::this_thread::yield();
  return 0;
}

int KeyCreate(uint32_t* key, void (*destructor)(void*)) {
  const uint32_t k = g_next_key.fetch_add(1);
  {
    std::lock_guard<std::mutex> lock(g_key_lock);
    if (g_key_destructors.size() <= k) g_key_destructors.resize(k + 1, nullptr);
    g_key_destructors[k] = destructor;
  }
  *key = k;
  return 0;
}
int KeyDelete(uint32_t) { return 0; }

// Runs every key destructor this thread has armed, the way Bionic does when a
// thread ends: a destructor may set the key again, so repeat a bounded number
// of times. Calls out to guest code, so it must not hold the registry lock.
void RunThreadDestructors() {
  struct Pending {
    uint32_t key;
    void* value;
    void (*destructor)(void*);
  };
  for (int round = 0; round < 4 /* PTHREAD_DESTRUCTOR_ITERATIONS */; ++round) {
    std::vector<Pending> pending;
    {
      std::lock_guard<std::mutex> lock(g_key_lock);
      for (uint32_t k = 0; k < g_key_destructors.size(); ++k) {
        if (!g_key_destructors[k]) continue;
        if (k >= t_keys.values.size() || !t_keys.values[k]) continue;
        pending.push_back({k, t_keys.values[k], g_key_destructors[k]});
      }
    }
    if (pending.empty()) return;
    for (const Pending& p : pending) {
      if (p.key < t_keys.values.size() && t_keys.values[p.key] == p.value) {
        t_keys.values[p.key] = nullptr;
        const uint64_t args[1] = {reinterpret_cast<uint64_t>(p.value)};
        arc_call_guest(reinterpret_cast<uint64_t>(p.destructor), args, 1);
      }
    }
  }
}

// Threads here are always joinable; nothing asks for a detached one and the
// registry would keep its slot either way.
int AttrGetdetachstate(const void*, int* state) {
  if (state) *state = 0;  // PTHREAD_CREATE_JOINABLE
  return 0;
}
void* Getspecific(uint32_t key) {
  return key < t_keys.values.size() ? t_keys.values[key] : nullptr;
}
int Setspecific(uint32_t key, void* value) {
  if (key >= t_keys.values.size()) t_keys.values.resize(key + 1, nullptr);
  t_keys.values[key] = value;
  return 0;
}

struct GuestThreadStart {
  uint64_t entry;
  uint64_t arg;
  uint32_t id;
  Thread* slot;
};

#if defined(_WIN32)
// A guest thread had no fault handling at all. An access violation on one took
// the whole process down with no report attached to it, which is the least
// useful way a port can fail -- and it is where a title does its real work: the
// HTTP client, the asset decoder and the audio mixer all run here, not on the
// thread the entry points were called from. A fault reported badly on the main
// thread was still reported; a fault here was only ever an exit code.
//
// The thread does not survive it -- there is nothing sensible to resume -- but
// the process does, so the title reports the failure in its own terms and the
// run can still be read to the end.
int GuestThreadFault(EXCEPTION_POINTERS* ep, const Arm64Ctx* c, unsigned id) {
  const unsigned long code = ep->ExceptionRecord->ExceptionCode;
  unsigned long long at = 0;
  const char* kind = "";
  if (ep->ExceptionRecord->NumberParameters >= 2) {
    at = ep->ExceptionRecord->ExceptionInformation[1];
    switch (ep->ExceptionRecord->ExceptionInformation[0]) {
      case 0: kind = "read"; break;
      case 1: kind = "write"; break;
      case 8: kind = "execute"; break;
      default: break;
    }
  }
  fprintf(stderr, "\nguest thread %u faulted: %#lx on %s of %#llx\n",
          id, code, kind, at);
  const size_t frames = arc_frame_count();
  if (frames) {
    fprintf(stderr, "  guest functions entered, most recent first:\n");
    for (size_t i = 0; i < frames && i < 24; ++i) {
      const unsigned long long packed = arc_frame_at(i);
      fprintf(stderr, "    image %u +%#llx\n",
              (unsigned)(packed >> 56), packed & 0x00FFFFFFFFFFFFFFull);
    }
  }
  const size_t calls = arc_trace_count();
  if (calls) {
    fprintf(stderr, "  last calls out of the guest, most recent first:\n   ");
    for (size_t i = 0; i < calls && i < 12; ++i) {
      const char* what = arc_trace_at(i);
      fprintf(stderr, " %s", what ? what : "?");
    }
    fprintf(stderr, "\n");
  }
  // The frame ring above is global and so is shared with every other thread,
  // which makes it nearly useless here. The frame pointer is not: lifted code
  // keeps ARM64's stp x29, x30 chain, so this thread's own stack can be walked
  // out of its own context. Addresses are raw -- subtract the image base the
  // loader printed at startup.
  {
    unsigned long long fp = c->x[29];
    fprintf(stderr, "  guest call stack, innermost first:\n");
    if (c->x[30]) fprintf(stderr, "    in   %#llx\n", (unsigned long long)c->x[30]);
    for (int depth = 0; depth < 32; ++depth) {
      MEMORY_BASIC_INFORMATION mbi;
      if (!fp || (fp & 15)) break;
      if (!VirtualQuery((void*)fp, &mbi, sizeof mbi)) break;
      if (mbi.State != MEM_COMMIT || (mbi.Protect & PAGE_GUARD)) break;
      if (!(mbi.Protect & (PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY |
                           PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE |
                           PAGE_EXECUTE_WRITECOPY))) break;
      if (fp + 16 > (unsigned long long)mbi.BaseAddress + mbi.RegionSize) break;
      unsigned long long next = 0, ret = 0;
      memcpy(&next, (const void*)fp, sizeof next);
      memcpy(&ret, (const void*)(fp + 8), sizeof ret);
      if (!ret) break;
      fprintf(stderr, "    from %#llx\n", ret);
      if (next <= fp) break;
      fp = next;
    }
  }
  fprintf(stderr, "  guest registers:\n");
  for (int i = 0; i < 31; i += 4) {
    fprintf(stderr, "   ");
    for (int j = i; j < i + 4 && j < 31; ++j)
      fprintf(stderr, "  x%-2d=%016llx", j, (unsigned long long)c->x[j]);
    fprintf(stderr, "\n");
  }
  fprintf(stderr, "     sp =%016llx\n", (unsigned long long)c->sp);
  return EXCEPTION_EXECUTE_HANDLER;
}
#endif

// Kept free of C++ objects: MSVC will not put structured exception handling in
// a frame that also needs unwinding.
void GuestThreadBody(uint64_t entry, Arm64Ctx* ctx, unsigned id) {
#if defined(_WIN32)
  __try {
    arc_dispatch(ctx, entry);
  } __except (GuestThreadFault(GetExceptionInformation(), ctx, id)) {
  }
#else
  (void)id;
  arc_dispatch(ctx, entry);
#endif
}

// A guest stack, sixteen megabytes of address space and almost none of it
// real.
//
// This was a std::vector, whose constructor zeroes what it allocates -- so
// every guest thread touched all sixteen megabytes before running an
// instruction, and every page of it was resident for the life of the thread.
// A title that leaks threads then leaks *memory* at sixteen megabytes a go,
// and what should have been a wasted handle becomes a machine that stops
// responding. Demand-zero instead: the pages arrive as the guest stack is
// actually used, which for most threads is a few kilobytes, and the size can
// stay generous because it costs nothing to promise it.
void* AllocGuestStack(size_t size) {
#if defined(_WIN32)
  return VirtualAlloc(nullptr, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#else
  // calloc, not malloc+memset: for a block this size it is a fresh mapping,
  // which the kernel already guarantees zeroed without anyone writing to it.
  return calloc(1, size);
#endif
}

void FreeGuestStack(void* p) {
#if defined(_WIN32)
  if (p && getenv("ARC_TRACE_IMAGE_MAP") &&
      arc_image_owns(reinterpret_cast<uint64_t>(p)))
    fprintf(stderr, "[free-stack-image] %p\n", p);
  if (p) VirtualFree(p, 0, MEM_RELEASE);
#else
  free(p);
#endif
}

void RunGuestThread(GuestThreadStart* s) {
  t_self_id = s->id;
  {
    static const bool on = getenv("ARC_TRACE_THREADS") != nullptr;
    if (on)
      fprintf(stderr, "[thread] id=%u start=%#llx arg=%#llx\n", s->id,
              (unsigned long long)s->entry, (unsigned long long)s->arg);
  }
  ++g_live_threads;
  void* stack = AllocGuestStack(kGuestThreadStack);
  Arm64Ctx ctx;
  memset(&ctx, 0, sizeof(ctx));
  ctx.sp = (reinterpret_cast<uint64_t>(stack) + kGuestThreadStack -
            kGuestThreadHeadroom) & ~15ULL;
  ctx.x[0] = s->arg;
  // pthread_exit lands here. `value` is whatever the guest passed, and it
  // becomes the thread's return value exactly as if the start routine had
  // returned it.
  void* value = nullptr;
  jmp_buf exit_jmp;
  t_exit_jmp = &exit_jmp;
  // A guest trap on a worker used to abort the process: arc_trap only knew
  // about the boot thread's recovery point. Arm the same buffer pthread_exit
  // jumps to, so a bad branch in one worker ends that worker instead of the
  // boot. ARC_TRAP_ABORT restores the old fail-fast behaviour for diagnosis.
  const bool trap_recover = getenv("ARC_TRAP_ABORT") == nullptr;
  if (trap_recover) arc_set_recovery(&exit_jmp);
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4611)  // setjmp beside a frame that could unwind
#endif
  if (setjmp(exit_jmp) == 0) {
    // The setjmp frame has no C++ objects of its own, for the same reason
    // GuestThreadBody exists: MSVC will not mix SEH with unwinding.
#if defined(_WIN32)
    __try {
      GuestThreadBody(s->entry, &ctx, s->id);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
      const unsigned long code = GetExceptionCode();
      fprintf(stderr, "[thread %u] fault %#lx\n", s->id, code);
      const size_t depth = arc_frame_count();
      for (size_t i = 0; i < depth && i < 24; ++i) {
        const uint64_t packed = arc_frame_at(i);
        const size_t image = static_cast<size_t>(packed >> 56);
        const uint64_t addr = packed & ((UINT64_C(1) << 56) - 1);
        fprintf(stderr, "  #%zu image %zu + %#llx\n", i, image,
                static_cast<unsigned long long>(addr));
      }
    }
#else
    GuestThreadBody(s->entry, &ctx, s->id);
#endif
    value = reinterpret_cast<void*>(ctx.x[0]);
  } else {
    value = t_exit_value;
  }
#ifdef _MSC_VER
#pragma warning(pop)
#endif
  if (trap_recover) arc_set_recovery(nullptr);
  t_exit_jmp = nullptr;
  t_exit_value = nullptr;
  RunThreadDestructors();
  s->slot->result = value;
  FreeGuestStack(stack);
  --g_live_threads;
  delete s;
}

#if defined(_WIN32)
unsigned long __stdcall GuestThreadMain(void* p) {
  RunGuestThread(static_cast<GuestThreadStart*>(p));
  return 0;
}
#else
void* GuestThreadMain(void* p) {
  RunGuestThread(static_cast<GuestThreadStart*>(p));
  return nullptr;
}
#endif

// Which guest function each thread was started on, and how many there are so
// far. A title that leaks threads looks from outside like a title that leaks
// memory -- every thread carries a stack -- and the two have very different
// fixes. ARC_TRACE_THREADS=1; the address is an image offset, so it names a
// function in the same terms as every other trace here.
void TraceCreate(uint64_t start) {
  static const bool on = getenv("ARC_TRACE_THREADS") != nullptr;
  const long n = ++g_threads_made;
  if (!on) return;
  fprintf(stderr, "[thread] #%ld start=%#llx from", n,
          (unsigned long long)start);
  // Who asked. "A title makes threads it never stops" is a symptom; the
  // function doing the asking is the cause, and the frame ring already knows
  // it -- printed as image offsets, the same currency as every other trace.
  const size_t frames = arc_frame_count();
  for (size_t i = 0; i < frames && i < 8; ++i) {
    const unsigned long long packed = arc_frame_at(i);
    fprintf(stderr, " %llx", packed & 0x00FFFFFFFFFFFFFFull);
  }
  fputc('\n', stderr);
}

int Create(uint64_t* out, const void*, uint64_t start, uint64_t arg) {
  TraceCreate(start);
  if (g_live_threads.load() >= GuestThreadCeiling()) {
    // Said once. The caller retries, and a line per retry buries whatever the
    // run was actually about.
    static std::atomic<bool> said{false};
    if (!said.exchange(true)) {
      // How fast, not just how many. A rate near the frame rate says the
      // caller is doing this once per frame, which is a different bug from
      // one that does it once per request or once per retry -- and it is the
      // first thing worth knowing about a runaway.
      const double secs = std::chrono::duration<double>(
                              std::chrono::steady_clock::now() - g_first_thread)
                              .count();
      fprintf(stderr,
              "\n%ld guest threads are alive and none are finishing -- "
              "refusing to make more.\n%.0f started in %.1f seconds (%.0f a "
              "second).\nRun with ARC_TRACE_THREADS=1 to see which function "
              "is asking, or raise ARC_MAX_THREADS.\n",
              g_live_threads.load(), (double)g_threads_made.load(), secs,
              secs > 0 ? g_threads_made.load() / secs : 0.0);
    }
    return 11 /* EAGAIN */;
  }
  uint32_t id = 0;
  Thread* t = g_threads.Obtain(&id);
  if (!t) return 11 /* EAGAIN */;

  auto* s = new GuestThreadStart{start, arg, id, t};
#if defined(_WIN32)
  // STACK_SIZE_PARAM_IS_A_RESERVATION, and it is not optional. Without the
  // flag CreateThread reads its size argument as how much to *commit*, so a
  // quarter-gigabyte stack is a quarter-gigabyte of real memory per thread
  // before the thread has run an instruction. A title with a thread pool
  // reaches ten gigabytes in seconds and dies -- with no fault, no message,
  // and an exit code that reads like anything else. Reserved, the address
  // space is there for a deep guest call chain and the pages arrive as the
  // stack is actually used.
  t->handle = CreateThread(nullptr, kHostThreadStack, GuestThreadMain, s,
                           STACK_SIZE_PARAM_IS_A_RESERVATION, nullptr);
  if (!t->handle) {
    delete s;
    return 11;
  }
  t->host_id = GetThreadId(static_cast<HANDLE>(t->handle));
#else
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setstacksize(&attr, kHostThreadStack);
  if (pthread_create(&t->handle, &attr, GuestThreadMain, s) != 0) {
    pthread_attr_destroy(&attr);
    delete s;
    return 11;
  }
  pthread_attr_destroy(&attr);
  t->started = true;
#endif
  *out = id;
  return 0;
}

int Join(uint64_t id, void** result) {
  Thread* t = g_threads.Find(static_cast<uint32_t>(id));
  if (!t) return 3 /* ESRCH */;
#if defined(_WIN32)
  if (t->handle) {
    WaitForSingleObject(t->handle, INFINITE);
    CloseHandle(t->handle);
    t->handle = nullptr;
  }
#else
  if (t->started && !t->detached) {
    pthread_join(t->handle, nullptr);
    t->started = false;
  }
#endif
  if (result) *result = t->result;
  return 0;
}

int Detach(uint64_t id) {
  Thread* t = g_threads.Find(static_cast<uint32_t>(id));
  if (!t) return 3;
  t->detached = true;
#if defined(_WIN32)
  if (t->handle) {
    CloseHandle(t->handle);
    t->handle = nullptr;
  }
#else
  if (t->started) {
    pthread_detach(t->handle);
    t->started = false;
  }
#endif
  return 0;
}

uint64_t Self() { return SelfId(); }
int Equal(uint64_t a, uint64_t b) { return a == b; }
int Kill(uint64_t id, int sig) {
  static const bool trace = getenv("ARC_TRACE_SIGNALS") != nullptr;
  if (trace)
    fprintf(stderr, "[kill] self=%u target=%llu sig=%d\n", SelfId(),
            static_cast<unsigned long long>(id), sig);
  if (id == SelfId() || id == 0) {
    arc_signal_post_current(sig);
#if defined(_WIN32)
  } else if (Thread* t = g_threads.Find(static_cast<uint32_t>(id))) {
    if (!t->host_id) {
      if (trace)
        fprintf(stderr, "[kill] target=%llu has no host id\n",
                static_cast<unsigned long long>(id));
      return 0;
    }
    if (trace)
      fprintf(stderr, "[kill] target=%llu wait=%s\n",
              static_cast<unsigned long long>(id),
              t->wait_reason ? t->wait_reason : "-");
    arc_signal_post_host(t->host_id, sig);
#endif
  }
  return 0;
}
int Setname(uint64_t, const char*) { return 0; }
int Setschedparam(uint64_t, int, const void*) { return 0; }

// pthread_exit ends the calling thread. A lifted guest frame cannot be returned
// from here -- the frames on the host stack belong to the lifter -- so the
// thread's entry point armed a setjmp above and this jumps back to it. That is
// the same shape as the guest's own longjmp support, one frame further out.
[[noreturn]] void Exit(void* value) {
  {
    static const bool on = getenv("ARC_TRACE_THREADS") != nullptr;
    if (on) {
      fprintf(stderr, "[thread] pthread_exit id=%u value=%p from",
              t_self_id, value);
      const size_t frames = arc_frame_count();
      for (size_t i = 0; i < frames && i < 8; ++i) {
        const unsigned long long packed = arc_frame_at(i);
        fprintf(stderr, " %llx", packed & 0x00FFFFFFFFFFFFFFull);
      }
      fputc('\n', stderr);
    }
  }
  RunThreadDestructors();
  if (t_exit_jmp) {
    t_exit_value = value;
    longjmp(*t_exit_jmp, 1);
  }
  // The main thread has no RunGuestThread frame to return to, and neither has a
  // host thread the shim made for other reasons. POSIX would end just this
  // thread and let the process live; the closest host equivalent is the same.
  fprintf(stderr,
          "pthread_exit on a thread with no guest trampoline (id=%u value=%p)\n",
          t_self_id, value);
  fflush(stderr);
#if defined(_WIN32)
  ExitThread(0);
#else
  pthread_exit(value);
#endif
}

int AttrInit(uint32_t* a) {
  *a = 0;
  g_attrs.Obtain(a);
  return 0;
}
int AttrDestroy(uint32_t* a) {
  g_attrs.Release(a);
  return 0;
}
int AttrSetdetachstate(uint32_t* a, int state) {
  if (auto* p = g_attrs.Obtain(a)) p->detach_state = state;
  return 0;
}
int AttrSetstacksize(uint32_t* a, size_t size) {
  if (auto* p = g_attrs.Obtain(a)) p->stack_size = size;
  return 0;
}
int AttrSetschedparam(uint32_t*, const void*) { return 0; }
int AttrSetschedpolicy(uint32_t*, int) { return 0; }

static bool TraceSem() {
  static const bool on = getenv("ARC_TRACE_SEM") != nullptr;
  return on;
}
static void TraceSemValue(const char* op, uint32_t* s, int value) {
  if (!TraceSem()) return;
  static thread_local int last = -0x7fffffff;
  if (value == last) return;
  last = value;
  fprintf(stderr, "[sem] %s slot=%p id=%u value=%d self=%u\n", op,
          static_cast<void*>(s), s ? *s : 0, value, t_self_id);
}

int SemInit(uint32_t* s, int, unsigned value) {
  *s = 0;
  if (auto* p = g_semaphores.Obtain(s)) p->count = value;
  if (TraceSem())
    fprintf(stderr, "[sem] init slot=%p id=%u value=%u self=%u\n",
            static_cast<void*>(s), s ? *s : 0, value, t_self_id);
  return 0;
}
int SemDestroy(uint32_t* s) {
  g_semaphores.Release(s);
  return 0;
}
int SemPost(uint32_t* s) {
  if (auto* p = g_semaphores.Obtain(s)) {
    std::lock_guard<std::mutex> g(p->m);
    ++p->count;
    p->cv.notify_one();
    TraceSemValue("post", s, static_cast<int>(p->count));
  }
  return 0;
}
int SemWait(uint32_t* s) {
  if (TraceSem())
    fprintf(stderr, "[sem] wait enter slot=%p id=%u self=%u\n",
            static_cast<void*>(s), s ? *s : 0, t_self_id);
  if (auto* p = g_semaphores.Obtain(s)) {
    std::unique_lock<std::mutex> lk(p->m);
    SetWaitReason("sem_wait");
    while (p->count == 0) {
      if (arc_signal_pending_current()) {
        lk.unlock();
        arc_run_pending_signals();
        lk.lock();
        continue;
      }
      p->cv.wait_for(lk, std::chrono::milliseconds(5));
    }
    --p->count;
    SetWaitReason(nullptr);
    TraceSemValue("wait", s, static_cast<int>(p->count));
  }
  return 0;
}
int SemTrywait(uint32_t* s) {
  auto* p = g_semaphores.Obtain(s);
  if (!p) {
    errno = 22 /* EINVAL */;
    return -1;
  }
  std::lock_guard<std::mutex> g(p->m);
  if (p->count == 0) {
    errno = 11 /* EAGAIN */;
    return -1;
  }
  --p->count;
  return 0;
}

// Bionic's sem_timedwait is an absolute CLOCK_REALTIME wait and returns -1
// with errno set, not an error number. The Boehm collector in libil2cpp waits
// for a thread with a ten-millisecond deadline in a loop and reads __errno to
// tell EINTR/ETIMEDOUT from a real failure; the previous forwarding stub
// answered -1 and left errno alone, so the first timeout looked like a
// broken semaphore and the collector aborted the process.
int SemTimedwait(uint32_t* s, const GuestTimespec* abstime) {
  auto* p = g_semaphores.Find(*s);
  if (!p || !abstime) {
    errno = 22 /* EINVAL */;
    return -1;
  }
  std::unique_lock<std::mutex> lk(p->m);
  if (p->count > 0) {
    --p->count;
    return 0;
  }
  const auto since_epoch = std::chrono::seconds(abstime->tv_sec) +
                           std::chrono::nanoseconds(abstime->tv_nsec);
  const auto deadline = std::chrono::system_clock::time_point(
      std::chrono::duration_cast<std::chrono::system_clock::duration>(
          since_epoch));
  SetWaitReason("sem_timedwait");
  while (p->count == 0) {
    if (arc_signal_pending_current()) {
      lk.unlock();
      arc_run_pending_signals();
      lk.lock();
      continue;
    }
    const auto now = std::chrono::system_clock::now();
    if (now >= deadline) {
      SetWaitReason(nullptr);
      errno = 110 /* ETIMEDOUT */;
      if (TraceSem()) {
        static thread_local unsigned n = 0;
        if ((n++ % 1000) == 0)
          fprintf(stderr, "[sem] timedwait timeout slot=%p id=%u self=%u\n",
                  static_cast<void*>(s), *s, t_self_id);
      }
      return -1;
    }
    auto slice = now + std::chrono::milliseconds(5);
    if (slice > deadline) slice = deadline;
    p->cv.wait_until(lk, slice);
  }
  --p->count;
  SetWaitReason(nullptr);
  TraceSemValue("timedwait", s, static_cast<int>(p->count));
  return 0;
}

int SemGetvalue(uint32_t* s, int* value) {
  auto* p = g_semaphores.Find(*s);
  if (!p || !value) {
    errno = 22 /* EINVAL */;
    return -1;
  }
  std::lock_guard<std::mutex> g(p->m);
  *value = static_cast<int>(p->count);
  TraceSemValue("getvalue", s, *value);
  return 0;
}

int SchedYield() {
  std::this_thread::yield();
  return 0;
}
int SchedGetPriorityMin(int) { return 0; }

struct Entry {
  const char* name;
  void* fn;
};

#define E(sym, fn) {sym, reinterpret_cast<void*>(&fn)}
const Entry kTable[] = {
    E("pthread_mutex_init", MutexInit),
    E("pthread_mutex_destroy", MutexDestroy),
    E("pthread_mutex_lock", MutexLock),
    E("pthread_mutex_unlock", MutexUnlock),
    E("pthread_mutex_trylock", MutexTrylock),
    E("pthread_mutex_timedlock", MutexTimedlock),
    E("pthread_mutexattr_init", MutexattrInit),
    E("pthread_mutexattr_destroy", MutexattrDestroy),
    E("pthread_mutexattr_settype", MutexattrSettype),
    E("pthread_condattr_init", CondattrInit),
    E("pthread_condattr_destroy", CondattrDestroy),
    E("pthread_condattr_setclock", CondattrSetclock),
    E("pthread_cond_init", CondInit),
    E("pthread_cond_destroy", CondDestroy),
    E("pthread_cond_signal", CondSignal),
    E("pthread_cond_broadcast", CondBroadcast),
    E("pthread_cond_wait", CondWait),
    E("pthread_cond_timedwait", CondTimedwait),
    E("pthread_rwlock_init", RwlockInit),
    E("pthread_rwlock_destroy", RwlockDestroy),
    E("pthread_rwlock_rdlock", RwlockRdlock),
    E("pthread_rwlock_wrlock", RwlockWrlock),
    E("pthread_rwlock_unlock", RwlockUnlock),
    E("pthread_once", Once),
    E("pthread_key_create", KeyCreate),
    E("pthread_attr_getdetachstate", AttrGetdetachstate),
    E("pthread_key_delete", KeyDelete),
    E("pthread_getspecific", Getspecific),
    E("pthread_setspecific", Setspecific),
    E("pthread_create", Create),
    E("pthread_join", Join),
    E("pthread_detach", Detach),
    E("pthread_self", Self),
    E("pthread_equal", Equal),
    E("pthread_kill", Kill),
    E("pthread_exit", Exit),
    E("pthread_setname_np", Setname),
    E("pthread_setschedparam", Setschedparam),
    E("pthread_attr_init", AttrInit),
    E("pthread_attr_destroy", AttrDestroy),
    E("pthread_attr_setdetachstate", AttrSetdetachstate),
    E("pthread_attr_setstacksize", AttrSetstacksize),
    E("pthread_attr_setschedparam", AttrSetschedparam),
    E("pthread_attr_setschedpolicy", AttrSetschedpolicy),
    E("sem_init", SemInit),
    E("sem_destroy", SemDestroy),
    E("sem_post", SemPost),
    E("sem_wait", SemWait),
    E("sem_trywait", SemTrywait),
    E("sem_timedwait", SemTimedwait),
    E("sem_getvalue", SemGetvalue),
    E("sched_yield", SchedYield),
    E("sched_get_priority_min", SchedGetPriorityMin),
};
#undef E

}  // namespace

uint64_t ShimResolvePthread(const char* name) {
  for (const Entry& e : kTable)
    if (strcmp(e.name, name) == 0) return reinterpret_cast<uint64_t>(e.fn);
  return 0;
}

size_t ShimPthreadCount() { return sizeof(kTable) / sizeof(kTable[0]); }

}  // namespace arc
