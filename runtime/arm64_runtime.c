// Runtime support for lifted code that cannot be expressed inline.
//
// `arc_dispatch` is deliberately not here: it needs the address -> function
// table, which only the generated module knows, so the generated code defines
// it.

#include "arm64_context.h"
#include "arc_signal.h"

// Implemented in arc_guard.cpp. A host shim may use C++ internally; its
// exceptions must not unwind through lifted frames, which have no Microsoft
// unwind metadata.
uint64_t arc_call_native12_guarded(uint64_t fn, const char* name, uint64_t a0,
                                   uint64_t a1, uint64_t a2, uint64_t a3,
                                   uint64_t a4, uint64_t a5, uint64_t a6,
                                   uint64_t a7, uint64_t a8, uint64_t a9,
                                   uint64_t a10, uint64_t a11);
void arc_call_ctx_native_guarded(uint64_t fn, const char* name, Arm64Ctx* c);

#include <errno.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <time.h>
#endif

#if defined(_MSC_VER)
#include <intrin.h>
#define ARC_THREAD_LOCAL __declspec(thread)
#else
#define ARC_THREAD_LOCAL __thread
#endif

// One reservation per thread, which is what the architecture gives you.
static ARC_THREAD_LOCAL Arm64Reservation t_reservation;

uint64_t arc_load_exclusive(Arm64Ctx* c, uint64_t addr, int bytes) {
  (void)c;
  uint64_t v = 0;
  switch (bytes) {
    case 1: v = arc_ld8(addr); break;
    case 2: v = arc_ld16(addr); break;
    case 4: v = arc_ld32(addr); break;
    default: v = arc_ld64(addr); break;
  }
  t_reservation.address = addr;
  t_reservation.value = v;
  t_reservation.valid = 1;
  return v;
}

static int compare_exchange(uint64_t addr, uint64_t expected, uint64_t desired,
                            int bytes) {
#if defined(_MSC_VER)
  switch (bytes) {
    case 1:
      return _InterlockedCompareExchange8((char*)(uintptr_t)addr,
                                          (char)desired,
                                          (char)expected) == (char)expected;
    case 2:
      return _InterlockedCompareExchange16((short*)(uintptr_t)addr,
                                           (short)desired,
                                           (short)expected) == (short)expected;
    case 4:
      return _InterlockedCompareExchange((long*)(uintptr_t)addr, (long)desired,
                                         (long)expected) == (long)expected;
    default:
      return _InterlockedCompareExchange64((__int64*)(uintptr_t)addr,
                                           (__int64)desired,
                                           (__int64)expected) ==
             (__int64)expected;
  }
#else
  switch (bytes) {
    case 1:
      return __sync_bool_compare_and_swap((uint8_t*)(uintptr_t)addr,
                                          (uint8_t)expected, (uint8_t)desired);
    case 2:
      return __sync_bool_compare_and_swap((uint16_t*)(uintptr_t)addr,
                                          (uint16_t)expected,
                                          (uint16_t)desired);
    case 4:
      return __sync_bool_compare_and_swap((uint32_t*)(uintptr_t)addr,
                                          (uint32_t)expected,
                                          (uint32_t)desired);
    default:
      return __sync_bool_compare_and_swap((uint64_t*)(uintptr_t)addr, expected,
                                          desired);
  }
#endif
}

// Returns 0 on success, 1 on failure -- the same sense as the instruction's
// status register.
uint32_t arc_store_exclusive(Arm64Ctx* c, uint64_t addr, uint64_t value,
                             int bytes) {
  (void)c;
  if (!t_reservation.valid || t_reservation.address != addr) return 1;
  t_reservation.valid = 0;
  return compare_exchange(addr, t_reservation.value, value, bytes) ? 0 : 1;
}

// --- high half of a 128-bit product ----------------------------------------
// No portable C type holds this, so it is the compiler's intrinsic where one
// exists and long multiplication where it does not.
uint64_t arc_umulh(uint64_t a, uint64_t b) {
#if defined(_MSC_VER) && defined(_M_X64)
  return __umulh(a, b);
#elif defined(__SIZEOF_INT128__)
  return (uint64_t)(((unsigned __int128)a * b) >> 64);
#else
  uint64_t alo = a & 0xFFFFFFFFu, ahi = a >> 32;
  uint64_t blo = b & 0xFFFFFFFFu, bhi = b >> 32;
  uint64_t mid = ahi * blo + ((alo * blo) >> 32);
  return ahi * bhi + (mid >> 32) + ((alo * bhi + (mid & 0xFFFFFFFFu)) >> 32);
#endif
}

uint64_t arc_smulh(uint64_t a, uint64_t b) {
#if defined(_MSC_VER) && defined(_M_X64)
  return (uint64_t)__mulh((__int64)a, (__int64)b);
#elif defined(__SIZEOF_INT128__)
  return (uint64_t)((((__int128)(int64_t)a * (int64_t)b)) >> 64);
#else
  // Unsigned high, corrected for each operand's sign.
  uint64_t hi = arc_umulh(a, b);
  if ((int64_t)a < 0) hi -= b;
  if ((int64_t)b < 0) hi -= a;
  return hi;
#endif
}

// The thread pointer. Bionic points TPIDR_EL0 at a per-thread control block,
// and compiled code indexes off it and dereferences without checking -- so
// returning zero is not "no value", it is a null pointer with a small offset
// added, which faults at a low address that looks like a lifter bug.
//
// Nothing here reads meaningful values out of the block; the guest wants it for
// its own thread-local storage. It only has to be real, zeroed, writable memory
// of a plausible size.
#define ARC_TLS_BLOCK 16384
static ARC_THREAD_LOCAL uint64_t t_tpidr;

uint64_t arc_tpidr_read(void) {
  if (!t_tpidr) t_tpidr = (uint64_t)(uintptr_t)calloc(1, ARC_TLS_BLOCK);
  return t_tpidr;
}

void arc_tpidr_write(uint64_t v) { t_tpidr = v; }

uint64_t arc_cntvct_read(void) {
#if defined(_WIN32)
  LARGE_INTEGER now, freq;
  QueryPerformanceCounter(&now);
  QueryPerformanceFrequency(&freq);
  return (uint64_t)((now.QuadPart * 1000000000LL) / freq.QuadPart);
#else
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
#endif
}

static ARC_THREAD_LOCAL uint64_t t_fpcr;
uint64_t arc_fpcr_read(void) { return t_fpcr; }
void arc_fpcr_write(uint64_t v) { t_fpcr = v; }

// The guest's jmp_buf is its own memory and its own size; ours has to live
// somewhere we control, so the two are paired here rather than laid on top of
// one another. Sixteen is generous for nesting -- a decoder arms one and an
// unwinder maybe another -- and reusing the oldest is better than refusing,
// because a refused setjmp is a jump that lands nowhere later.
#define ARC_JMP_SLOTS 16
typedef struct {
  uint64_t guest;
  jmp_buf host;
  /* x19-x30 and the stack pointer: what the architecture says a jump puts
     back, and what the returned-to function's own frame check depends on. */
  uint64_t saved[12];
  uint64_t sp;
  int used;
} ArcJmpSlot;
static ARC_THREAD_LOCAL ArcJmpSlot t_jmps[ARC_JMP_SLOTS];
static ARC_THREAD_LOCAL int t_jmp_next;

static void arc_jmp_save(ArcJmpSlot* slot, const Arm64Ctx* c) {
  int r;
  for (r = 19; r <= 30; ++r) slot->saved[r - 19] = c->x[r];
  slot->sp = c->sp;
}

static void arc_jmp_restore(const ArcJmpSlot* slot, Arm64Ctx* c) {
  int r;
  for (r = 19; r <= 30; ++r) c->x[r] = slot->saved[r - 19];
  c->sp = slot->sp;
}

void* arc_jmpbuf_for(Arm64Ctx* c, uint64_t guest_buffer) {
  int i;
  for (i = 0; i < ARC_JMP_SLOTS; ++i)
    if (t_jmps[i].used && t_jmps[i].guest == guest_buffer) {
      arc_jmp_save(&t_jmps[i], c);
      return &t_jmps[i].host;
    }
  for (i = 0; i < ARC_JMP_SLOTS; ++i)
    if (!t_jmps[i].used) {
      t_jmps[i].used = 1;
      t_jmps[i].guest = guest_buffer;
      arc_jmp_save(&t_jmps[i], c);
      return &t_jmps[i].host;
    }
  i = t_jmp_next;
  t_jmp_next = (t_jmp_next + 1) % ARC_JMP_SLOTS;
  t_jmps[i].guest = guest_buffer;
  arc_jmp_save(&t_jmps[i], c);
  return &t_jmps[i].host;
}

void arc_longjmp(Arm64Ctx* c, uint64_t guest_buffer, int value) {
  int i;
  for (i = 0; i < ARC_JMP_SLOTS; ++i)
    if (t_jmps[i].used && t_jmps[i].guest == guest_buffer) {
      arc_jmp_restore(&t_jmps[i], c);
      longjmp(t_jmps[i].host, value ? value : 1);
    }
  // Nothing armed this on this thread. Saying so beats jumping somewhere
  // plausible: a longjmp to the wrong frame is a crash with no explanation
  // attached to it.
  {
    char msg[96];
    snprintf(msg, sizeof(msg),
             "longjmp(%d) to a buffer no setjmp on this thread armed", value);
    arc_trap(NULL, msg);
  }
}

static ARC_THREAD_LOCAL jmp_buf* t_recovery;
static ARC_THREAD_LOCAL char t_last_trap[512];
// Declared here so a trap can tell the boot thread's recovery point (where the
// caller prints a full backtrace) from a worker's, which dies with one line.
static ARC_THREAD_LOCAL int t_is_main_thread;

void arc_set_recovery(void* jmp_buffer) {
  t_recovery = (jmp_buf*)jmp_buffer;
}

const char* arc_last_trap(void) { return t_last_trap; }

    ARC_THREAD_LOCAL Arm64Ctx t_trap_context;
    ARC_THREAD_LOCAL int t_trap_have;

void arc_trap(Arm64Ctx* c, const char* what) {
  if (c) {
    t_trap_context = *c;
    t_trap_have = 1;
  }
  snprintf(t_last_trap, sizeof(t_last_trap), "%s", what ? what : "?");
  if (t_recovery) {
    /* RunWithRecovery prints its own backtrace for the boot thread. A guest
       worker has nobody above it to do that, so leave one line and the saved
       register file before its setjmp trampoline takes it out. */
    if (!t_is_main_thread) {
      fprintf(stderr, "[trap] worker dying: %s\n", t_last_trap);
      fflush(stderr);
    }
    longjmp(*t_recovery, 1);
  }
  fprintf(stderr, "guest trap: %s\n", t_last_trap);
  abort();
}

// What a trap left behind, for the caller that wants to explain it. The
// runtime reports where control went; only the program on top knows what its
// pointers mean, so the register file is handed over rather than interpreted.
const Arm64Ctx* arc_last_trap_context(void) {
  return t_trap_have ? &t_trap_context : NULL;
}

// --- native call bridge ----------------------------------------------------

typedef struct {
  uint64_t address;
  const char* name;
} NativeEntry;

// ponytail: a linear array. It holds a few hundred imports and is searched only
// on a dispatch miss, which is once per call *out* of the guest. Sort it and
// bisect if a profile ever says otherwise.
#define ARC_MAX_NATIVES 2048
static NativeEntry g_natives[ARC_MAX_NATIVES];
static size_t g_native_count;

typedef struct {
  uint64_t address;
  const char* name;
  ArcCtxFn fn;
} CtxNative;

// Large enough for a whole JNI table -- 233 environment entries plus the
// invocation interface -- with room for the variadic handlers beside it. It
// was 64, which is smaller than the JNI table alone, so every slot past the
// sixty-fourth was dropped. The guest then branched to a real stub address the
// dispatcher had no record of, and the report said the branch went nowhere
// rather than that the table was full.
#define ARC_MAX_CTX_NATIVES 512
static CtxNative g_ctx_natives[ARC_MAX_CTX_NATIVES];
static size_t g_ctx_native_count;

// Every registration bumps this, which invalidates the per-thread dispatch
// caches below without any cross-thread teardown. A cache entry remembers the
// generation it was built in; a late registration simply misses once.
static unsigned g_native_generation = 1;

// Overflowing either table is a configuration mistake, and one that produces a
// symptom pointing anywhere but here. Say so, once, rather than dropping the
// registration in silence.
static void arc_full(const char* which, const char* name) {
  static int said[2];
  const int i = which[0] == 'c' ? 0 : 1;
  if (said[i]) return;
  said[i] = 1;
  fprintf(stderr,
          "arc: the %s table is full; %s and everything after it is "
          "unregistered, and calls to them will look like branches to "
          "nowhere\n",
          which, name ? name : "?");
}

void arc_register_ctx_native(uint64_t address, const char* name, ArcCtxFn fn) {
  if (!address) return;
  if (g_ctx_native_count >= ARC_MAX_CTX_NATIVES) {
    arc_full("context native", name);
    return;
  }
  for (size_t i = 0; i < g_ctx_native_count; ++i)
    if (g_ctx_natives[i].address == address) return;
  g_ctx_natives[g_ctx_native_count].address = address;
  g_ctx_natives[g_ctx_native_count].name = name;
  g_ctx_natives[g_ctx_native_count].fn = fn;
  ++g_ctx_native_count;
  ++g_native_generation;
}

extern int arc_image_owns(uint64_t address);
extern uint64_t arc_image_bases[];

void arc_register_native(uint64_t address, const char* name) {
  if (!address) return;
  if (getenv("ARC_TRACE_NATIVES") && arc_image_owns(address))
    fprintf(stderr, "[native] guest image address registered as %s: %#llx\n",
            name ? name : "?", (unsigned long long)address);
  if (g_native_count >= ARC_MAX_NATIVES) {
    arc_full("native", name);
    return;
  }
  for (size_t i = 0; i < g_native_count; ++i)
    if (g_natives[i].address == address) return;
  g_natives[g_native_count].address = address;
  g_natives[g_native_count].name = name;
  ++g_native_count;
  ++g_native_generation;
}

// Twelve integer arguments in, one integer result out: eight from the
// registers AArch64 passes them in, and four more read from the guest stack,
// where that convention puts the rest.
//
// Eight was not enough, and the way it failed is worth keeping. glTexImage2D
// takes nine -- the ninth being the pixel data -- so the texture was uploaded
// from whatever the driver found in that argument slot. Nothing reported a
// missing argument; the call simply read an address nobody had passed.
//
// Handing a function more arguments than it declares is harmless here: the
// caller sets them up and the callee ignores them. So one thunk serves every
// arity up to twelve rather than needing one per signature.
//
// It still does NOT carry floating-point arguments, which live in v0-v7 and
// would need per-signature thunks to place correctly. Generate those from the
// import list when a title actually needs one.
typedef uint64_t (*ArcNative12)(uint64_t, uint64_t, uint64_t, uint64_t,
                                uint64_t, uint64_t, uint64_t, uint64_t,
                                uint64_t, uint64_t, uint64_t, uint64_t);

#define ARC_TRACE 64
static ARC_THREAD_LOCAL const char* t_trace[ARC_TRACE];
static ARC_THREAD_LOCAL size_t t_trace_next;
static ARC_THREAD_LOCAL size_t t_trace_seen;

void arc_trace_note(const char* what) {
  t_trace[t_trace_next] = what;
  t_trace_next = (t_trace_next + 1) % ARC_TRACE;
  ++t_trace_seen;
}

size_t arc_trace_count(void) {
  return t_trace_seen < ARC_TRACE ? t_trace_seen : ARC_TRACE;
}

const char* arc_trace_at(size_t back) {
  if (back >= arc_trace_count()) return 0;
  size_t i = (t_trace_next + ARC_TRACE - 1 - back) % ARC_TRACE;
  return t_trace[i];
}

void arc_trace_clear(void) {
  t_trace_next = 0;
  t_trace_seen = 0;
}

static ARC_THREAD_LOCAL const char* t_last_native_name;
static ARC_THREAD_LOCAL uint64_t t_last_native_lr;
static ARC_THREAD_LOCAL uint64_t t_last_native_lr_shadow;
static ARC_THREAD_LOCAL uint64_t t_last_native_args[3];

const char* arc_last_native(uint64_t* lr, uint64_t* lr_shadow, uint64_t* a0,
                            uint64_t* a1, uint64_t* a2) {
  if (lr) *lr = t_last_native_lr;
  if (lr_shadow) *lr_shadow = t_last_native_lr_shadow;
  if (a0) *a0 = t_last_native_args[0];
  if (a1) *a1 = t_last_native_args[1];
  if (a2) *a2 = t_last_native_args[2];
  return t_last_native_name;
}

// Global rather than per-thread. A per-thread ring can only be read from the
// thread that filled it, which forces the reporting to happen inside the guest
// call itself -- exactly where it is least safe to do anything. Interleaving
// between threads costs some precision in the trail; being able to read it
// after the fact is worth more.
#define ARC_FRAME_RING 256
static uint64_t t_frames[ARC_FRAME_RING];
static size_t t_frame_next;
static size_t t_frame_seen;

/* The global ring is dominated by whichever worker thread is busiest, which
   hides the thread that is actually waiting. Keep a second ring for the boot
   thread so a watchdog can say what it is blocked in without a guest
   unwinder. */
static uint64_t t_main_frames[ARC_FRAME_RING];
static size_t t_main_frame_next;
static size_t t_main_frame_seen;

/* Exact call chains for the boot thread. The entered-function ring is a
   history and the frame pointer chain goes cold at leaf functions; this one is
   pushed and popped by the dispatcher, so it is a real stack. The watchdog
   reads the boot thread's copy, which is why it is separate from the TLS
   stacks used by workers. */
#define ARC_SHADOW_MAX 96
static ARC_THREAD_LOCAL uint64_t t_shadow[ARC_SHADOW_MAX];
static ARC_THREAD_LOCAL size_t t_shadow_depth;
static uint64_t g_main_shadow[ARC_SHADOW_MAX];
static volatile size_t g_main_shadow_depth;

void arc_shadow_push(uint64_t address) {
  if (t_shadow_depth >= ARC_SHADOW_MAX) return;
  t_shadow[t_shadow_depth++] = address;
  if (t_is_main_thread) {
    g_main_shadow[t_shadow_depth - 1] = address;
    g_main_shadow_depth = t_shadow_depth;
  }
}

void arc_shadow_pop(void) {
  if (!t_shadow_depth) return;
  --t_shadow_depth;
  if (t_is_main_thread) g_main_shadow_depth = t_shadow_depth;
}

size_t arc_main_shadow_depth(void) { return g_main_shadow_depth; }

uint64_t arc_main_shadow_at(size_t back) {
  const size_t depth = g_main_shadow_depth;
  if (back >= depth) return 0;
  return g_main_shadow[depth - 1 - back];
}

unsigned arc_trace_thread(void);

void arc_probe(const char* tag, uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
  static const char* enabled;
  static int looked;  if (!looked) {
    looked = 1;
    enabled = getenv("ARC_PROBE");
  }
  if (!enabled) return;
  fprintf(stderr, "[probe] t%u %s a=%#llx b=%#llx c=%#llx d=%#llx\n",
          arc_trace_thread(), tag, (unsigned long long)a,
          (unsigned long long)b, (unsigned long long)c,
          (unsigned long long)d);
}

/* Identifying an object usually means naming its class, and a class name is
   a plain NUL-terminated char* in guest memory.  arc_probe prints numbers,
   which cannot show a name, so this prints the text next to the pointer. */
void arc_probe_text(const char* tag, uint64_t address) {
  static const char* enabled;
  static int looked;
  if (!looked) {
    looked = 1;
    enabled = getenv("ARC_PROBE");
  }
  if (!enabled) return;
  if (!address || !arc_can_read(address, 1)) {
    fprintf(stderr, "[probe] t%u %s %#llx <unreadable>\n", arc_trace_thread(),
            tag, (unsigned long long)address);
    return;
  }
  const char* text = (const char*)(uintptr_t)address;
  size_t n = 0;
  while (n < 200 && arc_can_read(address + n, 1) && text[n]) ++n;
  fprintf(stderr, "[probe] t%u %s %#llx \"%.*s\"\n", arc_trace_thread(), tag,
          (unsigned long long)address, (int)n, text);
}

void arc_probe_dump(const char* tag, uint64_t address, uint64_t length) {
  const char* base = getenv("ARC_PROBE_DUMP");
  if (!base || !length) return;
  if (!arc_can_read(address, (size_t)length)) {
    fprintf(stderr, "[probe] dump %s: %#llx..%#llx not readable\n", tag,
            (unsigned long long)address,
            (unsigned long long)(address + length));
    return;
  }
  /* One file per tag: a diagnosis usually needs the inputs, the state and the
     output side by side, and a single path would keep only the last. */
  char path[1024];
  snprintf(path, sizeof path, "%s.%s", base, tag);
  FILE* out = fopen(path, "wb");
  if (!out) {
    fprintf(stderr, "[probe] dump %s: cannot write %s\n", tag, path);
    return;
  }
  const size_t written =
      fwrite((const void*)(uintptr_t)address, 1, (size_t)length, out);
  fclose(out);
  fprintf(stderr, "[probe] dump %s %s %zu bytes from %#llx\n", tag, path,
          written, (unsigned long long)address);
}

/* A compiler-generated iterator keeps its state word at +0x10 and its captured
   locals after it. When the state stops changing, the class name of the object
   in the wrap field is the step the boot is waiting on -- a number alone cannot
   say that, and reading the decompiled state machine to guess it takes longer
   than one run that names it.

   A stalled step is asked about every frame for minutes, so this reports a tag
   when its state word changes and every two-thousandth call otherwise. The tag
   is the call site's literal, which is what makes the slot lookup work. */
void arc_probe_iterator(const char* tag, uint64_t object, int fields) {
  static const char* enabled;
  static int looked;
  static const char* seen_tag[16];
  static int seen_state[16];
  static unsigned seen_calls[16];
  static int seen_count;
  if (!looked) {
    looked = 1;
    enabled = getenv("ARC_PROBE");
  }
  if (!enabled || !object || !arc_can_read(object, 0x20)) return;
  int slot = -1;
  for (int i = 0; i < seen_count; ++i) {
    if (seen_tag[i] == tag || (seen_tag[i] && tag && !strcmp(seen_tag[i], tag))) {
      slot = i;
      break;
    }
  }
  if (slot < 0) {
    if (seen_count >= 16) return;
    slot = seen_count++;
    seen_tag[slot] = tag;
    seen_state[slot] = 0x7fffffff;
    seen_calls[slot] = 0;
  }
  ++seen_calls[slot];
  const int state = (int)arc_ld32(object + 0x10);
  if (state == seen_state[slot] && (seen_calls[slot] % 2000) != 0) return;
  seen_state[slot] = state;
  fprintf(stderr, "[probe] t%u %s state=%d object=%#llx call=%u\n",
          arc_trace_thread(), tag, state, (unsigned long long)object,
          seen_calls[slot]);
  if (fields > 16) fields = 16;
  for (int i = 0; i < fields; ++i) {
    const uint64_t value = arc_ld64(object + 0x18 + (uint64_t)i * 8);
    if (!value || (value & 7) || !arc_can_read(value, 0x10)) continue;
    const uint64_t klass = arc_ld64(value);
    if (!klass || (klass & 7) || !arc_can_read(klass + 0x20, 8)) continue;
    const uint64_t name = arc_ld64(klass + 0x10);
    if (!name || !arc_can_read(name, 1)) continue;
    const char* text = (const char*)(uintptr_t)name;
    size_t n = 0;
    while (n < 120 && arc_can_read(name + n, 1) && text[n]) ++n;
    char nsbuf[96];
    nsbuf[0] = 0;
    const uint64_t nsp = arc_ld64(klass + 0x18);
    if (nsp && arc_can_read(nsp, 1)) {
      const char* ns = (const char*)(uintptr_t)nsp;
      size_t k = 0;
      while (k < sizeof(nsbuf) - 2 && arc_can_read(nsp + k, 1) && ns[k]) ++k;
      if (k) {
        nsbuf[0] = '.';
        memcpy(nsbuf + 1, ns, k);
        nsbuf[k + 1] = 0;
      }
    }
    fprintf(stderr, "[probe] t%u %s   +%#x %#llx \"%s%.*s\"\n", arc_trace_thread(),
            tag, 0x18 + i * 8, (unsigned long long)value, nsbuf, (int)n, text);
  }
}

/* Naming the class of a guest object, shared by the probes above and below. */
static void arc_probe_print_class(const char* tag, uint64_t value) {
  if (!value || (value & 7) || !arc_can_read(value, 0x10)) return;
  const uint64_t klass = arc_ld64(value);
  if (!klass || (klass & 7) || !arc_can_read(klass + 0x20, 8)) return;
  const uint64_t name = arc_ld64(klass + 0x10);
  if (!name || !arc_can_read(name, 1)) return;
  const char* text = (const char*)(uintptr_t)name;
  size_t n = 0;
  while (n < 120 && arc_can_read(name + n, 1) && text[n]) ++n;
  fprintf(stderr, "[probe] t%u %s   %#llx \"%.*s\"\n", arc_trace_thread(), tag,
          (unsigned long long)value, (int)n, text);
}

/* A managed string is an object header, an int32 length and UTF-16 data, so it
   is printed the same way the ARC_TRACE_STRING path prints one. */
static void arc_probe_print_string(const char* tag, uint64_t value) {
  if (!value || (value & 7) || !arc_can_read(value, 0x18)) return;
  const int32_t length = (int32_t)arc_ld32(value + 0x10);
  if (length < 0 || length > 512 || !arc_can_read(value + 0x14, 2u * (size_t)length))
    return;
  const uint16_t* chars = (const uint16_t*)(uintptr_t)(value + 0x14);
  char buf[1024];
  size_t n = 0;
  for (int32_t k = 0; k < length && n + 6 < sizeof(buf); ++k) {
    const unsigned c = chars[k];
    if (c >= 0x20 && c < 0x7f) {
      buf[n++] = (char)c;
    } else if (c == '\n' || c == '\r') {
      buf[n++] = ' ';
    } else {
      n += (size_t)snprintf(buf + n, sizeof(buf) - n, "\\u%04x", c);
    }
  }
  buf[n] = 0;
  fprintf(stderr, "[probe] t%u %s   \"%s\"\n", arc_trace_thread(), tag, buf);
}

/* The wait chain is a linked list of task contexts, but the same field holds
   the owning manager on the last one, so a hop only counts when the object
   really is a context.  Both SakashoAsyncTaskContext and its typed variant
   TSakashoAsyncTaskContext`1 carry the same wait fields, so the test is on the
   shared part of the class name rather than on an exact match. */
static int arc_probe_is_class(uint64_t object, const char* want) {
  if (!object || (object & 7) || !arc_can_read(object, 0x10)) return 0;
  const uint64_t klass = arc_ld64(object);
  if (!klass || (klass & 7) || !arc_can_read(klass + 0x18, 8)) return 0;
  const uint64_t name = arc_ld64(klass + 0x10);
  if (!name || !arc_can_read(name, 1)) return 0;
  const char* text = (const char*)(uintptr_t)name;
  const size_t want_len = strlen(want);
  for (size_t i = 0;; ++i) {
    if (!arc_can_read(name + i, want_len + 1)) return 0;
    if (!text[i]) return 0;
    if (memcmp(text + i, want, want_len) == 0) return 1;
  }
}

/* SakashoDirector keeps one SakashoAsyncTaskContext per dispatched sub task at
   +0x70, and the director only finishes initializing when every context says
   IsFinished.  The contexts are what a stall has to be attributed to: the
   waiting task's class is the sub task's own state machine. */
void arc_probe_subtasks(const char* tag, uint64_t director) {
  static const char* enabled;
  static int looked;
  static unsigned calls;
  static uint64_t last_sig;
  if (!looked) {
    looked = 1;
    enabled = getenv("ARC_PROBE");
  }
  if (!enabled || !director || !arc_can_read(director + 0x70, 8)) return;
  const uint64_t list = arc_ld64(director + 0x70);
  if (!list || !arc_can_read(list + 0x20, 8)) return;
  /* The SubTasks container keeps its count at +0x18 and the element slots
     inline from +0x20 (checkAllSubTaskFinished walks x27 = 0x20 .. 0xE8 by 8),
     so the array is not reached through a +0x10 items pointer. */
  const int size = (int)arc_ld32(list + 0x18);
  if (size <= 0 || size > 64 || !arc_can_read(list + 0x20, (size_t)size * 8))
    return;
  uint64_t sig = 1469598103u;
  for (int i = 0; i < size; ++i) {
    /* Each slot holds SubTaskInfo, whose TaskContext is one hop in at +0x18;
       the context is what carries the finished/pending bytes. */
    const uint64_t info = arc_ld64(list + 0x20 + (uint64_t)i * 8);
    sig = (sig ^ (uint64_t)i) * 1099511628211u;
    if (!info || !arc_can_read(info + 0x20, 8)) continue;
    const uint64_t ctx = arc_ld64(info + 0x18);
    sig = (sig ^ (uint64_t)arc_ld8(info + 0x10)) * 1099511628211u;
    if (!ctx || !arc_can_read(ctx + 0x30, 8)) continue;
    sig = (sig ^ (uint64_t)arc_ld8(ctx + 0x10)) * 1099511628211u;
    sig = (sig ^ (uint64_t)arc_ld8(ctx + 0x11)) * 1099511628211u;
  }
  ++calls;
  if (sig == last_sig && (calls % 1000) != 0) return;
  last_sig = sig;
  fprintf(stderr, "[probe] t%u %s contexts=%d call=%u unfinished=%d timedout=%u\n",
          arc_trace_thread(), tag, size, calls,
          (int)arc_ld32(director + 0x80), (unsigned)arc_ld8(director + 0x238));
  for (int i = 0; i < size; ++i) {
    const uint64_t info = arc_ld64(list + 0x20 + (uint64_t)i * 8);
    if (!info || !arc_can_read(info + 0x20, 8)) continue;
    const uint64_t ctx = arc_ld64(info + 0x18);
    const unsigned completed = (unsigned)arc_ld8(info + 0x10);
    if (!ctx || !arc_can_read(ctx + 0x30, 8)) {
      fprintf(stderr, "[probe] t%u %s [%d] info=%#llx completed=%u ctx=<none>\n",
              arc_trace_thread(), tag, i, (unsigned long long)info, completed);
      continue;
    }
    const unsigned finished = (unsigned)arc_ld8(ctx + 0x10);
    const unsigned pending = (unsigned)arc_ld8(ctx + 0x11);
    const int id = (int)arc_ld32(ctx + 0x20);
    fprintf(stderr,
            "[probe] t%u %s [%d] info=%#llx completed=%u ctx=%#llx finished=%u "
            "pending=%u id=%d\n",
            arc_trace_thread(), tag, i, (unsigned long long)info, completed,
            (unsigned long long)ctx, finished, pending, id);
    if (!finished) {
      /* +0x18 is the Shelp.Result the context finished with (empty while the
         task runs), +0x28 the task object it is waiting on: its class names
         the coroutine that has to make progress. */
      arc_probe_print_class(tag, arc_ld64(ctx + 0x18));
      uint64_t hop = arc_ld64(ctx + 0x28);
      int depth = 0;
      for (; depth < 6; ++depth) {
        if (!arc_probe_is_class(hop, "SakashoAsyncTaskContext")) break;
        if (!arc_can_read(hop + 0x48, 8)) break;
        fprintf(stderr,
                "[probe] t%u %s [%d] wait%d=%#llx id=%d orig=%d fin=%u "
                "pending=%u flag=%u wait=%#llx\n",
                arc_trace_thread(), tag, i, depth, (unsigned long long)hop,
                (int)arc_ld32(hop + 0x20), (int)arc_ld32(hop + 0x24),
                (unsigned)arc_ld8(hop + 0x10), (unsigned)arc_ld8(hop + 0x11),
                (unsigned)arc_ld8(hop + 0x12),
                (unsigned long long)arc_ld64(hop + 0x28));
        const uint64_t next = arc_ld64(hop + 0x28);
        if (next == hop) break;
        hop = next;
      }
      if (hop) {
        fprintf(stderr, "[probe] t%u %s [%d] end%d=%#llx\n", arc_trace_thread(),
                tag, i, depth, (unsigned long long)hop);
        arc_probe_print_class(tag, hop);
      }
    }
  }
}

/* Which context is being started or finished, by identity rather than by
   address: ContextId and OriginalContextId are what tie a nested wait back to
   the sub task that dispatched it. */
void arc_probe_ctx(const char* tag, uint64_t ctx) {
  static const char* enabled;
  static int looked;
  static unsigned calls;
  if (!looked) {
    looked = 1;
    enabled = getenv("ARC_PROBE");
  }
  if (!enabled) return;
  if (++calls > 4000) return;
  if (!ctx || (ctx & 7) || !arc_can_read(ctx + 0x30, 8)) {
    fprintf(stderr, "[probe] t%u %s ctx=%#llx <unreadable>\n",
            arc_trace_thread(), tag, (unsigned long long)ctx);
    return;
  }
  fprintf(stderr,
          "[probe] t%u %s ctx=%#llx id=%d orig=%d fin=%u pending=%u flag=%u "
          "wait=%#llx\n",
          arc_trace_thread(), tag, (unsigned long long)ctx,
          (int)arc_ld32(ctx + 0x20), (int)arc_ld32(ctx + 0x24),
          (unsigned)arc_ld8(ctx + 0x10), (unsigned)arc_ld8(ctx + 0x11),
          (unsigned)arc_ld8(ctx + 0x12), (unsigned long long)arc_ld64(ctx + 0x28));
}

/* An object whose class is the whole answer, for call sites that are only
   reached on one branch. */
void arc_probe_obj(const char* tag, uint64_t object) {
  static const char* enabled;
  static int looked;
  static unsigned calls;
  if (!looked) {
    looked = 1;
    enabled = getenv("ARC_PROBE");
  }
  if (!enabled) return;
  if (++calls > 4000) return;
  fprintf(stderr, "[probe] t%u %s obj=%#llx\n", arc_trace_thread(), tag,
          (unsigned long long)object);
  arc_probe_print_class(tag, object);
}

/* A Shelp request is identified by its module/method id pair, not by address:
   the module id says which API family the request belongs to, and +0x7c is the
   queue state (1 queued, 2 dispatched, 4 retry wait, 5 cancelled).  Logging
   that triple at the queue entry, the dispatch and the completion is enough to
   tell a request that never left the queue from one that was answered. */
void arc_probe_req(const char* tag, uint64_t req) {
  static const char* enabled;
  static int looked;
  static unsigned calls;
  if (!looked) {
    looked = 1;
    enabled = getenv("ARC_PROBE");
  }
  if (!enabled) return;
  if (++calls > 4000) return;
  if (!req || (req & 7) || !arc_can_read(req + 0x80, 8)) {
    fprintf(stderr, "[probe] t%u %s req=%#llx <unreadable>\n",
            arc_trace_thread(), tag, (unsigned long long)req);
    return;
  }
  fprintf(stderr,
          "[probe] t%u %s req=%#llx module=%d method=%d state=%d errors=%d "
          "id=%d\n",
          arc_trace_thread(), tag, (unsigned long long)req,
          (int)arc_ld32(req + 0x3c), (int)arc_ld32(req + 0x40),
          (int)arc_ld32(req + 0x7c), (int)arc_ld32(req + 0x60),
          (int)arc_ld32(req + 0x78));
}

/* A call site that only needs to say "reached, with this value".  Used for
   walking a chain of functions where the first one that never appears is the
   one that stopped the work. */
/* One field of one object, resolved to an offset inside libil2cpp when it is
   a code pointer.  This is how a delegate's method is named: the pointer is
   the image base plus the address the method table lists. */
void arc_probe_slot(const char* tag, uint64_t object, uint64_t offset) {
  static const char* enabled;
  static int looked;
  static unsigned calls;
  uint64_t value = 0;
  if (!looked) {
    looked = 1;
    enabled = getenv("ARC_PROBE");
  }
  if (!enabled) return;
  if (++calls > 20000) return;
  if (object && arc_can_read(object + offset, 8))
    value = arc_ld64(object + offset);
  if (value >= arc_image_bases[0] &&
      value < arc_image_bases[0] + UINT64_C(0x7800000))
    fprintf(stderr, "[probe] t%u %s libil2cpp+%#llx\n", arc_trace_thread(),
            tag, (unsigned long long)(value - arc_image_bases[0]));
  else
    fprintf(stderr, "[probe] t%u %s %#llx\n", arc_trace_thread(), tag,
            (unsigned long long)value);
}

void arc_probe_val(const char* tag, uint64_t value) {
  static const char* enabled;
  static int looked;
  static unsigned calls;
  if (!looked) {
    looked = 1;
    enabled = getenv("ARC_PROBE");
  }
  if (!enabled) return;
  if (++calls > 20000) return;
  fprintf(stderr, "[probe] t%u %s %#llx\n", arc_trace_thread(), tag,
          (unsigned long long)value);
}

/* The child context a yield instruction owns is created before the instruction
   is posted, so this runs for every awaited work item in a boot.  The method
   name is the work item: a child that never leaves 0/0 is a work item that was
   never invoked, which is a different bug from one that ran and threw. */
void arc_probe_task_start(const char* tag, uint64_t yieldInstr, uint64_t outer,
                          uint64_t method) {
  static const char* enabled;
  static int looked;
  static unsigned calls;
  if (!looked) {
    looked = 1;
    enabled = getenv("ARC_PROBE");
  }
  if (!enabled) return;
  if (++calls > 600) return;
  const uint64_t klass = (method && (method & 7) == 0) ? arc_ld64(method + 0x20) : 0;
  const uint64_t name = (method && (method & 7) == 0) ? arc_ld64(method + 0x18) : 0;
  fprintf(stderr,
          "[probe] t%u %s #%u yield=%#llx outer=%#llx method=%#llx klass=%#llx\n",
          arc_trace_thread(), tag, calls, (unsigned long long)yieldInstr,
          (unsigned long long)outer, (unsigned long long)method,
          (unsigned long long)klass);
  arc_probe_print_class(tag, yieldInstr);
  if (name) arc_probe_text(tag, name);
  /* A klass is not an object: its name is at +0x10, not behind its first word. */
  if (klass && (klass & 7) == 0 && arc_can_read(klass + 0x18, 8)) {
    const uint64_t kname = arc_ld64(klass + 0x10);
    if (kname) arc_probe_text(tag, kname);
  }
}

void arc_mark_main_thread(void) {
  t_is_main_thread = 1;
  t_main_frame_next = 0;
  t_main_frame_seen = 0;
  t_shadow_depth = 0;
  g_main_shadow_depth = 0;
}

size_t arc_main_frame_count(void) {
  return t_main_frame_seen < ARC_FRAME_RING ? t_main_frame_seen : ARC_FRAME_RING;
}

uint64_t arc_main_frame_at(size_t back) {
  if (back >= arc_main_frame_count()) return 0;
  return t_main_frames[(t_main_frame_next + ARC_FRAME_RING - 1 - back) %
                       ARC_FRAME_RING];
}

size_t arc_main_frame_seen(void) { return t_main_frame_seen; }

/* Which thread a trace line came from. Both traces below are written by
   every guest thread at once, and interleaved they answer nothing: the
   question is always what one thread did in order, and a title does its
   networking on a worker while the renderer fills the log. A small number
   rather than the OS id because it has to be readable in a grep. */
static ARC_THREAD_LOCAL unsigned t_trace_id;
static unsigned g_trace_ids;

unsigned arc_trace_thread(void) {
  if (!t_trace_id) t_trace_id = ++g_trace_ids;
  return t_trace_id;
}

/* ARC_TRACE_GUEST prints every guest function entered, which for a title
   that has booted is millions of lines a second -- a firehose that costs far
   more than the thing it is watching, and buries the one call the question is
   about. ARC_TRACE_FN names that call instead: a comma-separated list of
   packed values in hex, and only those are printed. Same trace, with a
   filter, so a question about one function costs one function's worth of
   output. */
static int arc_traced_fn(uint64_t packed) {
  static const char* list;
  static int looked;
  const char* p;
  if (!looked) {
    looked = 1;
    list = getenv("ARC_TRACE_FN");
  }
  if (!list) return 1;
  for (p = list; *p;) {
    char* end;
    unsigned long long v = strtoull(p, &end, 16);
    if (end != p && v == packed) return 1;
    p = (end != p) ? end : p + 1;
    while (*p && *p != ',') ++p;
    while (*p == ',') ++p;
  }
  return 0;
}

static int arc_can_read(uint64_t p, size_t n) {
#if defined(_WIN32)
  MEMORY_BASIC_INFORMATION mbi;
  if (!p || !n) return 0;
  if (!VirtualQuery((const void*)(uintptr_t)p, &mbi, sizeof mbi)) return 0;
  if (mbi.State != MEM_COMMIT || (mbi.Protect & PAGE_GUARD)) return 0;
  if (!(mbi.Protect & (PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY |
                       PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE |
                       PAGE_EXECUTE_WRITECOPY)))
    return 0;
  return p <= (uint64_t)(uintptr_t)mbi.BaseAddress + mbi.RegionSize &&
         n <= (uint64_t)(uintptr_t)mbi.BaseAddress + mbi.RegionSize - p;
#else
  return p != 0;
#endif
}

#if defined(ARC_FRAMES)
void arc_frame_note(uint64_t packed) {
  /* Every lifted body calls this on entry, and the loader's deserialiser
     enters a body millions of times. The ring is what a fault report reads, so
     it stays; the per-entry filters below are diagnostics, and when none is
     armed they are pure overhead. Read the environment once and take the ring
     store alone in the common case. */
  static int diag = -1;
  if (diag < 0)
    diag = getenv("ARC_TRACE_GUEST") || getenv("ARC_TRACE_FN") ||
           getenv("ARC_TRACE_LR") || getenv("ARC_TRACE_STRING") ||
           getenv("ARC_TRACE_FIELD") || getenv("ARC_TRACE_ARGS") ||
           getenv("ARC_TRACE_METHODS") || getenv("ARC_TRACE_ALLOC") ||
           getenv("ARC_FN_CENSUS") || getenv("ARC_FN_CENSUS_FROM");
  if (!diag) {
    t_frames[t_frame_next] = packed;
    t_frame_next = (t_frame_next + 1) % ARC_FRAME_RING;
    ++t_frame_seen;
    /* The second ring exists only so the profiler can read the boot thread's
       history from another thread. Keep it when the sampler (or the watchdog)
       is on and skip it otherwise: on the boot thread the write is two more
       stores and a branch per lifted entry, and a hot loop enters millions of
       bodies per second. */
    static int main_ring = -1;
    if (main_ring < 0)
      main_ring = getenv("ARC_SAMPLE") != NULL || getenv("ARC_WATCHDOG") != NULL;
    if (t_is_main_thread && main_ring) {
      t_main_frames[t_main_frame_next] = packed;
      t_main_frame_next = (t_main_frame_next + 1) % ARC_FRAME_RING;
      ++t_main_frame_seen;
    }
    return;
  }
  /* The ring holds sixteen frames, which says where a fault happened and
     nothing about how a call that returned cleanly spent its time. A whole
     trace answers the other question: which branch a state machine took. */
  static int trace = -1;
  if (trace < 0)
    trace = getenv("ARC_TRACE_GUEST") != NULL || getenv("ARC_TRACE_FN") != NULL;
  /* errno is saved across this: the trace fires on every guest function
     entry, including the one between a call that set errno and the guest
     reading it, and fprintf may set errno itself. Without this a traced
     run is not a run of the same program -- which is the one thing a
     diagnostic must never be. */
  if (trace && arc_traced_fn(packed)) {
    const int saved = errno;
    fprintf(stderr, "[fn] t%u %llx\n", arc_trace_thread(),
            (unsigned long long)packed);
    errno = saved;
  }
  /* A string argument is the one thing a register trace cannot show. ARC_TRACE_STRING
     names a function, the register that carries the string it was handed, and
     optionally the fields to step through first, `<packed function>:<register>
     [+<offset>...]`, comma-separated: the pointer alone says a key was passed,
     and a key that is the wrong length or empty is the whole difference between
     a decrypt that works and one that throws. The hops reach what a function is
     really given -- an exception object is not a string, but its message at
     +0x10 is. The layout is the managed ABI's (class, monitor, int32 length,
     UTF-16 data at 0x14). */
  {
    /* ARC_TRACE_LR names the functions whose *caller* is the question. A throw,
       a null check and an allocation helper are each one shared body reached
       from hundreds of sites, so the fact that it ran says nothing about where
       it was called from; x30 at entry is the one register that still knows.
       Same `image<<56|offset` values as ARC_TRACE_FN, comma-separated. */
    static uint64_t lr_fn[8];
    static int lr_count = -1;
    if (lr_count < 0) {
      lr_count = 0;
      const char* list = getenv("ARC_TRACE_LR");
      for (const char* p = list; p && *p && lr_count < 8;) {
        char* end;
        const uint64_t fn = strtoull(p, &end, 16);
        if (end != p) lr_fn[lr_count++] = fn;
        while (*end && *end != ',') ++end;
        p = (*end == ',') ? end + 1 : end;
      }
    }
    for (int i = 0; i < lr_count; ++i) {
      if (lr_fn[i] != packed) continue;
      Arm64Ctx* sc = arc_current_guest_context();
      const int saved = errno;
      fprintf(stderr, "[lr] t%u %llx caller=%#llx\n", arc_trace_thread(),
              (unsigned long long)packed,
              (unsigned long long)(sc ? sc->x[30] : 0));
      errno = saved;
    }
  }
  {
    static uint64_t string_fn[8];
    static int string_reg[8];
    static uint64_t string_hop[8][3];
    static int string_depth[8];
    static int string_array[8];
    static int string_count = -1;
    if (string_count < 0) {
      string_count = 0;
      const char* list = getenv("ARC_TRACE_STRING");
      for (const char* p = list; p && *p && string_count < 8;) {
        char* end;
        const uint64_t fn = strtoull(p, &end, 16);
        if (end == p) {
          ++p;
          continue;
        }
        if (*end == ':') {
          const int reg = (int)strtol(end + 1, &end, 10);
          if (reg >= 0 && reg < 31) {
            string_fn[string_count] = fn;
            string_reg[string_count] = reg;
            string_depth[string_count] = 0;
            /* `a` reads a managed array instead of a string: a key handed to a
               cipher is a byte[], whose length sits at 0x18 and whose data
               starts at 0x20. */
            string_array[string_count] = 0;
            if (*end == 'a') {
              string_array[string_count] = 1;
              ++end;
            }
            while (*end == '+' && string_depth[string_count] < 3) {
              const uint64_t hop = strtoull(end + 1, &end, 16);
              if (end == p) break;
              string_hop[string_count][string_depth[string_count]++] = hop;
            }
            ++string_count;
          }
        }
        while (*end && *end != ',') ++end;
        p = (*end == ',') ? end + 1 : end;
      }
    }
    for (int i = 0; i < string_count; ++i) {
      if (string_fn[i] != packed) continue;
      Arm64Ctx* sc = arc_current_guest_context();
      uint64_t str = sc ? sc->x[string_reg[i]] : 0;
      const int saved = errno;
      fprintf(stderr, "[str] t%u %llx lr=%llx arg%d=%#llx", arc_trace_thread(),
              (unsigned long long)packed,
              (unsigned long long)(sc ? sc->x[30] : 0), string_reg[i],
              (unsigned long long)str);
      for (int d = 0; d < string_depth[i] && str; ++d) {
        const uint64_t at = str + string_hop[i][d];
        if (!arc_can_read(at, 8)) {
          str = 0;
          break;
        }
        str = *(const uint64_t*)(uintptr_t)at;
        fprintf(stderr, " +%#llx=%#llx", (unsigned long long)string_hop[i][d],
                (unsigned long long)str);
      }
      fprintf(stderr, " ");
      if (string_array[i]) {
        if (str && arc_can_read(str, 0x28)) {
          const uint64_t length = *(const uint64_t*)(uintptr_t)(str + 0x18);
          const uint64_t shown = length > 48 ? 48 : length;
          fprintf(stderr, "[%llu] ", (unsigned long long)length);
          for (uint64_t k = 0; k < shown; ++k) {
            uint8_t byte = 0;
            if (!arc_can_read(str + 0x20 + k, 1)) break;
            byte = *(const uint8_t*)(uintptr_t)(str + 0x20 + k);
            fprintf(stderr, "%02x", byte);
          }
          if (length > shown) fprintf(stderr, "...");
        }
        fprintf(stderr, "\n");
        errno = saved;
        continue;
      }
      if (str && arc_can_read(str, 0x20)) {
        /* A managed string is object header, int32 length, then UTF-16 data:
           the data starts at 0x14, not at a pointer-sized 0x18. */
        const int32_t length = *(const int32_t*)(uintptr_t)(str + 0x10);
        const uint16_t* chars = (const uint16_t*)(uintptr_t)(str + 0x14);
        if (length < 0 || length > 4096 || !arc_can_read(str + 0x14, 2u * (size_t)length)) {
          fprintf(stderr, "<unreadable length %lld>", (long long)length);
        } else {
          fprintf(stderr, "\"");
          for (int64_t k = 0; k < length; ++k) {
            const unsigned c = chars[k];
            if (c >= 0x20 && c < 0x7f)
              fputc((int)c, stderr);
            else
              fprintf(stderr, "\\u%04x", c);
          }
          fprintf(stderr, "\" (%lld)", (long long)length);
        }
      }
      fprintf(stderr, "\n");
      errno = saved;
    }
  }
  /* A coroutine's MoveNext is one function with fifteen entry points behind
     one instruction pointer: the state it is in lives in a field of the object
     it was handed, not in the register set. ARC_TRACE_FIELD names that field,
     `<packed function>:<offset>[+<offset>...][:<size>][@<reg>]`,
     comma-separated, and the trace says which state each call resumed -- which
     is the difference between "the coroutine is advancing" and "the coroutine
     keeps starting over". The optional register says which register the object
     is in, because the compiler keeps `this` in a callee-saved register and a
     callee that does not restore it is itself the bug being looked for. The
     `+` chain dereferences one pointer per extra offset, so a wait flag two
     nodes deep (`this+0x30 -> +0x18 -> +0x11`) can be watched without a
     title-specific probe. */
  {
    static uint64_t field_fn[16];
    static uint64_t field_off[16][4];
    static int field_depth[16];
    static int field_size[16];
    static int field_reg[16];
    static int field_count = -1;
    if (field_count < 0) {
      field_count = 0;
      const char* list = getenv("ARC_TRACE_FIELD");
      for (const char* p = list; p && *p && field_count < 16;) {
        char* end;
        const uint64_t fn = strtoull(p, &end, 16);
        if (end == p) {
          ++p;
          continue;
        }
        p = end;
        uint64_t chain[4] = {0, 0, 0, 0};
        int depth = 0;
        int size = 4;
        if (*p == ':') {
          chain[0] = strtoull(p + 1, &end, 16);
          p = (end != p + 1) ? end : p + 1;
          while (*p == '+' && depth < 3) {
            const uint64_t hop = strtoull(p + 1, &end, 16);
            if (end == p + 1) break;
            p = end;
            chain[++depth] = hop;
          }
          if (*p == ':') {
            size = (int)strtol(p + 1, &end, 10);
            p = (end != p + 1) ? end : p + 1;
          }
        }
        int reg = 0;
        if (*p == '@') {
          reg = (int)strtol(p + 1, &end, 10);
          p = (end != p + 1) ? end : p + 1;
        }
        if (reg < 0 || reg > 30) reg = 0;
        field_fn[field_count] = fn;
        memcpy(field_off[field_count], chain, sizeof chain);
        field_depth[field_count] = depth;
        field_size[field_count] = size == 8 ? 8 : size == 1 ? 1 : 4;
        field_reg[field_count] = reg;
        ++field_count;
        while (*p && *p != ',') ++p;
        while (*p == ',') ++p;
      }
    }
    for (int i = 0; i < field_count; ++i) {
      if (field_fn[i] != packed) continue;
      Arm64Ctx* fc = arc_current_guest_context();
      uint64_t base = fc ? fc->x[field_reg[i]] : 0;
      if (!base) {
        fprintf(stderr, "[field] t%u %#llx x%u=0 unreadable\n",
                arc_trace_thread(), (unsigned long long)packed,
                (unsigned)field_reg[i]);
        continue;
      }
      fprintf(stderr, "[field] t%u %#llx x%u=%#llx",
              arc_trace_thread(), (unsigned long long)packed,
              (unsigned)field_reg[i], (unsigned long long)base);
      int readable = 1;
      for (int d = 0; d <= field_depth[i]; ++d) {
        const uint64_t at = base + field_off[i][d];
        if (!base ||
            !arc_can_read(at, d == field_depth[i] ? (size_t)field_size[i] : 8)) {
          fprintf(stderr, " +%#llx=<unreadable>",
                  (unsigned long long)field_off[i][d]);
          readable = 0;
          break;
        }
        uint64_t v;
        if (d < field_depth[i] || field_size[i] == 8)
          v = *(const uint64_t*)(uintptr_t)at;
        else if (field_size[i] == 1)
          v = *(const uint8_t*)(uintptr_t)at;
        else
          v = *(const uint32_t*)(uintptr_t)at;
        fprintf(stderr, " +%#llx=%#llx",
                (unsigned long long)field_off[i][d], (unsigned long long)v);
        base = v;
      }
      fprintf(stderr, " lr=%#llx seen=%llu%s\n",
              (unsigned long long)(fc ? fc->x[30] : 0),
              (unsigned long long)t_frame_seen, readable ? "" : " (short)");
    }
  }
  /* A filtered entry trace names a function but not the arguments that make
     one call different from the next. A title that loops over a buffer asks
     "is this the same element again?", which only the registers answer. */
  {
    static uint64_t arg_targets[16];
    static int arg_count = -1;
    static unsigned long long arg_shown;
    static unsigned long long arg_max = 200;
    if (arg_count < 0) {
      const char* cap = getenv("ARC_TRACE_ARGS_MAX");
      if (cap && *cap) arg_max = strtoull(cap, NULL, 0);
      if (!arg_max || arg_max > 1000000) arg_max = 200;
      const char* list = getenv("ARC_TRACE_ARGS");
      arg_count = 0;
      if (list) {
        const char* p = list;
        while (*p && arg_count < 16) {
          char* end;
          uint64_t v = strtoull(p, &end, 16);
          if (end != p) arg_targets[arg_count++] = v;
          p = (end != p) ? end : p + 1;
          while (*p && *p != ',') ++p;
          while (*p == ',') ++p;
        }
      }
    }
    for (int i = 0; i < arg_count; ++i) {
      if (arg_targets[i] == packed) {
        if (arg_shown < arg_max) {
          Arm64Ctx* c = arc_current_guest_context();
          ++arg_shown;
          if (packed == UINT64_C(0x42F63D0) && c &&
              getenv("ARC_TRACE_DICT")) {
            const uint64_t d = c->x[0];
            uint64_t buckets = 0, entries = 0;
            int32_t count = 0, bucket_len = 0, entry_len = 0;
            if (arc_can_read(d, 0x38)) {
              memcpy(&buckets, (const void*)(uintptr_t)(d + 0x10), 8);
              memcpy(&entries, (const void*)(uintptr_t)(d + 0x18), 8);
              memcpy(&count, (const void*)(uintptr_t)(d + 0x20), 4);
            }
            if (arc_can_read(buckets + 0x18, 4))
              memcpy(&bucket_len, (const void*)(uintptr_t)(buckets + 0x18), 4);
            if (arc_can_read(entries + 0x18, 4))
              memcpy(&entry_len, (const void*)(uintptr_t)(entries + 0x18), 4);
            fprintf(stderr,
                    "[dict] d=%#llx buckets=%#llx len=%d entries=%#llx "
                    "len=%d count=%d hash=%#llx\n",
                    (unsigned long long)d, (unsigned long long)buckets,
                    bucket_len, (unsigned long long)entries, entry_len, count,
                    (unsigned long long)c->x[1]);
          }
          fprintf(stderr,
                  "[args] t%u fn=%llx x0=%#llx x1=%#llx x2=%#llx x3=%#llx "
                  "x4=%#llx x5=%#llx lr=%#llx\n",
                  arc_trace_thread(), (unsigned long long)packed,
                  c ? (unsigned long long)c->x[0] : 0,
                  c ? (unsigned long long)c->x[1] : 0,
                  c ? (unsigned long long)c->x[2] : 0,
                  c ? (unsigned long long)c->x[3] : 0,
                  c ? (unsigned long long)c->x[4] : 0,
                  c ? (unsigned long long)c->x[5] : 0,
                  c ? (unsigned long long)c->x[30] : 0);
        }
        break;
      }
    }
  }
  /* Reflection reaches managed code through a MethodInfo, whose first field is
     the method name. Printing it names the managed method a native caller is
     about to enter without needing a symbol map. */
  if (packed == UINT64_C(0x1e8de5c)) {
    static int method_trace = -1;
    static unsigned method_shown;
    if (method_trace < 0)
      method_trace = getenv("ARC_TRACE_METHODS") != NULL;
    if (method_trace && method_shown < 2000) {
      Arm64Ctx* c = arc_current_guest_context();
      const uint64_t mi = c ? c->x[0] : 0;
      uint64_t q0 = 0, namep = 0, klass = 0;
      const char* name = "?";
      const char* cls = "?";
      const char* ns = "?";
      if (arc_can_read(mi, 8)) q0 = arc_ld64(mi);
      if (arc_can_read(mi + 0x18, 8)) {
        namep = arc_ld64(mi + 0x18);
        if (arc_can_read(namep, 1)) name = (const char*)(uintptr_t)namep;
      }
      if (arc_can_read(mi + 0x20, 8)) {
        klass = arc_ld64(mi + 0x20);
        if (arc_can_read(klass + 0x10, 8)) {
          uint64_t cp = arc_ld64(klass + 0x10);
          if (arc_can_read(cp, 1)) cls = (const char*)(uintptr_t)cp;
        }
        if (arc_can_read(klass + 0x18, 8)) {
          uint64_t np = arc_ld64(klass + 0x18);
          if (arc_can_read(np, 1)) ns = (const char*)(uintptr_t)np;
        }
      }
      ++method_shown;
      fprintf(stderr,
              "[method] t%u mi=%#llx q0=%#llx name=\"%.140s\" class=\"%.80s.%.80s\" lr=%#llx\n",
              arc_trace_thread(), (unsigned long long)mi,
              (unsigned long long)q0, name, ns, cls,
              c ? (unsigned long long)c->x[30] : 0);
    }
  }
  if (packed == UINT64_C(0x1f51cd0)) {
    static int alloc_trace = -1;
    static unsigned long long alloc_n;
    if (alloc_trace < 0) alloc_trace = getenv("ARC_TRACE_ALLOC") != NULL;
    if (alloc_trace && (alloc_n < 40 || (alloc_n % 1000) == 0)) {
      Arm64Ctx* c = arc_current_guest_context();
      fprintf(stderr, "[alloc] n=%llu x0=%#llx x1=%#llx lr=%#llx\n", alloc_n,
              c ? (unsigned long long)c->x[0] : 0,
              c ? (unsigned long long)c->x[1] : 0,
              c ? (unsigned long long)c->x[30] : 0);
    }
    ++alloc_n;
  }
  arc_census_note(packed);
  t_frames[t_frame_next] = packed;
  t_frame_next = (t_frame_next + 1) % ARC_FRAME_RING;
  ++t_frame_seen;
  if (t_is_main_thread) {
    t_main_frames[t_main_frame_next] = packed;
    t_main_frame_next = (t_main_frame_next + 1) % ARC_FRAME_RING;
    ++t_main_frame_seen;
  }
}

/* A guest call inside one image is a direct call to a lifted function, so it
   never passes through arc_dispatch and neither ARC_SKIP_GUEST nor
   ARC_GUEST_RET can answer it.  The entry hook is the only seam such a call
   has: the generated body calls this first and returns at once when it says
   the answer is already known.  Entries are `offset=value` pairs in image-0
   address space, the same shape as ARC_GUEST_RET, and the caller stores the
   answer in x0 itself so the register is written by the body that owns it. */
int arc_entry_return(uint64_t packed, uint64_t* out) {
  static uint64_t offsets[16];
  static uint64_t values[16];
  static int count = -1;
  static unsigned long long hits;
  if (count < 0) {
    count = 0;
    const char* p = getenv("ARC_FN_RET");
    while (p && *p && count < 16) {
      char* end;
      const unsigned long long off = strtoull(p, &end, 0);
      if (end != p && *end == '=') {
        char* vend;
        const unsigned long long val = strtoull(end + 1, &vend, 0);
        if (vend != end + 1) {
          offsets[count] = (uint64_t)off;
          values[count] = (uint64_t)val;
          ++count;
          p = vend;
        } else {
          p = vend;
        }
      } else {
        p = (end != p) ? end : p + 1;
      }
      while (*p && *p != ',') ++p;
      while (*p == ',') ++p;
    }
  }
  if (!count || (packed >> 56) != 0) return 0;
  for (int i = 0; i < count; ++i) {
    if ((packed & UINT64_C(0xffffffffffffff)) != offsets[i]) continue;
    ++hits;
    if (hits <= 8 || hits % 1000 == 0) {
      const int saved = errno;
      fprintf(stderr, "[fn-ret] %#llx -> %#llx (%llu so far)\n",
              (unsigned long long)offsets[i], (unsigned long long)values[i],
              hits);
      errno = saved;
    }
    *out = values[i];
    return 1;
  }
  return 0;
}
#endif

/* How often each guest function runs.
 *
 * The ring says which functions ran most recently; it cannot say which ran
 * ten thousand times. That is the question when a layer draws far more
 * sprites than it should: the loop responsible is whichever function runs
 * once per sprite, and nothing else in this kit counts. ARC_FN_CENSUS turns
 * it on; arc_census_report prints the busiest and starts again.
 *
 * ponytail: a fixed open-addressed table with no growth and no eviction --
 * it counts a few thousand distinct functions and stops learning new ones
 * after that, which is enough to find a hot loop. Make it grow if a run ever
 * needs more than the report says it saw. */
#define ARC_CENSUS_SLOTS 8192
static uint64_t g_census_key[ARC_CENSUS_SLOTS];
static uint64_t g_census_hits[ARC_CENSUS_SLOTS];
static int g_census_on = -1;
static uint64_t g_census_lost;

/* Where to start counting from.
 *
 * A frame number is the wrong handle on work that happens once. The town's
 * sprites are built during loading, and a tally that starts when the process
 * does buries twenty thousand one-time calls under a million steady-state
 * ones. ARC_FN_CENSUS_FROM=<image offset> clears the tally the first time
 * that guest function runs, so the next report covers the build and not the
 * history before it. */
static uint64_t g_census_from;
static int g_census_started;
static int g_census_pending;

/* True once the marker has run and the tally has not been reported yet, so
   the caller can report at the first frame boundary after it -- which is the
   work the marker introduces, and nothing after. */
int arc_census_pending(void) { return g_census_pending; }

static void arc_census_clear(void) {
  size_t i;
  for (i = 0; i < ARC_CENSUS_SLOTS; ++i) {
    g_census_key[i] = 0;
    g_census_hits[i] = 0;
  }
  g_census_lost = 0;
}

void arc_census_note(uint64_t packed) {
  size_t i;
  if (g_census_on < 0) {
    const char* from = getenv("ARC_FN_CENSUS_FROM");
    g_census_on = getenv("ARC_FN_CENSUS") != NULL || from != NULL;
    if (from) g_census_from = strtoull(from, NULL, 16);
  }
  if (!g_census_on) return;
  /* The table is a linear-probe hash with a plain read-modify-write per sample.
     Two guest threads sampling it at once can leave a slot holding a key with
     no hits, or a zero key beside live hits, and the insert loop can then walk
     past the end of its own probe window. The reported stall is work on the
     boot thread anyway, so make that thread the only writer; the watchdog that
     reads and clears the table then meets one producer rather than twenty. */
  if (!t_is_main_thread) return;
  if (g_census_from && !g_census_started && packed == g_census_from) {
    g_census_started = 1;
    g_census_pending = 1;
    arc_census_clear();
    fprintf(stderr, "[fncensus] counting from %#llx\n",
            (unsigned long long)packed);
  }
  i = (size_t)((packed * 0x9E3779B97F4A7C15ull) >> 51) % ARC_CENSUS_SLOTS;
  for (;;) {
    if (g_census_key[i] == packed) { ++g_census_hits[i]; return; }
    if (g_census_key[i] == 0) {
      g_census_key[i] = packed;
      g_census_hits[i] = 1;
      return;
    }
    i = (i + 1) % ARC_CENSUS_SLOTS;
    if (g_census_key[i] == 0 && g_census_key[(i + 1) % ARC_CENSUS_SLOTS] == 0) {
      /* Wrapped far enough to call it full. */
      ++g_census_lost;
      return;
    }
  }
}

void arc_census_report(int top) {
  size_t i, k;
  if (g_census_on <= 0) return;
  g_census_pending = 0;
  fprintf(stderr, "[fncensus] busiest guest functions this frame:\n");
  for (k = 0; k < (size_t)top; ++k) {
    size_t best = ARC_CENSUS_SLOTS;
    uint64_t most = 0;
    for (i = 0; i < ARC_CENSUS_SLOTS; ++i)
      if (g_census_hits[i] > most) { most = g_census_hits[i]; best = i; }
    if (best == ARC_CENSUS_SLOTS) break;
    fprintf(stderr, "[fncensus]   %#llx  %llu\n",
            (unsigned long long)g_census_key[best],
            (unsigned long long)g_census_hits[best]);
    g_census_hits[best] = 0;
  }
  arc_census_clear();
}

size_t arc_frame_count(void) {
  return t_frame_seen < ARC_FRAME_RING ? t_frame_seen : ARC_FRAME_RING;
}

/* The ring saturates, so its count stops telling a busy frame from an idle
   one once it fills. This is how many calls actually happened. */
size_t arc_frame_seen(void) { return t_frame_seen; }

uint64_t arc_frame_at(size_t back) {
  if (back >= arc_frame_count()) return 0;
  return t_frames[(t_frame_next + ARC_FRAME_RING - 1 - back) % ARC_FRAME_RING];
}

void arc_frame_clear(void) {
  t_frame_next = 0;
  t_frame_seen = 0;
}

static ArcDispatchFn g_dispatch;

void arc_set_dispatch(ArcDispatchFn fn) { g_dispatch = fn; }

static ArcExplainFn g_explain;

void arc_set_explain(ArcExplainFn fn) { g_explain = fn; }

/* One guest function, watched: its arguments going in and its result coming
   out, with the memory each pointer argument names.
 *
 * A statically linked library still reaches its own exported functions through
 * the PLT, and a lifted PLT stub is an indirect branch -- so every such call
 * arrives here, with the context in hand. That makes this the one place a
 * specific guest function can be observed without regenerating the program:
 * ARC_WATCH names it by address, and the dump is what a printf inside it would
 * have printed if the source were ours to edit. Answering "is this routine
 * computing the right thing" is otherwise a rebuild of 430 MB of C away.
 */
/* Named by image offset rather than by mapped address: the address moves
   every run, and the offset is what a disassembler and every other trace in
   this kit already speak. */
static uint64_t g_watch_base;

void arc_set_image_base(uint64_t base) { g_watch_base = base; }

static int arc_watched(uint64_t target) {
  static const char* list;
  static int looked;
  const char* p;
  if (!looked) {
    looked = 1;
    list = getenv("ARC_WATCH");
  }
  if (!list || target < g_watch_base) return 0;
  target -= g_watch_base;
  for (p = list; *p;) {
    char* end;
    unsigned long long v = strtoull(p, &end, 16);
    if (end != p && v == target) return 1;
    p = (end != p) ? end : p + 1;
    while (*p && *p != ',') ++p;
    while (*p == ',') ++p;
  }
  return 0;
}

static void arc_watch_bytes(const char* label, uint64_t p, int n) {
  int i;
#if defined(_WIN32)
  MEMORY_BASIC_INFORMATION mbi;
  if (!p || !VirtualQuery((void*)(uintptr_t)p, &mbi, sizeof mbi)) return;
  if (mbi.State != MEM_COMMIT || (mbi.Protect & PAGE_GUARD)) return;
  if (p + (uint64_t)n >
      (uint64_t)(uintptr_t)mbi.BaseAddress + (uint64_t)mbi.RegionSize)
    return;
#else
  if (!p) return;
#endif
  fprintf(stderr, "  %s %#llx:", label, (unsigned long long)p);
  for (i = 0; i < n; ++i)
    fprintf(stderr, " %02x", ((const unsigned char*)(uintptr_t)p)[i]);
  fprintf(stderr, "  |");
  for (i = 0; i < n; ++i) {
    unsigned char b = ((const unsigned char*)(uintptr_t)p)[i];
    fputc(b >= 0x20 && b < 0x7f ? b : '.', stderr);
  }
  fprintf(stderr, "|\n");
}

/* x8 is dumped on the way out as well as in, from the value it had on entry.
   A C++ function returning anything larger than two words returns it through
   a buffer the caller passes in x8, so for a great deal of real code the
   result is not in x0 at all -- and x8 itself is caller-saved, so reading it
   afterwards reads whatever the callee left there. */
static void arc_watch_report(const char* when, const Arm64Ctx* c,
                             uint64_t target, uint64_t sret) {
  int i;
  fprintf(stderr, "[watch] %s %#llx t%u  ", when, (unsigned long long)target,
          arc_trace_thread());
  for (i = 0; i < 6; ++i)
    fprintf(stderr, "x%d=%llx ", i, (unsigned long long)c->x[i]);
  fprintf(stderr, "x8=%llx\n", (unsigned long long)sret);
  for (i = 0; i < 4; ++i) arc_watch_bytes("arg", c->x[i], 64);
  arc_watch_bytes("x8 ", sret, 32);
}

static void arc_watch_il2cpp_string(const char* label, uint64_t s) {
  if (!s) return;
  const uint32_t len = arc_ld32(s + 0x10);
  fprintf(stderr, "  %s len=%u \"", label, len);
  for (uint32_t i = 0; i < len && i < 256; ++i) {
    const uint16_t ch = arc_ld16(s + 0x14 + 2 * i);
    fputc(ch >= 0x20 && ch < 0x7f ? ch : '?', stderr);
  }
  fprintf(stderr, "\"\n");
}

static void arc_shadow_report(const char* tag) {
  const size_t depth = t_shadow_depth;
  fprintf(stderr, "[shadow:%s] depth=%llu\n", tag,
          (unsigned long long)depth);
  for (size_t i = 0; i < depth && i < 64; ++i)
    fprintf(stderr, "  sh[%llu] %#llx\n", (unsigned long long)i,
            (unsigned long long)t_shadow[depth - 1 - i]);
}

#define ARC_SIGNAL_MAX 64
#define ARC_SIGNAL_SLOTS 4096
static uint64_t g_signal_handlers[ARC_SIGNAL_MAX];

void arc_signal_set_handler(int sig, uint64_t handler) {
  if (sig > 0 && sig < ARC_SIGNAL_MAX) g_signal_handlers[sig] = handler;
}

uint64_t arc_signal_handler(int sig) {
  return (sig > 0 && sig < ARC_SIGNAL_MAX) ? g_signal_handlers[sig] : 0;
}

#if defined(_WIN32)
typedef struct ArcSignalSlot {
  DWORD host;
  volatile LONG mask;
} ArcSignalSlot;
static ArcSignalSlot g_signal_slots[ARC_SIGNAL_SLOTS];
static SRWLOCK g_signal_lock = SRWLOCK_INIT;
static volatile LONG g_signal_any;

static ArcSignalSlot* ArcSignalFind(DWORD host, int create) {
  ArcSignalSlot* free_slot = NULL;
  for (size_t i = 0; i < ARC_SIGNAL_SLOTS; ++i) {
    if (g_signal_slots[i].host == host) return &g_signal_slots[i];
    if (!free_slot && g_signal_slots[i].host == 0) free_slot = &g_signal_slots[i];
  }
  if (create && free_slot) {
    free_slot->host = host;
    return free_slot;
  }
  return NULL;
}

void arc_signal_post_handle(void* thread_handle, int sig) {
  if (sig <= 0 || sig >= 32) return;
  const DWORD host = thread_handle ? GetThreadId((HANDLE)thread_handle)
                                   : GetCurrentThreadId();
  arc_signal_post_host((unsigned int)host, sig);
}

void arc_signal_post_host(unsigned int host_id, int sig) {
  if (!host_id || sig <= 0 || sig >= 32) return;
  const DWORD host = (DWORD)host_id;
  AcquireSRWLockExclusive(&g_signal_lock);
  ArcSignalSlot* slot = ArcSignalFind(host, 1);
  if (slot) {
    slot->mask |= (1L << sig);
    g_signal_any = 1;
  }
  ReleaseSRWLockExclusive(&g_signal_lock);
}

void arc_signal_post_current(int sig) { arc_signal_post_handle(NULL, sig); }

int arc_signal_take_current(void) {
  if (!g_signal_any) return 0;
  const DWORD host = GetCurrentThreadId();
  int found = 0;
  AcquireSRWLockExclusive(&g_signal_lock);
  ArcSignalSlot* slot = ArcSignalFind(host, 0);
  if (slot) {
    for (int sig = 1; sig < 32; ++sig) {
      if (slot->mask & (1L << sig)) {
        slot->mask &= ~(1L << sig);
        found = sig;
        break;
      }
    }
  }
  if (found) {
    LONG any = 0;
    for (size_t i = 0; i < ARC_SIGNAL_SLOTS; ++i) {
      if (g_signal_slots[i].mask) {
        any = 1;
        break;
      }
    }
    g_signal_any = any;
  }
  ReleaseSRWLockExclusive(&g_signal_lock);
  return found;
}

int arc_signal_pending_current(void) {
  if (!g_signal_any) return 0;
  const DWORD host = GetCurrentThreadId();
  int found = 0;
  AcquireSRWLockShared(&g_signal_lock);
  ArcSignalSlot* slot = ArcSignalFind(host, 0);
  if (slot && slot->mask) found = 1;
  ReleaseSRWLockShared(&g_signal_lock);
  return found;
}
#else
void arc_signal_post_handle(void*, int) {}
void arc_signal_post_host(unsigned int, int) {}
void arc_signal_post_current(int) {}
int arc_signal_take_current(void) { return 0; }
int arc_signal_pending_current(void) { return 0; }
#endif

static ARC_THREAD_LOCAL int t_in_signal_handler;
static ARC_THREAD_LOCAL Arm64Ctx* t_current_guest_context;

#if defined(_WIN32)
static unsigned long ArcHostId(void) { return (unsigned long)GetCurrentThreadId(); }
#else
static unsigned long ArcHostId(void) { return 0; }
#endif

void arc_run_pending_signals(void) {
  if (t_in_signal_handler) return;
  for (;;) {
    const int sig = arc_signal_take_current();
    if (!sig) break;
    const uint64_t handler = arc_signal_handler(sig);
    if (!handler || handler == 1) continue;
    uint64_t args[1] = {(uint64_t)sig};
    fprintf(stderr, "[signal] delivering sig=%d handler=%#llx host=%lu\n", sig,
            (unsigned long long)handler, ArcHostId());
    fflush(stderr);
    t_in_signal_handler = 1;
    arc_call_guest(handler, args, 1);
    t_in_signal_handler = 0;
    fprintf(stderr, "[signal] returned sig=%d host=%lu\n", sig, ArcHostId());
    fflush(stderr);
  }
}

// --- port-registered answers -----------------------------------------------
// See arc_context.h: an entry names one guest entry point and the value it
// should return. Kept in the runtime rather than in the environment because the
// reason for an answer is a property of the port -- what the port pointed the
// guest at -- and it should travel with the code that made that choice.
#define ARC_ANSWERS 32
// The images are the generated program's; this is only a bound for the table
// above, which is filled from the host and cannot know the count without
// including the generated header.
#define ARC_ANSWER_IMAGES 8
typedef struct {
  int image;
  uint64_t offset;
  uint64_t value;
  const char* what;
  unsigned long long hits;
} ArcAnswer;

static ArcAnswer g_answers[ARC_ANSWERS];
static int g_answer_count;

void arc_guest_answer(int image, uint64_t offset, uint64_t value,
                      const char* what) {
  if (image < 0 || image >= ARC_ANSWER_IMAGES ||
      g_answer_count >= ARC_ANSWERS)
    return;
  ArcAnswer* a = &g_answers[g_answer_count++];
  a->image = image;
  a->offset = offset;
  a->value = value;
  a->what = what;
  a->hits = 0;
}

#define ARC_MON_RING 2097152

/* Monitor history.  An Exit that finds the lock held by "somebody else" is
   either a second thread or an object carrying a block that was already
   released or handed to someone else -- and only the record of what happened
   to that object can tell them apart.  Kinds: 0 = Enter called, 1 = block
   installed into obj->monitor, 2 = obj->monitor cleared on release. */
typedef struct {
  uint64_t seq;
  uint64_t kind;
  uint64_t obj;
  uint64_t data;
  uint64_t self;
  uint64_t host;
} ArcMonRec;

extern uint32_t arc_guest_self_id(void);

static ArcMonRec g_mon_ring[ARC_MON_RING];
static uint64_t g_mon_seq;

static int mon_trace_on(void) {
  static int on = -1;
  if (on < 0) on = getenv("ARC_TRACE_MONITOR") ? 1 : 0;
  return on;
}

static uint64_t mon_next_seq(void) {
#if defined(_MSC_VER)
  return (uint64_t)_InterlockedIncrement64((volatile __int64*)&g_mon_seq);
#else
  return (uint64_t)__sync_fetch_and_add(&g_mon_seq, 1) + 1;
#endif
}

static uint64_t mon_read_seq(void) {
#if defined(_MSC_VER)
  return (uint64_t)_InterlockedCompareExchange64((volatile __int64*)&g_mon_seq,
                                                 0, 0);
#else
  return (uint64_t)__sync_fetch_and_add(&g_mon_seq, 0);
#endif
}

void arc_mon_event(int kind, uint64_t obj, uint64_t data) {
  if (!mon_trace_on()) return;
  const uint64_t seq = mon_next_seq();
  ArcMonRec* r = &g_mon_ring[seq % ARC_MON_RING];
  r->kind = (uint64_t)kind;
  r->obj = obj;
  r->data = data;
  r->self = (uint64_t)arc_guest_self_id();
#if defined(_WIN32)
  r->host = (uint64_t)GetCurrentThreadId();
#else
  r->host = (uint64_t)pthread_self();
#endif
#if defined(_MSC_VER)
  _InterlockedExchange64((volatile __int64*)&r->seq, (__int64)seq);
#else
  __sync_lock_test_and_set(&r->seq, seq);
#endif
}

/* Kept as its own entry point so the call site in the lifted Enter body reads
   as what it is. */
void arc_mon_enter(uint64_t obj) { arc_mon_event(0, obj, 0); }

/* Last recorded event that names `key` in `field` (0 = object, 1 = block). */
static int mon_last(int field, uint64_t key, ArcMonRec* out, uint64_t* ago) {
  const uint64_t now = mon_read_seq();
  for (uint64_t back = 1; back <= ARC_MON_RING && back <= now; ++back) {
    ArcMonRec* r = &g_mon_ring[(now - back) % ARC_MON_RING];
    const uint64_t s = r->seq;
    if (s != now - back) continue;
    const uint64_t v = field ? r->data : r->obj;
    if (v == key) {
      *out = *r;
      *ago = now - s;
      return 1;
    }
  }
  return 0;
}

static const char* mon_kind_name(uint64_t kind) {
  switch (kind) {
    case 0: return "enter";
    case 1: return "install";
    case 2: return "clear";
    default: return "?";
  }
}

static void mon_dump(int n) {
  const uint64_t now = mon_read_seq();
  fprintf(stderr, "[mon] last %d events:\n", n);
  for (int i = n; i >= 1; --i) {
    const uint64_t back = (uint64_t)i;
    if (back > now) break;
    ArcMonRec* r = &g_mon_ring[(now - back) % ARC_MON_RING];
    if (r->seq != now - back) continue;
    fprintf(stderr, "  seq=%llu %-7s obj=%#llx data=%#llx self=%u host=%llu\n",
            (unsigned long long)r->seq, mon_kind_name(r->kind),
            (unsigned long long)r->obj, (unsigned long long)r->data,
            (unsigned)r->self, (unsigned long long)r->host);
  }
}

/* A monitor was found held by someone other than the exiting thread.  The ids
   are the guest's own pthread_self values, so printing both answers the only
   question that matters: a second thread, or the same thread under a changed
   id?  Gated because it fires on a fault path only. */
static ARC_THREAD_LOCAL uint64_t t_mon_obj;
static ARC_THREAD_LOCAL uint64_t t_mon_lr;
void arc_mon_set(uint64_t obj, uint64_t lr) {
  t_mon_obj = obj;
  t_mon_lr = lr;
}

void arc_mon_note(uint64_t data, uint64_t owner, uint64_t self,
                  uint64_t flags, uint64_t nest) {
  if (!mon_trace_on()) return;
#if defined(_WIN32)
  const unsigned long host = GetCurrentThreadId();
#else
  const unsigned long host = (unsigned long)pthread_self();
#endif
  const uint64_t obj = t_mon_obj;
  const uint64_t lr = t_mon_lr;
  const uint64_t klass = obj ? arc_ld64(obj) : 0;
  const uint64_t kname = klass ? arc_ld64(klass + 0x10) : 0;
  const uint64_t kns = klass ? arc_ld64(klass + 0x18) : 0;
  fprintf(stderr,
          "[mon] exit-unowned obj=%#llx klass=%#llx type=%s.%s data=%#llx "
          "owner=%u self=%u flags=%u nest=%u hosttid=%lu lr=%#llx "
          "base0=%#llx ops=%llu\n",
          (unsigned long long)obj, (unsigned long long)klass,
          kname ? (const char*)(uintptr_t)kname : "?",
          kns ? (const char*)(uintptr_t)kns : "?",
          (unsigned long long)data, (unsigned)owner, (unsigned)self,
          (unsigned)flags, (unsigned)nest, host, (unsigned long long)lr,
          (unsigned long long)arc_image_bases[0],
          (unsigned long long)g_mon_seq);
  ArcMonRec ev;
  uint64_t ago = 0;
  if (mon_last(0, obj, &ev, &ago))
    fprintf(stderr,
            "[mon]   obj last: %s data=%#llx self=%u host=%llu ago=%llu\n",
            mon_kind_name(ev.kind), (unsigned long long)ev.data, (unsigned)ev.self,
            (unsigned long long)ev.host, (unsigned long long)ago);
  else
    fprintf(stderr, "[mon]   obj: never seen\n");
  if (mon_last(1, data, &ev, &ago))
    fprintf(stderr,
            "[mon]   block last: %s obj=%#llx self=%u host=%llu ago=%llu\n",
            mon_kind_name(ev.kind), (unsigned long long)ev.obj, (unsigned)ev.self,
            (unsigned long long)ev.host, (unsigned long long)ago);
  else
    fprintf(stderr, "[mon]   block: never seen\n");
  /* Which object, if any, currently owns that block?  The install record is
     the answer and it is one lookup away. */
  if (mon_last(1, obj, &ev, &ago))
    fprintf(stderr, "[mon]   obj install last: %s data=%#llx\n",
            mon_kind_name(ev.kind), (unsigned long long)ev.data);
  mon_dump(10);
  fflush(stderr);
  arc_shadow_report("mon");
}

int arc_lifted_return(void* fn, const char* what) {
#if defined(_M_X64) && defined(_WIN32)
  DWORD old = 0;
  if (!fn) return 0;
  if (!VirtualProtect(fn, 1, PAGE_EXECUTE_READWRITE, &old)) return 0;
  /* One byte is the whole edit: a call pushes the return address, and the body
     that would have set up its frame never runs, so `ret` here unwinds exactly
     the frame the caller made. */
  *(volatile unsigned char*)fn = 0xC3;
  FlushInstructionCache(GetCurrentProcess(), fn, 1);
  VirtualProtect(fn, 1, old, &old);
  fprintf(stderr, "[lifted] %s retired (%p)\n", what ? what : "function", fn);
  return 1;
#else
  (void)fn;
  fprintf(stderr, "[lifted] %s cannot be retired on this host\n",
          what ? what : "function");
  return 0;
#endif
}

// A watched guest qword, for finding the writer of a slot that should not
// change. A lifted store is an ordinary host store, so there is nothing to
// hook; what can be seen is the branch that follows it, and the frame ring at
// that moment names the function that just ran. ARC_WATCH_QWORD=<image>:<off>.
static uint64_t g_watch_addr;
static uint64_t g_watch_value;
static int g_watch_state;  /* 0 not looked, 1 watching, -1 off */

static int ArcWatchCheck(void) {
  if (g_watch_state == 0) {
    const char* spec = getenv("ARC_WATCH_QWORD");
    unsigned image = 0;
    unsigned long long offset = 0;
    g_watch_state = -1;
    if (spec && sscanf(spec, "%u:%lli", &image, &offset) == 2 &&
        image < 8 && arc_image_bases[image]) {
      g_watch_addr = arc_image_bases[image] + (uint64_t)offset;
      g_watch_value = *(volatile uint64_t*)(uintptr_t)g_watch_addr;
      g_watch_state = 1;
      fprintf(stderr, "[watch] %#llx begins at %#llx\n",
              (unsigned long long)g_watch_addr,
              (unsigned long long)g_watch_value);
    }
  }
  if (g_watch_state != 1) return 0;
  const uint64_t now = *(volatile uint64_t*)(uintptr_t)g_watch_addr;
  if (now == g_watch_value) return 0;
  fprintf(stderr, "[watch] %#llx %#llx -> %#llx\n",
          (unsigned long long)g_watch_addr,
          (unsigned long long)g_watch_value, (unsigned long long)now);
  g_watch_value = now;
  return 1;
}

// The site of the next indirect branch, set by arc_dispatch_pc and consumed by
// arc_dispatch. It is a hand-off rather than a field on the context because a
// branch site describes one dispatch: left on the context it would be inherited
// by every later dispatch that was not lifted with branch notes, and a trap in
// one of those would report where some earlier branch was taken from.
static ARC_THREAD_LOCAL uint64_t t_dispatch_pc;

/* --- data watch -----------------------------------------------------------
   The counterpart of ARC_WATCH_QWORD for a slot whose address is only known at
   runtime: arm it from a probe and every guest store into the range reports the
   value together with the function that was running. GUESTMEM stores are host
   stores, so the store helpers are the only place that can see them. */
#define ARC_DATA_WATCH_MAX 8
static uint64_t g_data_watch_lo[ARC_DATA_WATCH_MAX];
static uint64_t g_data_watch_hi[ARC_DATA_WATCH_MAX];
unsigned arc_data_watch_count;
static int g_data_watch_quiet;

void arc_watch_data(uint64_t address, uint64_t length) {
  // The lifted probe that arms these is still in one generated shard, but the
  // watch itself is a diagnostic. Disarmed, every guest store pays only the
  // count check in arc_st*; leaving it armed costs a range scan on every store
  // and was visible in the profile long after the bug it watched for was gone.
  // The environment is read once: this is reached from a lifted body, and
  // getenv per call dominated the profile while the watch was disarmed.
  static int enabled = -1;
  if (enabled < 0) {
    const char* value = getenv("ARC_DATA_WATCH");
    enabled = value && *value ? 1 : 0;
  }
  if (!enabled) return;
  if (address == 0 || length == 0) return;
  for (unsigned i = 0; i < arc_data_watch_count; ++i)
    if (address == g_data_watch_lo[i]) return;
  if (arc_data_watch_count >= ARC_DATA_WATCH_MAX) return;
  g_data_watch_lo[arc_data_watch_count] = address;
  g_data_watch_hi[arc_data_watch_count] = address + length;
  ++arc_data_watch_count;
  fprintf(stderr, "[watch-data] arm %#llx..%#llx\n",
          (unsigned long long)address, (unsigned long long)(address + length));
  fflush(stderr);
}

void arc_data_watch_hit(uint64_t address, uint64_t value, unsigned width) {
  for (unsigned i = 0; i < arc_data_watch_count; ++i) {
    if (address >= g_data_watch_hi[i] ||
        address + width <= g_data_watch_lo[i])
      continue;
    const size_t frames = arc_frame_count();
    const uint64_t frame = frames ? arc_frame_at(0) : 0;
    if (g_data_watch_quiet < 200)
      fprintf(stderr,
              "[watch-data] t%u w%u %#llx=%#llx fn=%u:%#llx caller=%llx\n",
              arc_trace_thread(), width, (unsigned long long)address,
              (unsigned long long)value, (unsigned)(frame >> 56),
              (unsigned long long)(frame & 0x00FFFFFFFFFFFFFFull),
              (unsigned long long)(frames > 1 ? (arc_frame_at(1) &
                                                0x00FFFFFFFFFFFFFFull)
                                              : 0));
    ++g_data_watch_quiet;
    return;
  }
}

/* A title that reads a stream a few bytes at a time turns every read into a
   full dispatch: the PLT stub enters arc_dispatch, which searches the images
   and then the native table before calling the host memcpy. Measured on the
   loader that is roughly half of the main thread. A tiny memcpy is just a
   copy, so name its resolved address once and do it here. The list of things
   the normal path does that this skips -- signals, watches and the call trail
   -- are all diagnostics, so the shortcut turns itself off when one is on. */
static uint64_t g_small_copy_target;
static int g_small_copy_enabled = -1;
static int g_small_copy_log = -1;
static unsigned long long g_small_copy_hits;

static int ArcSmallCopyEnabled(void) {
  if (g_small_copy_enabled < 0) {
    g_small_copy_enabled = !getenv("ARC_WATCH") && !getenv("ARC_TRACE_CALLS") &&
                          !getenv("ARC_TRACE_MISS") &&
                          !getenv("ARC_MEMCPY_TRACE");
    g_small_copy_log = getenv("ARC_MEMCPY_FAST_LOG") != NULL;
  }
  return g_small_copy_enabled;
}

static void ArcSmallCopy(uint64_t dst, uint64_t src, uint64_t n) {
  if (!n || dst == src) return;
  if (dst < src || dst >= src + n) {
    while (n >= 8) {
      arc_st64(dst, arc_ld64(src));
      dst += 8;
      src += 8;
      n -= 8;
    }
    if (n >= 4) {
      arc_st32(dst, arc_ld32(src));
      dst += 4;
      src += 4;
      n -= 4;
    }
    if (n >= 2) {
      arc_st16(dst, arc_ld16(src));
      dst += 2;
      src += 2;
      n -= 2;
    }
    if (n) arc_st8(dst, arc_ld8(src));
    return;
  }
  /* memcpy and memmove share one host address on this CRT, so the copy has to
     keep memmove's overlap promise even when the guest asked for memcpy. */
  for (uint64_t i = n; i-- > 0;) arc_st8(dst + i, arc_ld8(src + i));
}

static int ArcSmallCopyName(const char* name) {
  return name && (strcmp(name, "memcpy") == 0 || strcmp(name, "memmove") == 0);
}

/* The optional dispatch diagnostics -- the qword watch, the throw and managed
   traces, ARC_WATCH and ARC_GUEST_RET -- are resolved once here. When none is
   armed the dispatch path skips their loads and calls entirely, which is most
   of what is left of it. ARC_SKIP_GUEST stays outside because the port always
   uses it. */
static int g_dispatch_diag = -1;

static int ArcDispatchDiag(void) {
  if (g_dispatch_diag < 0)
    g_dispatch_diag = getenv("ARC_WATCH") != NULL ||
                      getenv("ARC_WATCH_QWORD") != NULL ||
                      getenv("ARC_TRACE_THROW") != NULL ||
                      getenv("ARC_TRACE_MANAGED") != NULL ||
                      getenv("ARC_GUEST_RET") != NULL;
  return g_dispatch_diag;
}

/* Metadata names are reached through pointers this code does not own: a trace
   that prints one as a C string can fault inside the C runtime, where the fault
   reads as a host bug instead of a bad guest pointer. Copying what is readable
   keeps the trace answering the question it was added for. */
static const char* ArcSafeText(uint64_t address) {
  static char texts[4][96];
  static unsigned next;
  char* text = texts[next++ & 3];
  size_t i = 0;
  for (; address && i + 1 < sizeof texts[0]; ++i) {
    if (!arc_can_read(address + i, 1)) break;
    const char c = *(const char*)(uintptr_t)(address + i);
    if (!c) break;
    text[i] = (c >= 0x20 && c < 0x7f) ? c : '?';
  }
  if (!i && !address) return "?";
  text[i] = 0;
  return text;
}

void arc_dispatch(Arm64Ctx* c, uint64_t target) {
  Arm64Ctx* const previous_guest = t_current_guest_context;
  c->pc = t_dispatch_pc;
  t_dispatch_pc = 0;
  if (target && target == g_small_copy_target && !arc_data_watch_count &&
      c->x[2] <= 4096 && ArcSmallCopyEnabled()) {
    ++g_small_copy_hits;
    if (g_small_copy_log && (g_small_copy_hits % 1000000) == 0) {
      fprintf(stderr, "[fastcopy] hits=%llu n=%llu\n", g_small_copy_hits,
              (unsigned long long)c->x[2]);
      fflush(stderr);
    }
    ArcSmallCopy(c->x[0], c->x[1], c->x[2]);
    return;
  }
  t_current_guest_context = c;
  /* ARC_TRACE_NEXT names the method pointer that a resolved generic delegate
     lands on. IL2CPP hands these out from the metadata at run time, so the
     target is not visible in the lifted tables; printing the callee's image
     and offset is what says whether the dispatcher reached a body at all. */
  {
    static int trace_next = -1;
    static unsigned next_shown;
    if (trace_next < 0) trace_next = getenv("ARC_TRACE_NEXT") != NULL;
    if (trace_next && next_shown < 12 && target) {
      /* The delegate invoker fn0_36245c0 publishes this return address. */
      static uint64_t next_lr;
      if (!next_lr && arc_image_bases[0]) next_lr = arc_image_bases[0] + UINT64_C(56772140);
      if (c->x[30] == next_lr) {
        ++next_shown;
        unsigned long long image = ~0ull, offset = 0;
        if (arc_image_bases[0] && target >= arc_image_bases[0] &&
            target < arc_image_bases[0] + UINT64_C(0x7800000)) {
          image = 0;
          offset = (unsigned long long)(target - arc_image_bases[0]);
        } else if (arc_image_bases[1] && target >= arc_image_bases[1] &&
                   target < arc_image_bases[1] + UINT64_C(0x1a00000)) {
          image = 1;
          offset = (unsigned long long)(target - arc_image_bases[1]);
        }
        const int saved = errno;
        fprintf(stderr, "[next] target=%#llx image=%llu +%#llx node=%#llx\n",
                (unsigned long long)target, image, offset,
                (unsigned long long)c->x[0]);
        fflush(stderr);
        errno = saved;
      }
    }
  }
  const int diag = ArcDispatchDiag();
  // Both of these are calls per dispatch. The signal queue is empty in every
  // ordinary run, and the qword watch is off unless it was asked for, so test
  // their flags here and keep the call out of the hot path.
  if (target != 0 && g_signal_any) arc_run_pending_signals();
  if (diag && target != 0 && g_watch_state != -1 && ArcWatchCheck()) {
    const size_t frames = arc_frame_count();
    fprintf(stderr, "[watch] frames:");
    for (size_t i = 0; i < frames && i < 16; ++i) {
      const uint64_t frame = arc_frame_at(i);
      fprintf(stderr, " %u:%llx", (unsigned)(frame >> 56),
              (unsigned long long)(frame & 0x00FFFFFFFFFFFFFFull));
    }
    fprintf(stderr, " target=%#llx\n", (unsigned long long)target);
  }
  if (target == 0) {
    fprintf(stderr, "[dispatch] target 0 x16=%#llx x17=%#llx lr=%#llx\n",
            (unsigned long long)c->x[16], (unsigned long long)c->x[17],
            (unsigned long long)c->x[30]);
  }
  static const char* trace_throw;
  static int trace_throw_looked;
  if (!trace_throw_looked) {
    trace_throw_looked = 1;
    trace_throw = getenv("ARC_TRACE_THROW");
  }
  if (diag && trace_throw && arc_image_bases[0] &&
      target == arc_image_bases[0] + UINT64_C(0x330ec70)) {
    fprintf(stderr,
            "[throw] lr=%#llx sp=%#llx fp=%#llx x0=%#llx x1=%#llx x2=%#llx\n",
            (unsigned long long)c->x[30], (unsigned long long)c->sp,
            (unsigned long long)c->x[29], (unsigned long long)c->x[0],
            (unsigned long long)c->x[1], (unsigned long long)c->x[2]);
    arc_shadow_report("throw");
    /* The dispatcher's shadow records function entries, not the direct calls
       that lead to the throw. The guest keeps a real frame chain, so walk it:
       its return addresses name the managed callers even when the throwing
       method is only a few instructions into a lifted function. */
    {
      uint64_t fp = c->x[29];
      fprintf(stderr, "[throw-fp] fp=%#llx\n", (unsigned long long)fp);
      for (int depth = 0; depth < 48; ++depth) {
        if (!fp || (fp & 7) || !arc_can_read(fp, 16)) break;
        const uint64_t prev = arc_ld64(fp);
        const uint64_t ret = arc_ld64(fp + 8);
        if (ret >= arc_image_bases[0] &&
            ret < arc_image_bases[0] + UINT64_C(0x7800000))
          fprintf(stderr, "  tfp[%d] libil2cpp+%#llx fp=%#llx\n", depth,
                  (unsigned long long)(ret - arc_image_bases[0]),
                  (unsigned long long)fp);
        else
          fprintf(stderr, "  tfp[%d] %#llx fp=%#llx\n", depth,
                  (unsigned long long)ret, (unsigned long long)fp);
        if (prev <= fp) break;
        fp = prev;
      }
    }
    /* A throw helper is usually a leaf, so the frame chain stops there. The
       return address into its caller is still on the guest stack; scanning the
       first page for code pointers recovers the managed call chain. */
    {
      const uint64_t sp = c->sp;
      fprintf(stderr, "[throw-stack] sp=%#llx\n", (unsigned long long)sp);
      for (uint64_t p = sp; p < sp + 0x1200; p += 8) {
        if (!arc_can_read(p, 8)) break;
        const uint64_t v = arc_ld64(p);
        if (v >= arc_image_bases[0] &&
            v < arc_image_bases[0] + UINT64_C(0x7800000))
          fprintf(stderr, "  tst[%#llx] libil2cpp+%#llx\n",
                  (unsigned long long)(p - sp),
                  (unsigned long long)(v - arc_image_bases[0]));
      }
    }
    const uint64_t wrapper = c->x[0];
    const uint64_t payload = wrapper ? arc_ld64(wrapper) : 0;
    arc_watch_bytes("throwobj", wrapper, 0x80);
    arc_watch_bytes("payload", payload, 0x100);
    arc_watch_bytes("typeinfo", c->x[1], 0x40);
    if (payload) {
      const uint64_t klass = arc_ld64(payload);
      const uint64_t message = arc_ld64(payload + 0x18);
      const uint64_t class_name = arc_ld64(payload + 0x28);
      const uint64_t stack_trace = arc_ld64(payload + 0x30);
      if (klass) {
        const uint64_t name = arc_ld64(klass + 0x10);
        const uint64_t ns = arc_ld64(klass + 0x18);
        fprintf(stderr, "[throw] klass=%#llx name=%s ns=%s\n",
                (unsigned long long)klass, ArcSafeText(name),
                ArcSafeText(ns));
      }
      arc_watch_il2cpp_string("message", message);
      arc_watch_il2cpp_string("classname", class_name);
      arc_watch_il2cpp_string("stack", stack_trace);
      arc_watch_bytes("message", message, 0x40);
      arc_watch_bytes("classname", class_name, 0x40);
      arc_watch_bytes("trace", stack_trace, 0x80);
    }
  }
  static const char* trace_managed;
  static int trace_managed_looked;
  if (!trace_managed_looked) {
    trace_managed_looked = 1;
    trace_managed = getenv("ARC_TRACE_MANAGED");
  }
  if (diag && trace_managed && arc_image_bases[0]) {
    static int managed_shown;
    const uint64_t base = arc_image_bases[0];
    if ((target == base + UINT64_C(0x3fc59a8) ||
         target == base + UINT64_C(0x1e92730)) &&
        managed_shown < 200) {
      ++managed_shown;
      const uint64_t ex = c->x[0];
      fprintf(stderr, "[managed] raise ex=%#llx lr=%#llx\n",
              (unsigned long long)ex, (unsigned long long)c->x[30]);
      if (ex) {
        const uint64_t klass = arc_ld64(ex);
        if (klass) {
          const uint64_t name = arc_ld64(klass + 0x10);
          const uint64_t ns = arc_ld64(klass + 0x18);
          fprintf(stderr, "[managed]   class=%#llx name=%s ns=%s\n",
                  (unsigned long long)klass,
                  name ? (const char*)(uintptr_t)name : "?",
                  ns ? (const char*)(uintptr_t)ns : "?");
        }
      }
    } else if (target == base + UINT64_C(0x1d2839c) ||
               target == base + UINT64_C(0x1e8de5c)) {
      static int loop_shown;
      const uint64_t method = c->x[0];
      if (method && arc_ld64(method) == base + UINT64_C(0x4429f80) &&
          loop_shown < 20) {
        ++loop_shown;
        const uint64_t name = arc_ld64(method + 0x10);
        const uint64_t klass = arc_ld64(method + 0x18);
        const uint64_t kname = klass ? arc_ld64(klass + 0x10) : 0;
        const uint64_t kns = klass ? arc_ld64(klass + 0x18) : 0;
        fprintf(stderr,
                "[managed] loop method=%#llx fn=%#llx name=%s class=%s.%s\n",
                (unsigned long long)method,
                (unsigned long long)arc_ld64(method), ArcSafeText(name),
                ArcSafeText(kns), ArcSafeText(kname));
      }
    } else if (target == base + UINT64_C(0x1ece0b8) && managed_shown < 400) {
      ++managed_shown;
      const uint64_t klass = c->x[0];
      fprintf(stderr, "[managed] class-init %#llx lr=%#llx\n",
              (unsigned long long)klass, (unsigned long long)c->x[30]);
      if (klass) {
        const uint64_t name = arc_ld64(klass + 0x10);
        const uint64_t ns = arc_ld64(klass + 0x18);
        fprintf(stderr, "[managed]   name=%s ns=%s\n", ArcSafeText(name),
                ArcSafeText(ns));
      }
    }
  }
  const int watch = diag && arc_watched(target);
  const uint64_t sret = c->x[8];
  if (watch) arc_watch_report("call", c, target, sret);
  /* Port-registered answers come first: a port that names an entry point in
     code has a reason it can explain next to the call, while the environment
     lists below are for measuring a guess. */
  if (g_answer_count) {
    for (int i = 0; i < g_answer_count; ++i) {
      ArcAnswer* a = &g_answers[i];
      const uint64_t image_base = arc_image_bases[a->image];
      if (!image_base || target != image_base + a->offset) continue;
      ++a->hits;
      if (a->hits <= 3 || a->hits % 500 == 0)
        fprintf(stderr, "[answer] %s image=%d +%#llx -> %#llx (%llu so far)\n",
                a->what ? a->what : "guest", a->image,
                (unsigned long long)a->offset, (unsigned long long)a->value,
                a->hits);
      c->x[0] = a->value;
      t_current_guest_context = previous_guest;
      return;
    }
  }
  /* A port can turn off a guest entry point that depends on a subsystem the
     host does not provide without teaching the lifter about it. The list is
     image-0 offsets, so it stays readable next to the disassembly it came
     from. */
  {
    static int skip_parsed;
    static uint64_t skips[16];
    static int skip_count;
    static unsigned long long skip_hits;
    if (!skip_parsed) {
      skip_parsed = 1;
      const char* p = getenv("ARC_SKIP_GUEST");
      while (p && *p && skip_count < 16) {
        char* end;
        const unsigned long long v = strtoull(p, &end, 0);
        if (end != p) skips[skip_count++] = (uint64_t)v;
        p = (end != p) ? end : p + 1;
        while (*p && *p != ',') ++p;
        while (*p == ',') ++p;
      }
    }
    if (skip_count && arc_image_bases[0]) {
      for (int i = 0; i < skip_count; ++i) {
        if (target == arc_image_bases[0] + skips[i]) {
          ++skip_hits;
          if (skip_hits <= 3 || skip_hits % 1000 == 0)
            fprintf(stderr, "[skip] guest %#llx (%llu so far)\n",
                    (unsigned long long)target, skip_hits);
          c->x[0] = 0;
          t_current_guest_context = previous_guest;
          return;
        }
      }
    }
  }
  /* A guest predicate that the port cannot answer yet can be given an exact
     host-side answer without editing generated code. Entries are
     offset=value, again in image-0 address space. This is deliberately
     separate from ARC_SKIP_GUEST: skipping returns zero, while an oracle can
     need one, a handle, or a length. */
  {
    static int ret_parsed;
    static uint64_t ret_offsets[16];
    static uint64_t ret_values[16];
    static int ret_count;
    static unsigned long long ret_hits;
    if (!ret_parsed) {
      ret_parsed = 1;
      const char* p = getenv("ARC_GUEST_RET");
      while (p && *p && ret_count < 16) {
        char* end;
        const unsigned long long off = strtoull(p, &end, 0);
        if (end != p) {
          if (*end == '=') {
            char* vend;
            const unsigned long long val = strtoull(end + 1, &vend, 0);
            if (vend != end + 1) {
              ret_offsets[ret_count] = (uint64_t)off;
              ret_values[ret_count] = (uint64_t)val;
              ++ret_count;
              p = vend;
            } else {
              p = vend;
            }
          } else {
            p = end;
          }
        } else {
          ++p;
        }
        while (*p && *p != ',') ++p;
        while (*p == ',') ++p;
      }
    }
    if (diag && ret_count && arc_image_bases[0]) {
      for (int i = 0; i < ret_count; ++i) {
        if (target == arc_image_bases[0] + ret_offsets[i]) {
          ++ret_hits;
          if (ret_hits <= 8 || ret_hits % 1000 == 0)
            fprintf(stderr, "[guest-ret] %#llx -> %#llx (%llu so far)\n",
                    (unsigned long long)target,
                    (unsigned long long)ret_values[i], ret_hits);
          c->x[0] = ret_values[i];
          t_current_guest_context = previous_guest;
          return;
        }
      }
    }
  }
  // Only guest entries belong on the shadow stack. The lifted dispatcher owns
  // that decision: it is the only place that knows a target resolved to a guest
  // body. A native import reached through the dispatcher is a leaf from the
  // guest's point of view; pushing it would put host addresses in a chain the
  // watchdog maps against guest images.
  if (g_dispatch)
    g_dispatch(c, target);
  else
    arc_dispatch_miss(c, target);
  if (watch) arc_watch_report("back", c, target, sret);
  t_current_guest_context = previous_guest;
}

// Big enough for an unwinder or a C++ initialiser, which is what these
// callbacks usually are, and small enough to allocate per call.
#define ARC_CALLBACK_STACK (4u << 20)
#define ARC_CALLBACK_HEADROOM 4096

// One stack per thread, kept and reused, because a comparator is called once
// per comparison and allocating megabytes each time would cost more than the
// sort. The busy flag is what keeps that safe: a callback is entitled to end
// up here again -- an unwinder's visitor can throw, an initialiser can sort --
// and the nested call takes a stack of its own rather than writing over the
// frames of the call it is nested inside.
static ARC_THREAD_LOCAL void* t_callback_stack;
static ARC_THREAD_LOCAL int t_callback_busy;

Arm64Ctx* arc_current_guest_context(void) { return t_current_guest_context; }

uint64_t arc_call_guest(uint64_t fn, const uint64_t* args, int n) {
  Arm64Ctx ctx;
  int i, reused = 0;
  void* stack;
  uint64_t result;

  if (!t_callback_busy) {
    if (!t_callback_stack) t_callback_stack = malloc(ARC_CALLBACK_STACK);
    stack = t_callback_stack;
    reused = 1;
  } else {
    stack = malloc(ARC_CALLBACK_STACK);
  }
  if (!stack) return 0;
  if (reused) t_callback_busy = 1;

  memset(&ctx, 0, sizeof(ctx));
  ctx.sp = (((uint64_t)(uintptr_t)stack + ARC_CALLBACK_STACK -
             ARC_CALLBACK_HEADROOM)) & ~(uint64_t)15;
  for (i = 0; i < n && i < 8; ++i) ctx.x[i] = args[i];
  arc_dispatch(&ctx, fn);
  result = ctx.x[0];

  if (reused)
    t_callback_busy = 0;
  else
    free(stack);
  return result;
}

// A dispatch that says which guest branch made it. The lifted program emits
// this form for indirect branches when it was built with branch notes, which is
// what turns "a branch went nowhere" into the instruction that computed the
// target. The cost is one store, so it is a build-time choice rather than a
// runtime one.
void arc_dispatch_pc(Arm64Ctx* c, uint64_t target, uint64_t pc) {
  t_dispatch_pc = pc;
  arc_dispatch(c, target);
}

const char* arc_native_near(uint64_t address, uint64_t* delta) {
  const char* best = NULL;
  uint64_t at = 0;
  size_t i;
  for (i = 0; i < g_native_count; ++i)
    if (g_natives[i].address <= address && g_natives[i].address >= at) {
      at = g_natives[i].address;
      best = g_natives[i].name;
    }
  for (i = 0; i < g_ctx_native_count; ++i)
    if (g_ctx_natives[i].address <= address && g_ctx_natives[i].address >= at) {
      at = g_ctx_natives[i].address;
      best = g_ctx_natives[i].name;
    }
  if (best && delta) *delta = address - at;
  return best;
}

// Is this span mapped and readable? A trap is the worst time to take a second
// fault: the session is already stopping, and losing the dump to a stack that
// ends inside a guard page would lose the only evidence there is.
static int ArcCommitted(uint64_t address, size_t size) {
#if defined(_WIN32)
  uint64_t end = address + size;
  while (address < end) {
    MEMORY_BASIC_INFORMATION info;
    if (!VirtualQuery((const void*)(uintptr_t)address, &info, sizeof info))
      return 0;
    if (info.State != MEM_COMMIT) return 0;
    if (info.Protect & (PAGE_NOACCESS | PAGE_GUARD)) return 0;
    {
      uint64_t next = (uint64_t)(uintptr_t)info.BaseAddress + info.RegionSize;
      if (next <= address) return 0;
      address = next;
    }
  }
  return 1;
#else
  (void)address;
  (void)size;
  return 1;
#endif
}

static void ArcTrapDump(Arm64Ctx* c) {
  fprintf(stderr, "[regs]");
  for (int i = 0; i <= 28; ++i)
    fprintf(stderr, "%s x%d=%#llx", (i % 6) ? "" : "\n[regs] ", i,
            (unsigned long long)c->x[i]);
  fprintf(stderr, "\n[regs]  x29=%#llx x30=%#llx sp=%#llx\n",
          (unsigned long long)c->x[29], (unsigned long long)c->x[30],
          (unsigned long long)c->sp);
  if (!getenv("ARC_TRAP_DUMP")) {
    fflush(stderr);
    return;
  }
  // The frame the failed branch was computed in. A target read out of a struct
  // is only diagnosable alongside the slots that held the struct and its
  // arguments, and those live in the frame.
  if (c->sp && ArcCommitted(c->sp, 0x400)) {
    for (size_t off = 0; off < 0x400; off += 0x20) {
      const uint64_t* q = (const uint64_t*)(uintptr_t)(c->sp + off);
      fprintf(stderr,
              "[trapdump] %#04zx: %#018llx %#018llx %#018llx %#018llx\n",
              off, (unsigned long long)q[0], (unsigned long long)q[1],
              (unsigned long long)q[2], (unsigned long long)q[3]);
    }
  }
  // And what the registers point at: the object whose field was called, the
  // method it was supposed to hold, and the class of either. Four words is
  // enough to read a class pointer and a vtable slot without guessing.
  {
    uint64_t seen[40];
    size_t count = 0;
    for (int i = 0; i <= 29 && count < 40; ++i) {
      const uint64_t at = c->x[i];
      size_t k;
      int dup = 0;
      if (at < 0x10000 || !ArcCommitted(at, 0x20)) continue;
      for (k = 0; k < count; ++k)
        if (seen[k] == at) dup = 1;
      if (dup) continue;
      seen[count++] = at;
      {
        const uint64_t* q = (const uint64_t*)(uintptr_t)at;
        fprintf(stderr, "[ptrs] x%d -> %#llx: %#llx %#llx %#llx %#llx\n", i,
                (unsigned long long)at, (unsigned long long)q[0],
                (unsigned long long)q[1], (unsigned long long)q[2],
                (unsigned long long)q[3]);
      }
    }
  }
  fflush(stderr);
}

/* A native call used to be a linear scan of the context table and then the
   import table, once per call out of the guest. The loader and the parser make
   millions of them, so remember the resolved entry per thread. The generation
   check covers a registration that lands after a thread has cached a target.
   The fast path is only taken when no diagnostic that needs the full walk is
   on: ARC_MEMCPY_FAST_LOG and ARC_TRACE_CALLS both inspect the resolution. */
typedef struct {
  uint64_t target;
  uint64_t fn;
  const char* name;
  unsigned generation;
  unsigned kind;
} ArcNativeCacheEntry;
#define ARC_NATIVE_CACHE 512
#define ARC_NATIVE_CTX 1
#define ARC_NATIVE_VALUE 2
static ARC_THREAD_LOCAL ArcNativeCacheEntry g_native_cache[ARC_NATIVE_CACHE];

void arc_dispatch_miss(Arm64Ctx* c, uint64_t target) {
  // Reached for every native call the guest makes. getenv here is not a cheap
  // string compare on Windows: it locks and walks the environment block, and
  // at millions of calls it was the largest single cost in the profile. Look
  // each flag up once per process rather than once per call.
  static int trace_miss = -1;
  static int fast_log = -1;
  static const char* call_filter;
  static int call_filter_looked;
  if (trace_miss < 0) trace_miss = getenv("ARC_TRACE_MISS") != NULL;
  if (fast_log < 0) fast_log = getenv("ARC_MEMCPY_FAST_LOG") != NULL;
  if (!call_filter_looked) {
    call_filter_looked = 1;
    call_filter = getenv("ARC_TRACE_CALLS");
  }
  const size_t nslot =
      (size_t)((target >> 2) * UINT64_C(0x9E3779B1)) & (ARC_NATIVE_CACHE - 1);
  ArcNativeCacheEntry* nce = &g_native_cache[nslot];
  if (!fast_log && !call_filter && nce->kind && nce->target == target &&
      nce->generation == g_native_generation) {
    arc_trace_note(nce->name);
    t_last_native_name = nce->name;
    t_last_native_lr = c->x[30];
    t_last_native_lr_shadow = c->lr_shadow;
    t_last_native_args[0] = c->x[0];
    t_last_native_args[1] = c->x[1];
    t_last_native_args[2] = c->x[2];
    if (nce->kind == ARC_NATIVE_CTX) {
      arc_call_ctx_native_guarded(nce->fn, nce->name, c);
    } else {
      const uint64_t* rest = (const uint64_t*)(uintptr_t)c->sp;
      c->x[0] = arc_call_native12_guarded(
          nce->fn, nce->name, c->x[0], c->x[1], c->x[2], c->x[3], c->x[4],
          c->x[5], c->x[6], c->x[7], rest[0], rest[1], rest[2], rest[3]);
    }
    return;
  }
  {
    static int dumped;
    if (!dumped && fast_log) {
      dumped = 1;
      fprintf(stderr, "[fastcopy] enabled=%d natives=%zu ctx=%zu target=%#llx\n",
              ArcSmallCopyEnabled(), g_native_count, g_ctx_native_count,
              (unsigned long long)target);
      for (size_t i = 0; i < g_native_count; ++i)
        if (g_natives[i].name && (strstr(g_natives[i].name, "mem") ||
                                  strstr(g_natives[i].name, "copy")))
          fprintf(stderr, "[fastcopy] native %#llx %s\n",
                  (unsigned long long)g_natives[i].address,
                  g_natives[i].name);
      fflush(stderr);
    }
  }
  if (trace_miss && arc_image_owns(target))
    fprintf(stderr, "[miss] guest target %#llx sp=%#llx x0=%#llx x30=%#llx\n",
            (unsigned long long)target, (unsigned long long)c->sp,
            (unsigned long long)c->x[0], (unsigned long long)c->x[30]);
  if (fast_log && arc_image_bases[0] &&
      c->x[16] == arc_image_bases[0] + UINT64_C(0x5731A90)) {
    static int slot_logged;
    if (!slot_logged) {
      slot_logged = 1;
      uint64_t delta = 0;
      const char* nearby = arc_native_near(target, &delta);
      fprintf(stderr,
              "[fastcopy] memcpy slot target=%#llx near=%s+%#llx n=%llu\n",
              (unsigned long long)target, nearby ? nearby : "?",
              (unsigned long long)delta, (unsigned long long)c->x[2]);
      fflush(stderr);
    }
  }
  if (fast_log && c->x[2] == 4) {
    static int small_logged;
    if (small_logged < 8) {
      ++small_logged;
      uint64_t delta = 0;
      const char* nearby = arc_native_near(target, &delta);
      fprintf(stderr,
              "[fastcopy] small target=%#llx near=%s+%#llx small_target=%#llx "
              "image=%d\n",
              (unsigned long long)target, nearby ? nearby : "?",
              (unsigned long long)delta,
              (unsigned long long)g_small_copy_target, arc_image_owns(target));
      fflush(stderr);
    }
  }
  // Context-taking natives first: they are a strict superset of what the
  // thunk can express, so a name registered both ways wants this one.
  for (size_t i = 0; i < g_ctx_native_count; ++i) {
    if (g_ctx_natives[i].address != target) continue;
    if (!g_small_copy_target && g_ctx_natives[i].name &&
        ArcSmallCopyName(g_ctx_natives[i].name)) {
      g_small_copy_target = target;
      if (fast_log)
        fprintf(stderr, "[fastcopy] set (ctx) target=%#llx name=%s\n",
                (unsigned long long)target, g_ctx_natives[i].name);
    }
    nce->target = target;
    nce->fn = (uint64_t)(uintptr_t)g_ctx_natives[i].fn;
    nce->name = g_ctx_natives[i].name;
    nce->generation = g_native_generation;
    nce->kind = ARC_NATIVE_CTX;
    arc_trace_note(g_ctx_natives[i].name);
    t_last_native_name = g_ctx_natives[i].name;
    t_last_native_lr = c->x[30];
    t_last_native_lr_shadow = c->lr_shadow;
    t_last_native_args[0] = c->x[0];
    t_last_native_args[1] = c->x[1];
    t_last_native_args[2] = c->x[2];
    arc_call_ctx_native_guarded(
        (uint64_t)(uintptr_t)g_ctx_natives[i].fn, g_ctx_natives[i].name, c);
    return;
  }
  for (size_t i = 0; i < g_native_count; ++i) {
    if (g_natives[i].address != target) continue;
    if (!g_small_copy_target && g_natives[i].name &&
        ArcSmallCopyName(g_natives[i].name)) {
      g_small_copy_target = target;
      if (fast_log)
        fprintf(stderr, "[fastcopy] set target=%#llx name=%s\n",
                (unsigned long long)target, g_natives[i].name);
    }
    nce->target = target;
    nce->fn = (uint64_t)(uintptr_t)target;
    nce->name = g_natives[i].name;
    nce->generation = g_native_generation;
    nce->kind = ARC_NATIVE_VALUE;
    arc_trace_note(g_natives[i].name);
    /* Optional argument trace. A name in the trail says the guest called
       memmove; the arguments say whether it asked for a sane length. Set
       ARC_TRACE_CALLS to a substring to see only the calls that matter. */
    {
      static const char* filter;
      static int checked;
      if (!checked) { filter = getenv("ARC_TRACE_CALLS"); checked = 1; }
      /* "*" matches everything: a shell cannot easily pass an empty value.
         Otherwise the filter is a comma-separated list of substrings, because
         a sequence is what answers most questions -- which call a socket got
         between connect and being closed is not visible one name at a time. */
      int matched = 0;
      if (filter && (!*filter || filter[0] == '*')) {
        matched = 1;
      } else if (filter) {
        const char* p = filter;
        while (*p && !matched) {
          const char* comma = strchr(p, ',');
          const size_t n = comma ? (size_t)(comma - p) : strlen(p);
          if (n) {
            char term[64];
            const size_t k = n < sizeof term - 1 ? n : sizeof term - 1;
            memcpy(term, p, k);
            term[k] = 0;
            if (strstr(g_natives[i].name, term)) matched = 1;
          }
          p = comma ? comma + 1 : p + n;
        }
      }
      if (matched)
        {
          const int saved_errno = errno;
          /* A pointer says nothing about which file was wanted. For the
             calls whose first argument is a path by contract, the name
             is the whole point of the trace. */
          static const char* const kPathFirst[] = {
              "fopen", "fopen64", "open", "open64", "stat", "stat64",
              "lstat", "access", "opendir", "unlink", "mkdir",
              /* Not a path, but the same argument: a name, and the whole
                 content of the event. Which host a title resolves is what
                 says where it thinks its server is. */
              "getaddrinfo", "gethostbyname", 0};
          int path_first = 0;
          for (const char* const* q = kPathFirst; *q; ++q)
            if (strcmp(g_natives[i].name, *q) == 0) { path_first = 1; break; }
          if (path_first && c->x[0])
            fprintf(stderr, "[call] t%u %-12s \"%.160s\"\n", arc_trace_thread(),
                    g_natives[i].name, (const char*)(uintptr_t)c->x[0]);
          else
            fprintf(stderr, "[call] t%u %-12s x0=%#llx x1=%#llx x2=%#llx\n",
                    arc_trace_thread(),
                    g_natives[i].name, (unsigned long long)c->x[0],
                    (unsigned long long)c->x[1],
                    (unsigned long long)c->x[2]);
          errno = saved_errno;
        }
    }
    ArcNative12 fn = (ArcNative12)(uintptr_t)target;
    /* Arguments past the eighth sit at the stack pointer, in order. A callee
       taking fewer simply never looks at them. */
    const uint64_t* rest = (const uint64_t*)(uintptr_t)c->sp;
    t_last_native_name = g_natives[i].name;
    t_last_native_lr = c->x[30];
    t_last_native_lr_shadow = c->lr_shadow;
    t_last_native_args[0] = c->x[0];
    t_last_native_args[1] = c->x[1];
    t_last_native_args[2] = c->x[2];
    uint64_t r = arc_call_native12_guarded(
        (uint64_t)(uintptr_t)fn, g_natives[i].name, c->x[0], c->x[1], c->x[2],
        c->x[3], c->x[4], c->x[5], c->x[6], c->x[7], rest[0], rest[1],
        rest[2], rest[3]);
    c->x[0] = r;
    return;
  }
  {
    char msg[512];
    const char* what = g_explain ? g_explain(target) : NULL;
    // A branch that goes nowhere is a question about *which* branch. The
    // program names the site in the context when it was lifted with branch
    // notes, so say it here rather than leaving it to be inferred from the
    // function it happened in. The registers that computed the target come
    // with it: a branch through a struct field is only diagnosable if the
    // pointer the field was read from is in front of you.
    if (c->pc)
      snprintf(msg, sizeof(msg),
               "indirect branch to %#llx, neither lifted nor a known import%s%s"
               " (branch at %u:%llx)",
               (unsigned long long)target, what ? " -- " : "",
               what ? what : "", (unsigned)(c->pc >> 56),
               (unsigned long long)(c->pc & 0x00FFFFFFFFFFFFFFull));
    else
      snprintf(msg, sizeof(msg),
               "indirect branch to %#llx, neither lifted nor a known import%s%s",
               (unsigned long long)target, what ? " -- " : "",
               what ? what : "");
    {
      const size_t used = strlen(msg);
      snprintf(msg + used, sizeof msg - used,
               " x0=%#llx x1=%#llx x2=%#llx x3=%#llx x4=%#llx x8=%#llx"
               " sp=%#llx lr=%#llx",
               (unsigned long long)c->x[0], (unsigned long long)c->x[1],
               (unsigned long long)c->x[2], (unsigned long long)c->x[3],
               (unsigned long long)c->x[4], (unsigned long long)c->x[8],
               (unsigned long long)c->sp, (unsigned long long)c->x[30]);
    }
    ArcTrapDump(c);
    arc_trap(c, msg);
  }
}

// The fallback for a host with no lifted program is above, in arc_dispatch
// itself: with nothing installed it goes straight to arc_dispatch_miss, which
// resolves the branch against the registered natives or traps loudly. That is
// the right behaviour for triage, loading, or bringing the shim up on an arm64
// machine, and it needs no compile-time flag to select it.
