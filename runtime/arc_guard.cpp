// Keep host C++ exceptions from unwinding through lifted guest frames.
//
// A shim is allowed to use the host standard library, but one of its
// exceptions (a filesystem error, an invalid mutex, a failed allocation) has
// no host unwind information for the guest frames above it. Letting such an
// exception escape makes the Microsoft C++ runtime walk a stack it cannot
// understand and turns a recoverable shim error into an access violation.
//
// The dispatcher calls through these two guards instead. A guard reports the
// shim and the exception and returns a neutral value, which is the same place
// a kernel would convert an internal failure into an errno. The guest may then
// take its own error path normally.

#include "arm64_context.h"

#include <cstdint>
#include <cstdio>
#include <exception>

namespace {

using Native12 = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t,
                              uint64_t, uint64_t, uint64_t, uint64_t,
                              uint64_t, uint64_t, uint64_t, uint64_t);
using CtxNative = void (*)(Arm64Ctx*);

void Report(const char* kind, uint64_t fn, const char* name,
            const char* what) {
  fprintf(stderr, "[host] %s threw in %s (%p): %s\n", kind,
          name ? name : "?", reinterpret_cast<void*>(fn),
          what ? what : "unknown");
  fflush(stderr);
}

}  // namespace

extern "C" uint64_t arc_call_native12_guarded(
    uint64_t fn, const char* name, uint64_t a0, uint64_t a1, uint64_t a2,
    uint64_t a3, uint64_t a4, uint64_t a5, uint64_t a6, uint64_t a7,
    uint64_t a8, uint64_t a9, uint64_t a10, uint64_t a11) {
  if (!fn) return 0;
  try {
    return reinterpret_cast<Native12>(fn)(a0, a1, a2, a3, a4, a5, a6, a7, a8,
                                          a9, a10, a11);
  } catch (const std::exception& e) {
    Report("exception", fn, name, e.what());
  } catch (...) {
    Report("exception", fn, name, "non-standard");
  }
  return 0;
}

extern "C" void arc_call_ctx_native_guarded(uint64_t fn, const char* name,
                                            Arm64Ctx* c) {
  if (!fn) return;
  try {
    reinterpret_cast<CtxNative>(fn)(c);
  } catch (const std::exception& e) {
    Report("exception", fn, name, e.what());
  } catch (...) {
    Report("exception", fn, name, "non-standard");
  }
}
