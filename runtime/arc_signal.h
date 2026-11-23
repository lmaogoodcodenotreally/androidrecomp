// Cooperative delivery of guest POSIX signals.
//
// The engine's collector stops the world by sending SIGPWR to every thread and
// waiting for each one to run its suspend handler. A desktop port has no way to
// run a guest handler asynchronously on another thread, so the signal is
// recorded per host thread and the handler is entered at the next dispatcher
// boundary on that thread. The handler runs on the interrupted guest stack, so
// the collector sees the same stack it would have seen from a real signal.
#ifndef ANDROIDRECOMP_ARC_SIGNAL_H
#define ANDROIDRECOMP_ARC_SIGNAL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*ArcSignalPostFn)(void* thread_handle, int sig);

// The guest context for the dispatcher call currently running on this host
// thread, or null outside one. A blocking shim needs this to name the guest
// return address that asked for the wait; the frame ring only records targets,
// not callers.
struct Arm64Ctx;
struct Arm64Ctx* arc_current_guest_context(void);

// Record the guest handler registered for `sig`. `handler` is a guest address.
void arc_signal_set_handler(int sig, uint64_t handler);
uint64_t arc_signal_handler(int sig);

// Ask the guest thread identified by a host thread handle to run `sig` at its
// next dispatcher entry. A null handle means the calling thread.
void arc_signal_post_handle(void* thread_handle, int sig);
// Same, but naming the host thread directly. Unlike a handle this survives
// pthread_detach(), which closes the handle while the thread is still alive.
void arc_signal_post_host(unsigned int host_id, int sig);
void arc_signal_post_current(int sig);

// Called by the dispatcher. Returns the next signal pending for this thread, or
// zero when there is none.
int arc_signal_take_current(void);
// True when a signal is waiting for this thread, without consuming it.
int arc_signal_pending_current(void);
// Run any pending handlers on the calling guest thread. Safe to call from a
// blocking shim after releasing the lock it would otherwise hold.
void arc_run_pending_signals(void);
// Publish what the calling guest thread is blocked in, for diagnostics.
void arc_set_wait_reason(const char* reason);

#ifdef __cplusplus
}
#endif

#endif
