// Resolves the engine's Android imports against the host.
//
// Three layers, in order:
//   1. Explicit implementations -- Android-only entry points that no host
//      runtime can provide, and library bindings we link ourselves.
//   2. Name aliases -- the same function under a different name on this host
//      (POSIX `strdup` is `_strdup` in the Microsoft CRT). Only where the
//      signature is genuinely identical; a same-name-different-ABI alias is a
//      crash that looks like a game bug.
//   3. The host C runtime, looked up by name at load time. Most of the 245
//      libc and 15 libm imports are ordinary standard C that the host already
//      has, so binding them by name costs no code per symbol.
//
// Whatever none of the three answers is the hand-written work list, and
// `arc_host` prints it.
#pragma once

#include <cstddef>
#include <cstdint>

namespace arc {

// How many CPUs the virtual /proc and /sys files should describe. Honours
// ARC_CORES and defaults to the same count sysconf reports.
int64_t HostCoreCount();

// Host address for an imported symbol, or 0 if nothing here provides it.
uint64_t ShimResolve(const char* name);

// How many symbols layer 1 covers, for the coverage report.
size_t ShimExplicitCount();

// A host function pointer handed to the guest at run time, from `dlsym` or
// `eglGetProcAddress`. Imports are announced to the dispatcher when they bind
// at load time; a pointer produced later has no such moment, and this is it.
// Without it the guest branches straight to host code the dispatcher has never
// heard of, and the branch is reported as going nowhere. Returns the address,
// so it can wrap a return value. Only ever pass a *host* address: registering
// a guest one would have it called as if it were host code.
uint64_t ShimHandOut(uint64_t address, const char* name);

// Optional native plugins. A title may depend on a small Android library that
// is not one of the core images and whose behaviour the port can answer with
// host code. Registering it here gives dlopen a non-null handle and dlsym a
// host thunk, without teaching the loader a title-specific name. `generic`
// answers every symbol that has no explicit entry; the symbol entry is for the
// few calls whose result is actually observed.
typedef uint64_t (*ArcPluginFn)(uint64_t, uint64_t, uint64_t, uint64_t,
                                uint64_t, uint64_t, uint64_t, uint64_t,
                                uint64_t, uint64_t, uint64_t, uint64_t);
void ShimRegisterPlugin(const char* soname, ArcPluginFn generic);
void ShimRegisterPluginSymbol(const char* soname, const char* symbol,
                              ArcPluginFn fn);
// A library whose answers depend on which symbol was asked for -- one calling
// convention carrying strings, handles and status codes -- cannot be described
// by a generic entry plus a handful of names. A resolver sees each symbol once,
// while dlsym still has its name, and returns the thunk that fits; the thunk
// itself is then reached without knowing its symbol, which is what a native
// call looks like anyway.
typedef ArcPluginFn (*ArcPluginResolve)(const char* symbol);
void ShimRegisterPluginResolver(const char* soname, ArcPluginResolve resolve);

// Threads, semaphores and thread-local keys. Kept in its own file because it
// carries real state rather than forwarding to something the host already has.
uint64_t ShimResolvePthread(const char* name);
size_t ShimPthreadCount();

// POSIX, BSD and locale entry points the host CRT does not export by name.
// The one monotonic clock. Android's SystemClock.uptimeMillis() and
// CLOCK_MONOTONIC are the same clock, and anything that reads one and
// compares against the other -- an HTTP stack timing a connect is the usual
// one -- needs them to still be the same clock here.
uint64_t ShimMonotonicMillis();

// A guest symbol's address, from the images the loader mapped. The shim
// occasionally has to call *into* the guest rather than answer a call out of
// it -- a Java callback the host has to deliver itself, say -- and the entry
// point for that is a guest symbol like any other.
uint64_t ShimGuestSymbol(const char* name);

// Trace a stream being closed, by the path it was opened with. A no-op
// unless ARC_TRACE_FILES is set.
void ShimTraceClose(void* stream);

// A host implementation that must win over a lifted dependency's, or 0.
// Consulted before anything else when an import is bound.
uint64_t ShimOverride(const char* name);

// Stop leaking freed memory, once whatever ARC_LEAK_FREE was covering for
// is behind us. Leaking is affordable for a boot and not for a session:
// measured at about 875 MB a minute during play. Blocks already leaked stay
// leaked -- only frees after this call are honoured.
void ShimStopLeaking();

// The float-taking and float-returning imports, bound as context-taking
// natives because the integer bridge cannot pass or return an FP register.
uint64_t ShimResolveMath(const char* name);

uint64_t ShimResolvePosix(const char* name);
size_t ShimPosixCount();

// Functions taking the guest's own va_list, which is a structure with
// separate register save areas rather than the host's flat pointer.
uint64_t ShimResolveVarargs(const char* name);

// Announces the variadic handlers to the dispatcher. Call once at startup.
void ShimRegisterVarargs();

// One integer-class argument out of a guest va_list, advancing the caller's
// own cursor. Copy the guest's va_list into a buffer of ShimVaListSize() bytes
// and walk that: a Java method called through Call*MethodV receives its
// arguments this way.
uint64_t ShimVaNextInt(void* cursor);
size_t ShimVaListSize();

// GL entry points, asked of the GL driver by name. Answers only once a context
// is current, which is why the window is opened before the engine is loaded.
uint64_t ShimResolveGL(const char* name);
uint64_t ShimResolveEGL(const char* name);

// The window and context the EGL shims hand out. The engine moves the context
// between its own threads with `eglMakeCurrent`, so the shim needs the pair the
// host created rather than whichever thread happens to be asking.
void ShimSetGfxWindow(void* window, void* context);

// Whether the engine may swap this frame, and whether it has been presenting on
// its own. The host owns the occlusion decision; the engine owns the swap.
void ShimGfxAllowSwap(bool allow);
bool ShimGfxPresents();

// The GL calls that take a float by value need the guest context to find it,
// so the dispatcher has to know them as context natives before the first one
// is reached. Called once, beside ShimRegisterVarargs.
void ShimRegisterGL();

// zlib, whose z_stream is a different size on this host than in the guest --
// see shim_zlib.cpp. Bound before the by-name table, which would forward
// straight through.
uint64_t ShimResolveZlib(const char* name);

// Entropy, from the same source the /dev/urandom stand-in uses. The syscall
// shim needs it: a TLS stack that cannot find getrandom as a symbol asks for
// it by number instead.
void ShimFillRandom(void* buf, size_t n);

// Winsock's last error, translated into errno using the guest libc's numbers
// rather than the CRT's, and returned. Every socket call that can fail has to
// go through this: the guest tests errno against Linux constants, and the two
// disagree exactly where it matters most (EINPROGRESS is 115 there, 112 here).
int ShimNetErrno();

// The host socket behind a descriptor the guest was handed, or 0 if that
// descriptor is not a socket this host opened. Android's descriptors and
// Windows' socket handles are different namespaces with different ranges: a
// SOCKET is a large opaque value, while a Bionic fd_set is a 1024-bit mask
// indexed by descriptor, so handing the guest a raw SOCKET is how FD_SET stops
// fitting in the caller's mask. The shim keeps the small number and the handle
// apart, and this is the lookup for the file layer, which cannot tell a socket
// from a file without it.
uint64_t ShimHostSocket(int fd);

// The guest's descriptor for a host socket, or -1 when there is no mapping.
int ShimGuestSocket(uint64_t host);

// Drops the mapping for a descriptor the guest has closed.
void ShimForgetGuestSocket(int fd);

// The same trace the socket entry points print, for the file layer: a client
// that speaks to a socket through read() and write() -- which every TLS stack
// wrapping a BIO does -- is otherwise invisible in a network trace, and that is
// exactly the layer where a redirect succeeds and the handshake still fails.
void ShimTraceSocketIo(const char* what, int fd, int64_t want, int64_t got,
                       const void* buf);

// Closes an epoll instance if the descriptor is one, so close() on the value
// epoll_create returned releases it rather than being handed to the CRT.
int ShimCloseEpoll(int fd);
// The Android asset manager, over an ordinary directory. Claimed only once a
// root is set, so that an unconfigured host leaves the names on the
// outstanding-import list rather than opening nothing.
uint64_t ShimResolveAsset(const char* name);
size_t ShimAssetCount();
void ShimSetAssetRoot(const char* path);
const char* ShimAssetRoot();

// The host file an asset name names, by the same rules the asset manager uses.
// A port answering a call that copies an asset out of the APK -- Nabe's Java
// `AndroidUtil.Copy` streams one to a file -- needs exactly that answer, and
// re-deriving it outside this file is how the two mappings drift apart.
bool ShimAssetPath(const char* name, char* out, size_t size);

// The directory holding the ARM64 images, for Java's ClassLoader.findLibrary.
// A bare "libil2cpp.so" would also work with this host's dlopen, but returning
// an absolute path keeps any caller that stats or opens the answer on a real
// file, and it is the answer Android would give.
void ShimSetLibraryRoot(const char* path);
const char* ShimLibraryRoot();

// File I/O, directories and memory mapping, at Bionic's struct layouts.
uint64_t ShimResolveFile(const char* name);
size_t ShimFileCount();

// The host name for a guest path, by the same rules the file entry points use.
// For host code answering a call that carries a path. False if it does not fit.
bool ShimGuestPath(const char* guest, char* out, size_t size);

// Sockets, dynamic linking, process and system.
uint64_t ShimResolveSys(const char* name);
size_t ShimSysCount();

// Whether a guest span is committed and readable right now.
//
// This is `VirtualQuery`, which is a kernel call, and the places that need the
// answer sit on the engine's hottest waits: every condition-variable wait
// probes the object it is waiting on, and every futex wait probes its word.
// Those are the same few arenas asked about over and over -- a job worker waits
// on its own condition a thousand times a second -- so the answer is kept per
// thread by *region*, which is what VirtualQuery reports anyway. A region only
// stops being readable when this port releases or decommits it, and those paths
// call GuestMemoryReleased, which drops every cached region on the next probe.
bool GuestRangeReadable(uint64_t address, size_t size);

// Guest memory was released or its protection changed: cached readability
// answers are no longer trustworthy. Cheap, and only called by the mapping
// entry points.
void GuestMemoryReleased();

// The guest ending the process. Reported rather than allowed to happen quietly,
// so that a title which decides it cannot continue is distinguishable from a
// lifter bug. Both the library call and the raw `exit_group` syscall come
// through here: libil2cpp issues the syscall itself in places, and a run that
// simply stopped with no message was that path.
[[noreturn]] void GuestExit(int status);

// Makes a loaded image visible to dlsym and dl_iterate_phdr. Call it for every
// image the host maps, or guest exception unwinding will not find .eh_frame.
class ElfImage;
void ShimRegisterImage(const ElfImage* image);

}  // namespace arc
