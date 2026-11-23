// Sockets, dynamic linking, and the system calls a game reaches for on Android.
//
// Sockets are mostly a straight pass to the host: `struct sockaddr_in` has the
// same layout on Linux and Windows, so addresses cross unchanged. Two things do
// not, and both would corrupt silently rather than fail:
//
//   * `struct addrinfo`. Bionic descends from NetBSD and orders it
//     ai_addrlen, ai_canonname, ai_addr; glibc swaps the two pointers, and
//     Winsock agrees with Bionic but makes ai_addrlen a size_t. Get the order
//     wrong and the engine is handed a canonical name where it expects a
//     sockaddr -- with no error, because the call still returns 0. So results
//     are translated into guest-shaped nodes.
//
//   * `SOL_SOCKET` option numbers. SO_REUSEADDR is 2 on Linux and 4 on
//     Windows; SO_RCVTIMEO is 20 against 0x1006. Unmapped, a timeout request
//     silently sets something else.
//
// Everything a desktop port has no use for -- fork, signal delivery, syslog --
// is a stub that succeeds. Those are honest: they exist so the engine's error
// paths are not taken, not to pretend the behaviour was reproduced.

#include <cerrno>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <ctime>
#include "arm64_context.h"
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "arm64_context.h"
#include "arc_signal.h"
#include "elf_image.h"
#include "shim.h"

#if defined(_WIN32)
#include <io.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#else
#include <arpa/inet.h>
#include <link.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace arc {
namespace {

// --- images, for dlsym and dl_iterate_phdr ---------------------------------
std::mutex g_images_lock;
std::vector<const ElfImage*> g_images;

// --- sockets ---------------------------------------------------------------

#if defined(_WIN32)
// Winsock needs starting, and nothing in the engine will do it for us.
void EnsureWinsock() {
  static std::once_flag once;
  std::call_once(once, [] {
    WSADATA data;
    WSAStartup(MAKEWORD(2, 2), &data);
  });
}

// Linux spells SOL_SOCKET 1; Winsock spells it 0xFFFF, and 1 there is
// IPPROTO_ICMP. Passing the level through unchanged is not a wrong-but-close
// option value, it fails the whole call with WSAENOPROTOOPT -- which matters
// most for the one nobody calls deliberately: a connect is followed by
// getsockopt(SOL_SOCKET, SO_ERROR) to find out whether it succeeded, and a
// stack that cannot read that treats the connection as dead, closes it and
// tries again. From outside that looks like a host refusing to answer.
constexpr int kGuestSolSocket = 1;
int TranslateSockLevel(int level) {
  return level == kGuestSolSocket ? SOL_SOCKET : level;
}
int TranslateSockopt(int level, int name) {
  if (level != kGuestSolSocket) return name;
  switch (name) {
    case 2: return SO_REUSEADDR;
    case 4: return SO_ERROR;
    case 7: return SO_SNDBUF;
    case 8: return SO_RCVBUF;
    case 9: return SO_KEEPALIVE;
    case 13: return SO_LINGER;
    case 20: return SO_RCVTIMEO;
    case 21: return SO_SNDTIMEO;
    default: return name;
  }
}
#else
void EnsureWinsock() {}
int TranslateSockLevel(int level) { return level; }
int TranslateSockopt(int, int name) { return name; }
#endif

// Address families and getaddrinfo flags, which Bionic spells with the Linux
// kernel's numbers and Winsock spells with its own.
//
// Only two of these actually differ, and both fail silently rather than
// loudly. AF_INET6 is 10 on Linux and 23 here, so a socket() for it is an
// unknown family and a getaddrinfo() hint of it matches nothing. AI_ADDRCONFIG
// is 0x20 on Linux and 0x400 here, and 0x20 here is not a flag at all --
// Winsock rejects the whole call with WSAEINVAL, so the name never resolves,
// the engine never opens a socket, and a title that checks for internet by
// resolving a well-known host decides it has none.
//
// The sockaddr the call returns carries the family too, in its first two
// bytes, so that is translated on the way back and on the way in.
constexpr int kGuestAfInet6 = 10;

// Winsock reports failures through WSAGetLastError and leaves errno alone; the
// guest reads errno, and reads it against Linux's numbers, which are not the
// CRT's. EINPROGRESS is the one that decides everything: a non-blocking connect
// is *expected* to fail with it, and a stack that gets any other answer treats
// the connection as refused. Linux says 115. The MSVC CRT says 112.
constexpr int kGuestEIntr = 4;
constexpr int kGuestEAgain = 11;   // == EWOULDBLOCK on Linux
constexpr int kGuestEInval = 22;
constexpr int kGuestEPipe = 32;
constexpr int kGuestEInProgress = 115;
constexpr int kGuestEAlready = 114;
constexpr int kGuestEIsConn = 106;
constexpr int kGuestEConnRefused = 111;
constexpr int kGuestEConnReset = 104;
constexpr int kGuestETimedOut = 110;
constexpr int kGuestENetUnreach = 101;
constexpr int kGuestEHostUnreach = 113;

int FamilyToHost(int family) {
#if defined(_WIN32)
  return family == kGuestAfInet6 ? AF_INET6 : family;
#else
  return family;
#endif
}

int FamilyToGuest(int family) {
#if defined(_WIN32)
  return family == AF_INET6 ? kGuestAfInet6 : family;
#else
  return family;
#endif
}

#if defined(_WIN32)
// Bionic's AI_* bits, from the Linux headers.
constexpr int kGuestAiPassive = 0x0001;
constexpr int kGuestAiCanonname = 0x0002;
constexpr int kGuestAiNumerichost = 0x0004;
constexpr int kGuestAiV4mapped = 0x0008;
constexpr int kGuestAiAll = 0x0010;
constexpr int kGuestAiAddrconfig = 0x0020;
constexpr int kGuestAiNumericserv = 0x0400;
#endif

int AiFlagsToHost(int flags) {
#if defined(_WIN32)
  int out = 0;
  if (flags & kGuestAiPassive) out |= AI_PASSIVE;
  if (flags & kGuestAiCanonname) out |= AI_CANONNAME;
  if (flags & kGuestAiNumerichost) out |= AI_NUMERICHOST;
  if (flags & kGuestAiV4mapped) out |= AI_V4MAPPED;
  if (flags & kGuestAiAll) out |= AI_ALL;
  if (flags & kGuestAiAddrconfig) out |= AI_ADDRCONFIG;
  if (flags & kGuestAiNumericserv) out |= AI_NUMERICSERV;
  return out;
#else
  return flags;
#endif
}

// A sockaddr the guest handed us, with its family in the host's terms. The
// copy is small and bounded; the alternative is writing into the guest's own
// structure, which it may well reuse.
struct HostSockaddr {
  alignas(8) unsigned char bytes[128];
  const sockaddr* get() const { return reinterpret_cast<const sockaddr*>(bytes); }
};

bool ToHostSockaddr(const void* addr, uint32_t len, HostSockaddr* out) {
  if (!addr || len < sizeof(uint16_t) || len > sizeof out->bytes) return false;
  memcpy(out->bytes, addr, len);
  uint16_t family = 0;
  memcpy(&family, out->bytes, sizeof family);
  family = static_cast<uint16_t>(FamilyToHost(family));
  memcpy(out->bytes, &family, sizeof family);
  return true;
}

void SockaddrToGuest(void* addr, uint32_t len) {
  if (!addr || len < sizeof(uint16_t)) return;
  uint16_t family = 0;
  memcpy(&family, addr, sizeof family);
  family = static_cast<uint16_t>(FamilyToGuest(family));
  memcpy(addr, &family, sizeof family);
}

// Guest descriptor numbers for sockets, on the host where a socket is not a
// descriptor.
//
// On Android a socket *is* a file descriptor: a small number, unique against
// open files, and usable anywhere a descriptor is -- in an fd_set, in poll,
// in dup2. On Windows a SOCKET is an opaque handle from a different namespace,
// and its value is large (a few thousand, once a few hundred have been
// opened). Handing that value to the guest looks harmless until the guest
// fills in an fd_set: a Bionic fd_set is a 1024-bit mask indexed by descriptor,
// so FD_SET(1344) writes past the end of the caller's mask and select() is
// told that nothing is being waited on. The connection then never completes,
// the HTTP stack times out, and the title reports that it has no network --
// with the connection actually open the whole time.
//
// So the number the guest sees is ours: a small one from a range the CRT will
// not hand out for files, mapped to the host socket. Every socket entry point
// translates on the way in, select translates both ways, and the file layer
// asks whether a descriptor is a socket instead of guessing from its value.
constexpr int kGuestSocketFirst = 512;
constexpr int kGuestSocketLast = 1023;

std::mutex g_socket_map_lock;
std::map<int, SOCKET> g_guest_to_host;
std::map<SOCKET, int> g_host_to_guest;
int g_next_guest_socket = kGuestSocketFirst;

// The guest's number for this host socket, allocating one if it has none.
int GuestSocketFor(SOCKET host) {
  std::lock_guard<std::mutex> held(g_socket_map_lock);
  const auto known = g_host_to_guest.find(host);
  if (known != g_host_to_guest.end()) return known->second;
  for (int fd = g_next_guest_socket; fd < kGuestSocketLast; ++fd) {
    if (g_guest_to_host.count(fd)) continue;
    g_guest_to_host[fd] = host;
    g_host_to_guest[host] = fd;
    g_next_guest_socket = fd + 1;
    return fd;
  }
  return -1;
}

// The host socket behind a guest descriptor. A descriptor that is not in the
// table is passed through as it is: a socket the guest obtained some other way
// is still a host socket, and that is the only interpretation that keeps
// working for paths this shim does not know about.
SOCKET HostSocketFor(int fd) {
  std::lock_guard<std::mutex> held(g_socket_map_lock);
  const auto it = g_guest_to_host.find(fd);
  return it == g_guest_to_host.end() ? static_cast<SOCKET>(fd) : it->second;
}

bool TracingNet();

int Socket(int domain, int type, int protocol) {
  EnsureWinsock();
  // Bionic's socket() takes SOCK_NONBLOCK and SOCK_CLOEXEC in the type, which
  // Winsock rejects outright -- and a caller that asks for them gets a plain
  // "Error" back from curl, not a socket. They are bits, not socket kinds, so
  // the kind is the low nibble and the flags are applied afterwards.
  const int kind = type & 0xf;
  const bool nonblock = (type & 0x800) != 0;
  const SOCKET host = ::socket(FamilyToHost(domain), kind, protocol);
  if (host == INVALID_SOCKET) {
    if (TracingNet())
      fprintf(stderr, "[net] socket(domain=%d type=%#x proto=%d) failed wsa=%d\n",
              domain, type, protocol, ::WSAGetLastError());
    ShimNetErrno();
    return -1;
  }
  if (nonblock) {
    u_long on = 1;
    ::ioctlsocket(host, FIONBIO, &on);
  }
  const int fd = GuestSocketFor(host);
  if (fd < 0) {
    ::closesocket(host);
    errno = EMFILE;
    return -1;
  }
  return fd;
}
int Bind(int fd, const void* addr, uint32_t len) {
  HostSockaddr host;
  if (!ToHostSockaddr(addr, len, &host)) return -1;
  return ::bind(HostSocketFor(fd), host.get(), static_cast<int>(len));
}
// What actually happened on a socket, which none of the callers report. A
// connection that is opened and then never written to, or written to and never
// answered, looks identical from outside to one that was never attempted --
// and the engine reports all three as "no internet". ARC_TRACE_NET=1.
bool TracingNet() {
  static const bool on = getenv("ARC_TRACE_NET") != nullptr;
  return on;
}

// Every trace below runs between a call setting errno and the guest reading
// it, and fprintf and inet_ntop are both entitled to set errno themselves.
// Left alone the trace destroys the value it exists to explain -- a run with
// tracing on stops being a run of the same program. Saved and restored.
struct KeepErrno {
  int saved = errno;
  ~KeepErrno() { errno = saved; }
};

void TraceAddr(const char* what, int fd, const sockaddr* sa, int rc) {
  if (!TracingNet()) return;
  KeepErrno keep;
  char host[64] = "?";
  int port = 0;
  if (sa && sa->sa_family == AF_INET) {
    const auto* in = reinterpret_cast<const sockaddr_in*>(sa);
    ::inet_ntop(AF_INET, &in->sin_addr, host, sizeof host);
    port = ntohs(in->sin_port);
  }
  fprintf(stderr, "[net] %s fd=%d %s:%d -> rc=%d err=%d\n", what, fd, host,
          port, rc,
#if defined(_WIN32)
          rc == 0 ? 0 : ::WSAGetLastError());
#else
          rc == 0 ? 0 : errno);
#endif
}

// The bytes themselves, not just how many. A title talking to a server it
// was not built for fails in the content of one request, and the sizes say
// nothing about which one: ARC_TRACE_HTTP=1 prints the payload. Text is
// printed as text and anything else as dots, because a protobuf body next to
// an HTTP header is still worth seeing the shape of.
void TraceBody(const char* what, const void* buf, int64_t rc) {
  static const bool on = getenv("ARC_TRACE_HTTP") != nullptr;
  if (!on || rc <= 0 || !buf) return;
  KeepErrno keep;
  // A body is headers plus content, and only the headers are ever the
  // question. Printing megabytes of asset payload costs more time than the
  // transfer does and buries the exchange it was meant to show, so anything
  // large is summarised instead.
  if (rc > 2048) {
    fprintf(stderr, "[http] %s %lld bytes (body not shown)\n", what,
            (long long)rc);
    return;
  }
  const auto* p = static_cast<const unsigned char*>(buf);
  const int64_t cap = rc < 1400 ? rc : 1400;
  fprintf(stderr, "[http] %s %lld bytes\n", what, (long long)rc);
  for (int64_t i = 0; i < cap; ++i)
    fputc((p[i] >= 0x20 && p[i] < 0x7f) || p[i] == 0x0a ? p[i] : '.', stderr);
  fprintf(stderr, "\n[http] --\n");
}

void TraceIo(const char* what, int fd, int64_t n, int64_t rc) {
  if (!TracingNet()) return;
  KeepErrno keep;
  fprintf(stderr, "[net] %s fd=%d want=%lld -> %lld%s", what, fd,
          static_cast<long long>(n), static_cast<long long>(rc),
          rc < 0 ? "" : "\n");
  if (rc < 0)
#if defined(_WIN32)
    fprintf(stderr, " wsa=%d errno=%d\n", ::WSAGetLastError(), errno);
#else
    fprintf(stderr, " errno=%d\n", errno);
#endif
}

// Where the game's server actually is.
//
// A title built for Android has its server address compiled in and rewritten
// before installation -- for Tapped Out, by patching the library at an exact
// byte offset for an exact sha256. That is a workable answer when you are
// shipping an APK and a hopeless one when you are not: the offset is specific
// to one build, and a library that has already been patched once no longer
// contains the string anybody knows how to find.
//
// A native host does not have to play that game. ARC_SERVER_REDIRECT names an
// address, and every outbound connection that is not already loopback goes
// there instead. The engine keeps believing whatever URL it was built with;
// the connection lands on the sidecar.
//
// Deliberately not a hostname map: the point is to catch connections whose
// destination we could not name in advance, which is the whole problem.
//
// It does not redirect TLS. A title checks for internet by reaching a
// well-known https:// host -- Tapped Out uses www.google.com -- and a sidecar
// speaking plain HTTP cannot answer a handshake. Capturing that check
// guarantees it fails, and the title then never gets as far as asking for its
// server at all. Sending it to the real internet is both what the check is for
// and the only answer that can succeed. A redirect target that is itself on
// 443 is taken at its word.
constexpr int kTlsPort = 443;

bool RedirectTarget(sockaddr_in* out) {
  static bool looked = false;
  static bool have = false;
  static sockaddr_in target{};
  if (!looked) {
    looked = true;
    const char* spec = getenv("ARC_SERVER_REDIRECT");
    if (spec && *spec) {
      char host[128] = {0};
      int port = 0;
      const char* colon = strrchr(spec, ':');
      const size_t n = colon ? static_cast<size_t>(colon - spec) : strlen(spec);
      if (n < sizeof host) {
        memcpy(host, spec, n);
        port = colon ? atoi(colon + 1) : 80;
        target.sin_family = AF_INET;
        target.sin_port = htons(static_cast<unsigned short>(port));
        if (::inet_pton(AF_INET, host, &target.sin_addr) == 1) have = true;
      }
      fprintf(stderr, "[net] redirecting outbound connections to %s\n",
              have ? spec : "(unparseable, ignored)");
    }
  }
  if (have) *out = target;
  return have;
}

// Loopback is left alone: a redirect that also catches the sidecar would send
// it to itself.
bool IsLoopback(const sockaddr* sa) {
  if (!sa || sa->sa_family != AF_INET) return false;
  const auto* in = reinterpret_cast<const sockaddr_in*>(sa);
  return (ntohl(in->sin_addr.s_addr) >> 24) == 127;
}

bool TlsRedirectAllowed() {
  static const bool allow = getenv("ARC_REDIRECT_TLS") != nullptr;
  return allow;
}

bool ShouldRedirect(const sockaddr* to, const sockaddr_in& target) {
  if (!to || to->sa_family != AF_INET) return false;
  const auto* in = reinterpret_cast<const sockaddr_in*>(to);
  // Already the sidecar, address and port both. Anything else that happens to
  // be loopback is not: a title that resolves its server to 127.0.0.1 because
  // this host answered the lookup keeps its own port, and the sidecar never
  // listens there. Leaving that alone is how the connection goes to a closed
  // port on the right machine instead of the right port on it.
  if (in->sin_addr.s_addr == target.sin_addr.s_addr &&
      in->sin_port == target.sin_port)
    return false;
  // A 443 destination normally means "reachability check", which a plaintext
  // sidecar cannot answer. A sidecar that itself terminates TLS is a different
  // case, and the title's own API is on 443 too, so the port mismatch is only
  // refused unless the port explicitly says its target speaks TLS.
  if (ntohs(in->sin_port) == kTlsPort && ntohs(target.sin_port) != kTlsPort &&
      !TlsRedirectAllowed())
    return false;
  return true;
}

// A connect on a non-blocking socket does not fail when it returns -1 -- it
// has started, and the caller is expected to see EINPROGRESS and wait for the
// socket to become writable. Winsock says WSAEWOULDBLOCK for that, which the
// general mapping turns into EAGAIN; here it has to be EINPROGRESS, or the
// caller reads "temporarily out of resources" and gives up on the address.
int ConnectResult(int fd, const sockaddr* sa, int rc, const char* what) {
#if defined(_WIN32)
  if (rc != 0) {
    const int wsa = ::WSAGetLastError();
    ShimNetErrno();
    if (wsa == WSAEWOULDBLOCK) errno = kGuestEInProgress;
  }
#endif
  TraceAddr(what, fd, sa, rc);
  return rc;
}

int Connect(int fd, const void* addr, uint32_t len) {
  HostSockaddr host;
  if (!ToHostSockaddr(addr, len, &host)) {
    errno = kGuestEInval;
    return -1;
  }
  const SOCKET socket = HostSocketFor(fd);
  sockaddr_in target{};
  if (RedirectTarget(&target) && ShouldRedirect(host.get(), target)) {
    const int rc = ::connect(socket, reinterpret_cast<const sockaddr*>(&target),
                             static_cast<int>(sizeof target));
    return ConnectResult(fd, reinterpret_cast<const sockaddr*>(&target), rc,
                         "connect(redirected)");
  }
  const int rc = ::connect(socket, host.get(), static_cast<int>(len));
  return ConnectResult(fd, host.get(), rc, "connect");
}
int Listen(int fd, int backlog) { return ::listen(HostSocketFor(fd), backlog); }
int Accept(int fd, void* addr, uint32_t* len) {
  sockaddr_storage from{};
  socklen_t from_len = sizeof from;
  const SOCKET accepted =
      ::accept(HostSocketFor(fd), reinterpret_cast<sockaddr*>(&from), &from_len);
  if (accepted == INVALID_SOCKET) {
    ShimNetErrno();
    return -1;
  }
  const int guest = GuestSocketFor(accepted);
  if (guest < 0) {
    ::closesocket(accepted);
    errno = EMFILE;
    return -1;
  }
  // The address goes back in the guest's numbering, as recvfrom's does.
  if (addr && len) {
    const uint32_t room = *len;
    const uint32_t take = room < from_len ? room : static_cast<uint32_t>(from_len);
    memcpy(addr, &from, take);
    *len = take;
    SockaddrToGuest(addr, take);
  }
  return guest;
}
// Bionic's MSG_* bits, from the Linux headers. Only the low three agree with
// Winsock's, and the rest are not merely ignored -- Winsock rejects a flag it
// does not recognise by failing the whole call with WSAEOPNOTSUPP.
//
// MSG_NOSIGNAL is the one that matters and the one every HTTP stack sets: on
// Linux it asks send() not to raise SIGPIPE on a closed peer. Forwarded here
// it fails every write on a perfectly good socket, so the request is never
// sent, the reply never comes, and the title reports the server as unreachable.
// Windows never raises SIGPIPE, so dropping the bit is the whole translation.
constexpr int kGuestMsgOob = 0x0001;
constexpr int kGuestMsgPeek = 0x0002;
constexpr int kGuestMsgDontroute = 0x0004;
constexpr int kGuestMsgWaitall = 0x0100;

int MsgFlagsToHost(int flags) {
#if defined(_WIN32)
  int host = 0;
  if (flags & kGuestMsgOob) host |= MSG_OOB;
  if (flags & kGuestMsgPeek) host |= MSG_PEEK;
  if (flags & kGuestMsgDontroute) host |= MSG_DONTROUTE;
  if (flags & kGuestMsgWaitall) host |= MSG_WAITALL;
  // MSG_NOSIGNAL has nothing to signal here, and MSG_DONTWAIT cannot be said
  // per call -- the socket carries that, and fcntl already set it.
  return host;
#else
  return flags;
#endif
}

int64_t Send(int fd, const void* buf, uint64_t n, int flags) {
  const int64_t rc = ::send(HostSocketFor(fd), static_cast<const char*>(buf),
                            static_cast<int>(n), MsgFlagsToHost(flags));
  if (rc < 0) ShimNetErrno();
  TraceIo("send", fd, static_cast<int64_t>(n), rc);
  TraceBody("send", buf, rc);
  return rc;
}
int64_t Recv(int fd, void* buf, uint64_t n, int flags) {
  const int64_t rc = ::recv(HostSocketFor(fd), static_cast<char*>(buf),
                            static_cast<int>(n), MsgFlagsToHost(flags));
  if (rc < 0) ShimNetErrno();
  TraceIo("recv", fd, static_cast<int64_t>(n), rc);
  TraceBody("recv", buf, rc);
  return rc;
}
int64_t Sendto(int fd, const void* buf, uint64_t n, int flags,
               const void* addr, uint32_t len) {
  // The same flag translation as Send, and the same address translation as
  // Connect: a sockaddr from the guest carries Bionic's AF_INET6 either way.
  HostSockaddr host;
  const bool have = addr && ToHostSockaddr(addr, len, &host);
  const int64_t rc =
      ::sendto(HostSocketFor(fd), static_cast<const char*>(buf),
               static_cast<int>(n), MsgFlagsToHost(flags),
               have ? host.get() : nullptr,
               have ? static_cast<int>(len) : 0);
  if (rc < 0) ShimNetErrno();
  TraceIo("sendto", fd, static_cast<int64_t>(n), rc);
  return rc;
}
int64_t Recvfrom(int fd, void* buf, uint64_t n, int flags, void* addr,
                 uint32_t* len) {
  const int64_t rc =
      ::recvfrom(HostSocketFor(fd), static_cast<char*>(buf),
                 static_cast<int>(n), MsgFlagsToHost(flags),
                 static_cast<sockaddr*>(addr),
                 reinterpret_cast<socklen_t*>(len));
  if (rc < 0) ShimNetErrno();
  // The family goes back in the guest's numbering, as it does from accept.
  if (rc >= 0 && addr && len) SockaddrToGuest(addr, *len);
  TraceIo("recvfrom", fd, static_cast<int64_t>(n), rc);
  return rc;
}
int Shutdown(int fd, int how) { return ::shutdown(HostSocketFor(fd), how); }
int Setsockopt(int fd, int level, int name, const void* val, uint32_t len) {
  const int host_level = TranslateSockLevel(level);
  const int host_name = TranslateSockopt(level, name);
  const SOCKET socket = HostSocketFor(fd);
#if defined(_WIN32)
  // Two options carry a payload that differs as well as a number. Both are set
  // by every HTTP stack, and both fail the call outright if forwarded as-is.
  // A timeout is a struct timeval (two 64-bit fields) on Linux and a DWORD of
  // milliseconds here; linger is two ints there and two shorts here.
  if (host_level == SOL_SOCKET &&
      (host_name == SO_RCVTIMEO || host_name == SO_SNDTIMEO) && val &&
      len >= 16) {
    int64_t tv[2] = {0, 0};
    memcpy(tv, val, sizeof tv);
    const DWORD ms = static_cast<DWORD>(tv[0] * 1000 + tv[1] / 1000);
    return ::setsockopt(socket, host_level, host_name,
                        reinterpret_cast<const char*>(&ms), sizeof ms);
  }
  if (host_level == SOL_SOCKET && host_name == SO_LINGER && val && len >= 8) {
    int32_t guest[2] = {0, 0};
    memcpy(guest, val, sizeof guest);
    ::linger host{static_cast<u_short>(guest[0]), static_cast<u_short>(guest[1])};
    return ::setsockopt(socket, host_level, host_name,
                        reinterpret_cast<const char*>(&host), sizeof host);
  }
#endif
  return ::setsockopt(socket, host_level, host_name,
                      static_cast<const char*>(val), len);
}
int Getsockopt(int fd, int level, int name, void* val, uint32_t* len) {
  const int host_level = TranslateSockLevel(level);
  const int host_name = TranslateSockopt(level, name);
  const int rc = ::getsockopt(HostSocketFor(fd), host_level, host_name,
                              static_cast<char*>(val),
                              reinterpret_cast<socklen_t*>(len));
  if (TracingNet()) {
    fprintf(stderr, "[net] getsockopt fd=%d level=%d opt=%d -> rc=%d val=%d\n",
            fd, level, name, rc,
            (val && len && *len >= 4) ? *static_cast<const int*>(val) : -1);
  }
  return rc;
}
int Getsockname(int fd, void* addr, uint32_t* len) {
  return ::getsockname(HostSocketFor(fd), static_cast<sockaddr*>(addr),
                       reinterpret_cast<socklen_t*>(len));
}
int Getpeername(int fd, void* addr, uint32_t* len) {
  return ::getpeername(HostSocketFor(fd), static_cast<sockaddr*>(addr),
                       reinterpret_cast<socklen_t*>(len));
}
int Gethostname(char* name, uint64_t len) {
  EnsureWinsock();
  return ::gethostname(name, static_cast<int>(len));
}
#if defined(_WIN32)
// A Linux fd_set is a bitmask indexed by descriptor; Winsock's is a count and
// an array of SOCKETs. Translated both ways, because a caller reads back which
// of its descriptors are ready -- and the descriptors in that mask are the
// guest's small numbers, not the host handles behind them.
void GuestFdsToHost(const void* guest, int nfds, fd_set* host) {
  FD_ZERO(host);
  if (!guest) return;
  const auto* bits = static_cast<const uint64_t*>(guest);
  for (int fd = 0; fd < nfds && fd < 1024; ++fd)
    if (bits[fd / 64] & (uint64_t{1} << (fd % 64)))
      FD_SET(HostSocketFor(fd), host);
}

void HostFdsToGuest(const fd_set* host, void* guest, int nfds) {
  if (!guest) return;
  auto* bits = static_cast<uint64_t*>(guest);
  for (int i = 0; i < (nfds + 63) / 64 && i < 16; ++i) bits[i] = 0;
  for (u_int i = 0; i < host->fd_count; ++i) {
    int fd = ShimGuestSocket(host->fd_array[i]);
    if (fd < 0) fd = static_cast<int>(host->fd_array[i]);
    if (fd >= 0 && fd < 1024) bits[fd / 64] |= uint64_t{1} << (fd % 64);
  }
}
#endif

int Select(int n, void* r, void* w, void* e, void* timeout) {
#if defined(_WIN32)
  EnsureWinsock();
  fd_set hr, hw, he;
  GuestFdsToHost(r, n, &hr);
  GuestFdsToHost(w, n, &hw);
  GuestFdsToHost(e, n, &he);
  // A Bionic `struct timeval` is two 64-bit fields; Winsock's is two 32-bit
  // ones. Forwarded as it is, the timeout is read out of the middle of the
  // guest's struct -- so a sub-second wait becomes a full second and a
  // one-second wait becomes whatever follows it on the stack.
  timeval host_timeout{};
  timeval* timeout_arg = nullptr;
  if (timeout) {
    int64_t tv[2] = {0, 0};
    memcpy(tv, timeout, sizeof tv);
    host_timeout.tv_sec = static_cast<long>(tv[0]);
    host_timeout.tv_usec = static_cast<long>(tv[1]);
    timeout_arg = &host_timeout;
  }
  int rc;
  const auto started = std::chrono::steady_clock::now();
  if (hr.fd_count == 0 && hw.fd_count == 0 && he.fd_count == 0) {
    // Nothing to wait on. Winsock rejects that outright (WSAEINVAL); POSIX
    // treats it as a plain sleep, and callers use it as one.
    const int64_t micros =
        timeout_arg ? static_cast<int64_t>(host_timeout.tv_sec) * 1000000 +
                          host_timeout.tv_usec
                    : 1000;
    if (micros > 0)
      std::this_thread::sleep_for(std::chrono::microseconds(micros));
    rc = 0;
  } else {
    rc = ::select(n, r ? &hr : nullptr, w ? &hw : nullptr,
                  e ? &he : nullptr,
                  timeout_arg);
    if (rc < 0) ShimNetErrno();
  }
  if (TracingNet()) {
    KeepErrno keep;
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - started)
                             .count();
    fprintf(stderr,
            "[net] select n=%d r=%u w=%u e=%u -> rc=%d in %lldms\n", n,
            r ? hr.fd_count : 0u, w ? hw.fd_count : 0u, e ? he.fd_count : 0u,
            rc, static_cast<long long>(elapsed));
  }
  if (r) HostFdsToGuest(&hr, r, n);
  if (w) HostFdsToGuest(&hw, w, n);
  if (e) HostFdsToGuest(&he, e, n);
  return rc;
#else
  return ::select(n, static_cast<fd_set*>(r), static_cast<fd_set*>(w),
                  static_cast<fd_set*>(e), static_cast<timeval*>(timeout));
#endif
}
const char* InetNtop(int af, const void* src, char* dst, uint32_t size) {
  return ::inet_ntop(af, src, dst, size);
}
int InetPton(int af, const char* src, void* dst) {
  return ::inet_pton(af, src, dst);
}

// Bionic's addrinfo, in Bionic's order -- which is BSD's, not glibc's.
//
// This is the field order the file's own header comment got backwards. glibc
// puts ai_addr before ai_canonname; Bionic descends from NetBSD and puts
// ai_canonname first, which happens to be Winsock's order too. Built the glibc
// way, every resolution "succeeded" and handed the engine a canonical name
// where it expected a sockaddr -- so it never opened a socket, never
// connected, and decided it had no internet. There is no error anywhere in
// that sequence: the call returns 0 and the answer is furniture.
struct GuestAddrinfo {
  int32_t ai_flags;
  int32_t ai_family;
  int32_t ai_socktype;
  int32_t ai_protocol;
  uint32_t ai_addrlen;
  char* ai_canonname;
  sockaddr* ai_addr;
  GuestAddrinfo* ai_next;
};

void FreeGuestAddrinfo(GuestAddrinfo* list) {
  while (list) {
    GuestAddrinfo* next = list->ai_next;
    free(list->ai_addr);
    free(list->ai_canonname);
    free(list);
    list = next;
  }
}

// "127.0.0.1" and "localhost" are already where the sidecar is, and sending
// them through the redirect would only send it to itself.
bool IsLoopbackName(const char* node) {
  if (!node) return false;
  in_addr a{};
  if (::inet_pton(AF_INET, node, &a) == 1)
    return (ntohl(a.s_addr) >> 24) == 127;
  return strcmp(node, "localhost") == 0;
}

// One answer, in the shape getaddrinfo returns and the guest expects.
GuestAddrinfo* OneAddress(const sockaddr_in& addr, const addrinfo* hints) {
  auto* g = static_cast<GuestAddrinfo*>(calloc(1, sizeof(GuestAddrinfo)));
  g->ai_family = FamilyToGuest(AF_INET);
  g->ai_socktype = hints && hints->ai_socktype ? hints->ai_socktype : SOCK_STREAM;
  g->ai_protocol = hints && hints->ai_protocol ? hints->ai_protocol : IPPROTO_TCP;
  g->ai_addrlen = sizeof(sockaddr_in);
  g->ai_addr = static_cast<sockaddr*>(malloc(sizeof(sockaddr_in)));
  memcpy(g->ai_addr, &addr, sizeof(sockaddr_in));
  SockaddrToGuest(g->ai_addr, sizeof(sockaddr_in));
  return g;
}

int Getaddrinfo(const char* node, const char* service, const void* hints_in,
                GuestAddrinfo** out) {
  EnsureWinsock();
  // Hints share the leading four ints, which is all getaddrinfo reads from
  // them, so a fresh host-shaped hints struct is built from those.
  addrinfo hints{};
  const addrinfo* hints_ptr = nullptr;
  if (hints_in) {
    const auto* g = static_cast<const GuestAddrinfo*>(hints_in);
    hints.ai_flags = AiFlagsToHost(g->ai_flags);
    hints.ai_family = FamilyToHost(g->ai_family);
    hints.ai_socktype = g->ai_socktype;
    hints.ai_protocol = g->ai_protocol;
    hints_ptr = &hints;
  }

  // A name whose connection is going to be redirected anyway does not need
  // resolving, and resolving it is not free: the servers this title was built
  // against are gone, so every lookup runs to the resolver's own timeout on a
  // thread the HTTP stack made for it. It then gives up, makes another
  // thread, and asks again. That is the whole of the thread storm -- thousands
  // of threads, each holding a stack, and a process that dies of memory
  // exhaustion long before it finishes loading. Answering here with the
  // address the connection was going to be sent to costs nothing and is the
  // same answer.
  //
  // TLS is the exception, as it is at connect: the reachability check reaches
  // a real well-known host over 443, and only a real lookup can answer it.
  {
    sockaddr_in target{};
    const int port = service ? atoi(service) : 0;
    if (node && RedirectTarget(&target) &&
        (port != kTlsPort || TlsRedirectAllowed()) && !IsLoopbackName(node)) {
      *out = OneAddress(target, hints_ptr);
      if (TracingNet())
        fprintf(stderr, "[net] getaddrinfo(%s, %s) answered with the redirect "
                        "target\n", node, service ? service : "(null)");
      return 0;
    }
  }

  addrinfo* host = nullptr;
  int rc = ::getaddrinfo(node, service, hints_ptr, &host);
  if (rc != 0) {
    // A name that will not resolve is the end of the road for whatever asked,
    // and the caller usually reports it as its own failure -- "no internet",
    // "cannot reach the server" -- with nothing saying which name or why.
    fprintf(stderr, "[net] getaddrinfo(%s, %s) failed: %d (flags=%#x fam=%d)\n",
            node ? node : "(null)", service ? service : "(null)", rc,
            hints_ptr ? hints_ptr->ai_flags : 0,
            hints_ptr ? hints_ptr->ai_family : -1);
    return rc;
  }

  GuestAddrinfo* head = nullptr;
  GuestAddrinfo** tail = &head;
  for (addrinfo* h = host; h; h = h->ai_next) {
    auto* g = static_cast<GuestAddrinfo*>(calloc(1, sizeof(GuestAddrinfo)));
    g->ai_flags = h->ai_flags;
    g->ai_family = FamilyToGuest(h->ai_family);
    g->ai_socktype = h->ai_socktype;
    g->ai_protocol = h->ai_protocol;
    g->ai_addrlen = static_cast<uint32_t>(h->ai_addrlen);
    if (h->ai_addr && h->ai_addrlen) {
      g->ai_addr = static_cast<sockaddr*>(malloc(h->ai_addrlen));
      memcpy(g->ai_addr, h->ai_addr, h->ai_addrlen);
      // The family lives in the sockaddr as well as in the addrinfo, and the
      // guest reads both.
      SockaddrToGuest(g->ai_addr, static_cast<uint32_t>(h->ai_addrlen));
    }
    if (h->ai_canonname) {
#if defined(_WIN32)
      g->ai_canonname = _strdup(h->ai_canonname);
#else
      g->ai_canonname = strdup(h->ai_canonname);
#endif
    }
    *tail = g;
    tail = &g->ai_next;
  }
  ::freeaddrinfo(host);
  *out = head;
  return 0;
}

void Freeaddrinfo(GuestAddrinfo* list) { FreeGuestAddrinfo(list); }

const char* GaiStrerror(int) { return "getaddrinfo error"; }
int Getnameinfo(const void*, uint32_t, char* host, uint32_t hostlen, char* serv,
                uint32_t servlen, int) {
  if (host && hostlen) host[0] = '\0';
  if (serv && servlen) serv[0] = '\0';
  return 0;
}
// The IPv4-only entry point older socket layers still use; libunity imports it
// as well as getaddrinfo, and the two do not share an answer. A name whose
// connection is going to be redirected does not need resolving here either, and
// returning null for it is not a neutral answer: the caller reads it as "cannot
// resolve destination host" and abandons the transfer before connect ever runs.
// LP64 puts the same 8-byte pointers and 4-byte ints on both sides of the shim,
// so the hostent is built in the shape the guest reads directly.
struct GuestHostent {
  char* h_name;
  char** h_aliases;
  int32_t h_addrtype;
  int32_t h_length;
  char** h_addr_list;
};

void* Gethostbyname(const char* name) {
  EnsureWinsock();
  sockaddr_in target{};
  if (name && RedirectTarget(&target) && !IsLoopbackName(name)) {
    static thread_local GuestHostent ent;
    static thread_local char text[256];
    static thread_local in_addr addr;
    static thread_local char* list[2];
    static thread_local char* alias[1];
    const size_t n = strlen(name);
    const size_t copy = n < sizeof text - 1 ? n : sizeof text - 1;
    memcpy(text, name, copy);
    text[copy] = '\0';
    addr = target.sin_addr;
    list[0] = reinterpret_cast<char*>(&addr);
    list[1] = nullptr;
    alias[0] = nullptr;
    ent.h_name = text;
    ent.h_aliases = alias;
    ent.h_addrtype = AF_INET;
    ent.h_length = 4;
    ent.h_addr_list = list;
    if (TracingNet())
      fprintf(stderr,
              "[net] gethostbyname(%s) answered with the redirect target\n",
              name);
    return &ent;
  }
  if (TracingNet())
    fprintf(stderr, "[net] gethostbyname(%s) left to the resolver\n",
            name ? name : "(null)");
  return ::gethostbyname(name);
}
unsigned IfNametoindex(const char*) { return 0; }
int Socketpair(int, int, int, int*) { return -1; }
int64_t Sendfile(int, int, void*, uint64_t) { return -1; }
// poll() and select(), which decide when a socket is ready and therefore
// whether anything on the network ever finishes.
//
// Neither could be forwarded. A `struct pollfd` is 8 bytes on Linux -- int fd,
// short events, short revents -- and 16 on Winsock, where the descriptor is a
// SOCKET; and the event bits share no values at all (Linux POLLIN is 0x001,
// Winsock's is 0x300). A Linux `fd_set` is a 1024-bit mask indexed by
// descriptor; Winsock's is a count followed by an array of descriptors.
//
// Poll was a stub returning 0, which says "nothing is ready" forever. That is
// not an error anyone reports: the engine's HTTP client connects, waits to be
// told the socket is readable, is never told, times out, and retries -- so a
// title sits on its loading screen looking like it has no network, having in
// fact opened the connection perfectly well.
constexpr short kGuestPollIn = 0x001;
constexpr short kGuestPollPri = 0x002;
constexpr short kGuestPollOut = 0x004;
constexpr short kGuestPollErr = 0x008;
constexpr short kGuestPollHup = 0x010;
constexpr short kGuestPollNval = 0x020;

struct GuestPollfd {
  int32_t fd;
  int16_t events;
  int16_t revents;
};

int Poll(void* fds_in, unsigned long nfds, int timeout) {
  auto* fds = static_cast<GuestPollfd*>(fds_in);
  // No descriptors is not a malformed call, it is how a caller sleeps: curl
  // waits between connection attempts with poll(NULL, 0, ms) and nothing else.
  // Returning 0 immediately turns that wait into a spin -- the retry loop runs
  // as fast as the CPU allows, which is why a title with no network pins a
  // core -- and it gives an in-flight connect no time to finish before the
  // next attempt is made.
  if (!fds || nfds == 0) {
    if (timeout > 0)
      std::this_thread::sleep_for(std::chrono::milliseconds(timeout));
    return 0;
  }
#if defined(_WIN32)
  EnsureWinsock();
  std::vector<WSAPOLLFD> host(nfds);
  for (unsigned long i = 0; i < nfds; ++i) {
    host[i].fd = HostSocketFor(fds[i].fd);
    short e = 0;
    if (fds[i].events & kGuestPollIn) e |= POLLRDNORM;
    if (fds[i].events & kGuestPollPri) e |= POLLRDBAND;
    if (fds[i].events & kGuestPollOut) e |= POLLWRNORM;
    host[i].events = e;
    host[i].revents = 0;
  }
  const auto started = std::chrono::steady_clock::now();
  const int rc = ::WSAPoll(host.data(), static_cast<ULONG>(nfds), timeout);
  const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::steady_clock::now() - started)
                           .count();
  // Only the waits that are asking "is this connect finished yet", which is
  // the one poll a stalled HTTP stack repeats and the one worth seeing. The
  // readable-data waits are the steady state and would bury it. A wait that
  // took a noticeable time is printed whatever it asked for: that is the
  // request a stack is sitting in while nothing happens, and it is invisible
  // from outside until the call returns.
  if (TracingNet() && ((host[0].events & POLLWRNORM) || elapsed > 1000)) {
    KeepErrno keep;
    fprintf(stderr,
            "[net] poll n=%lu fd=%d ev=%#x t=%d -> rc=%d rev=%#x in %lldms\n",
            nfds, (int)host[0].fd, host[0].events, timeout, rc,
            host[0].revents, static_cast<long long>(elapsed));
  }
  for (unsigned long i = 0; i < nfds; ++i) {
    short r = 0;
    if (host[i].revents & (POLLRDNORM | POLLRDBAND)) r |= kGuestPollIn;
    if (host[i].revents & POLLWRNORM) r |= kGuestPollOut;
    if (host[i].revents & POLLERR) r |= kGuestPollErr;
    if (host[i].revents & POLLHUP) r |= kGuestPollHup;
    if (host[i].revents & POLLNVAL) r |= kGuestPollNval;
    fds[i].revents = r;
  }
  return rc;
#else
  return ::poll(reinterpret_cast<struct pollfd*>(fds), nfds, timeout);
#endif
}
int Ioctl(int, unsigned long, ...) { return 0; }

// --- dynamic linking -------------------------------------------------------
// Every guest library is mapped before the program starts, so dlopen does not
// load anything: it answers with a handle to the image already registered, and
// dlsym then searches that image and the shim. Returning null here used to be
// harmless for a single-image boot, but Unity loads libil2cpp by name and
// treats a null handle as "the scripting backend is missing".
//
// Android system libraries have no guest image -- their symbols are the shim's
// -- so a small list of those names also gets a non-null sentinel and dlsym
// answers from the host.
// Wwise is the one native dependency the managed side loads by name at
// startup, through P/Invoke. Its real library is a 33 MB guest image that
// would have to be lifted in full to run, and audio is not what this port is
// for: what the engine needs from it is that loading succeeds and that its
// calls answer something usable. The integration is SWIG-generated, so its
// exports return three different shapes -- struct handles, char pointers and
// status codes -- through one signature, and no single answer is right for all
// three. This is the loader's fallback for a title that has not described its
// Wwise at all: nothing, answered as zero. A port that knows the shapes
// registers `libAkSoundEngine.so` with a resolver and answers each symbol.
uint64_t AkSoundStub(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
                     uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
                     uint64_t) {
  return 0;
}

// Audio is deliberately not brought up. The integration's LateUpdate runs a
// long managed render pass whenever IsInitialized says yes; with no real
// engine behind the stub that pass has nothing to finish against. Answering
// "not initialized" makes the integration idle instead of spinning.
uint64_t AkSoundNotInitialized(uint64_t, uint64_t, uint64_t, uint64_t,
                               uint64_t, uint64_t, uint64_t, uint64_t,
                               uint64_t, uint64_t, uint64_t, uint64_t) {
  return 0;
}

// Optional plugins registered by the port. The handle handed to the guest is
// the host-side description below, so dlsym can still tell one plugin from
// another without inventing a numbering scheme the loader would have to share.
struct PluginStub {
  std::string name;
  ArcPluginFn generic = nullptr;
  std::map<std::string, ArcPluginFn> symbols;
  ArcPluginResolve resolve = nullptr;
};

std::mutex g_plugin_lock;
std::vector<std::unique_ptr<PluginStub>> g_plugins;

void RegisterPlugin(const char* soname, ArcPluginFn generic) {
  if (!soname || !*soname) return;
  std::lock_guard<std::mutex> g(g_plugin_lock);
  for (const auto& p : g_plugins)
    if (p->name == soname) {
      p->generic = generic;
      return;
    }
  auto p = std::make_unique<PluginStub>();
  p->name = soname;
  p->generic = generic;
  g_plugins.push_back(std::move(p));
}

void RegisterPluginSymbol(const char* soname, const char* symbol,
                          ArcPluginFn fn) {
  if (!soname || !*soname || !symbol || !*symbol) return;
  std::lock_guard<std::mutex> g(g_plugin_lock);
  for (const auto& p : g_plugins)
    if (p->name == soname) {
      p->symbols[symbol] = fn;
      return;
    }
  auto p = std::make_unique<PluginStub>();
  p->name = soname;
  p->symbols[symbol] = fn;
  g_plugins.push_back(std::move(p));
}

PluginStub* FindPlugin(const std::string& base) {
  std::lock_guard<std::mutex> g(g_plugin_lock);
  for (const auto& p : g_plugins)
    if (p->name == base) return p.get();
  return nullptr;
}

void RegisterPluginResolver(const char* soname, ArcPluginResolve resolve) {
  if (!soname || !*soname) return;
  std::lock_guard<std::mutex> g(g_plugin_lock);
  for (const auto& p : g_plugins)
    if (p->name == soname) {
      p->resolve = resolve;
      return;
    }
  auto p = std::make_unique<PluginStub>();
  p->name = soname;
  p->resolve = resolve;
  g_plugins.push_back(std::move(p));
}

// One place decides which thunk a plugin's symbol gets: an explicit entry
// wins, then the resolver sees the name, then the generic entry answers. The
// order matters -- a resolver that classified a symbol differently from an
// explicit registration would make the port's table and its code disagree.
ArcPluginFn PluginAnswer(PluginStub* plugin, const char* symbol) {
  ArcPluginFn fn = plugin->generic;
  if (!symbol) return fn;
  const auto it = plugin->symbols.find(symbol);
  if (it != plugin->symbols.end()) return it->second;
  if (plugin->resolve) {
    if (ArcPluginFn resolved = plugin->resolve(symbol)) return resolved;
  }
  return fn;
}

PluginStub* PluginForHandle(void* handle) {
  std::lock_guard<std::mutex> g(g_plugin_lock);
  for (const auto& p : g_plugins)
    if (reinterpret_cast<void*>(p.get()) == handle) return p.get();
  return nullptr;
}

// Loader tracing is a debugging aid, not part of running a title. It used to
// print unconditionally, and the engine's plugin discovery calls dlsym
// thousands of times while loading: the formatted writes to a redirected
// stderr were a visible part of the first-frame stalls.
bool TraceDynamic() {
  static const bool on = getenv("ARC_TRACE_DL") != nullptr;
  return on;
}

void* Dlopen(const char* name, int) {
  if (TraceDynamic()) fprintf(stderr, "[dlopen] %s\n", name ? name : "(null)");
  if (!name) return nullptr;
  std::string base(name);
  const size_t slash = base.find_last_of("/\\");
  if (slash != std::string::npos) base = base.substr(slash + 1);
  if (base.find("AkSoundEngine") != std::string::npos)
    return reinterpret_cast<void*>(2);
  if (PluginStub* plugin = FindPlugin(base)) return plugin;
  {
    std::lock_guard<std::mutex> g(g_images_lock);
    for (const ElfImage* img : g_images)
      if (img->name() == base) return const_cast<ElfImage*>(img);
  }
  static const char* const kSystem[] = {
      "libEGL.so",      "libGLESv1_CM.so", "libGLESv2.so",
      "libandroid.so",  "liblog.so",       "libmediandk.so",
      "libOpenSLES.so", "libz.so",         "libc.so",
      "libm.so",        "libdl.so",        "libstdc++.so",
  };
  for (const char* s : kSystem)
    if (base == s) return reinterpret_cast<void*>(1);
  return nullptr;
}
int Dlclose(void*) { return 0; }
const char* Dlerror() { return nullptr; }

void* Dlsym(void* handle, const char* symbol) {
  if (TraceDynamic())
    fprintf(stderr, "[dlsym] request %s\n", symbol ? symbol : "(null)");
  // The AkSoundEngine handle is a sentinel, not an image: every name asked
  // of it belongs to the Wwise Unity integration, and the integration's
  // exports are not all prefixed the same way (CSharp_*, Ak*, ...). The
  // handle is the reliable part. A port that registers the library gets to
  // answer each symbol itself -- the integration returns strings, struct
  // handles and status codes through the same signature, and one answer for
  // all of them is wrong for at least one of them.
  if (handle == reinterpret_cast<void*>(2) && symbol) {
    PluginStub* plugin = FindPlugin("libAkSoundEngine.so");
    uint64_t (*stub)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
                     uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
                     uint64_t, uint64_t) = &AkSoundStub;
    if (plugin) {
      if (ArcPluginFn fn = PluginAnswer(plugin, symbol))
        stub = fn;
    } else if (strcmp(symbol, "CSharp_IsInitialized") == 0) {
      stub = &AkSoundNotInitialized;
    }
    void* answer = reinterpret_cast<void*>(
        ShimHandOut(reinterpret_cast<uint64_t>(stub), symbol));
    if (TraceDynamic())
      fprintf(stderr, "[dlsym] %s -> Ak stub %p\n", symbol, answer);
    return answer;
  }
  if (PluginStub* plugin = PluginForHandle(handle)) {
    ArcPluginFn fn = PluginAnswer(plugin, symbol);
    if (!fn) {
      if (TraceDynamic())
        fprintf(stderr, "[dlsym] %s -> no plugin stub\n",
                symbol ? symbol : "?");
      return nullptr;
    }
    void* answer = reinterpret_cast<void*>(
        ShimHandOut(reinterpret_cast<uint64_t>(fn), symbol));
    if (TraceDynamic())
      fprintf(stderr, "[dlsym] %s -> %s stub %p\n", symbol ? symbol : "?",
              plugin->name.c_str(), answer);
    return answer;
  }
  {
    std::lock_guard<std::mutex> g(g_images_lock);
    // A guest image's own symbol is guest code, and is deliberately *not*
    // announced as a native: the dispatcher would then call ARM instructions
    // as if they were host ones.
    for (const ElfImage* img : g_images)
      if (uint64_t a = img->Lookup(symbol)) {
        if (TraceDynamic())
          fprintf(stderr, "[dlsym] %s -> %#llx (%s)\n", symbol,
                  (unsigned long long)a, img->name().c_str());
        return reinterpret_cast<void*>(a);
      }
  }
  // The shim's answer is host code, and the guest will branch straight to it.
  void* answer = reinterpret_cast<void*>(
      ShimHandOut(ShimResolve(symbol), symbol));
  if (TraceDynamic())
    fprintf(stderr, "[dlsym] %s -> shim %p\n", symbol, answer);
  return answer;
}

// The C++ unwinder walks this to find each image's .eh_frame, so exceptions
// depend on it reporting the images we mapped. Bionic's dl_phdr_info leads
// with the fields the unwinder reads.
struct GuestPhdrInfo {
  uint64_t dlpi_addr;
  const char* dlpi_name;
  const void* dlpi_phdr;
  uint16_t dlpi_phnum;
};

// The visitor is guest code -- this is how the C++ unwinder finds each image's
// .eh_frame -- so it is dispatched, not called.
int DlIteratePhdr(uint64_t callback, void* data) {
  std::vector<const ElfImage*> images;
  {
    std::lock_guard<std::mutex> g(g_images_lock);
    images = g_images;
  }
  for (const ElfImage* img : images) {
    GuestPhdrInfo info{};
    info.dlpi_addr = reinterpret_cast<uint64_t>(img->base());
    info.dlpi_name = "";
    info.dlpi_phdr = img->phdrs();
    info.dlpi_phnum = static_cast<uint16_t>(img->phnum());
    const uint64_t args[3] = {reinterpret_cast<uint64_t>(&info),
                              sizeof(info),
                              reinterpret_cast<uint64_t>(data)};
    const int rc = static_cast<int32_t>(
        arc_call_guest(callback, args, 3) & 0xFFFFFFFFu);
    if (rc != 0) return rc;
  }
  return 0;
}

// --- process and system ----------------------------------------------------

int Getpid() {
#if defined(_WIN32)
  return static_cast<int>(GetCurrentProcessId());
#else
  return ::getpid();
#endif
}
int Gettid() {
#if defined(_WIN32)
  return static_cast<int>(GetCurrentThreadId());
#else
  return static_cast<int>(::syscall(186));
#endif
}
int Geteuid() { return 0; }

// Bionic's struct passwd, laid out the way the guest reads it: two pointers,
// the numeric ids packed into the next eight bytes, then three more pointers.
// Returning null -- which is what a Windows host would do for a user that does
// not exist -- is not a harmless answer here. The engine takes the result and
// reads pw_dir at offset 0x20 without checking it, which turns a missing
// account into a fault in the middle of startup.
struct GuestPasswd {
  char* pw_name;
  char* pw_passwd;
  uint32_t pw_uid;
  uint32_t pw_gid;
  char* pw_gecos;
  char* pw_dir;
  char* pw_shell;
};

GuestPasswd& PasswdEntry() {
  static char name[] = "android";
  static char password[] = "*";
  static char gecos[] = "Android";
  static char home[] = "/";
  static char shell[] = "/system/bin/sh";
  static GuestPasswd entry = {name, password, 0, 0, gecos, home, shell};
  return entry;
}

void* Getpwuid(uint32_t) { return &PasswdEntry(); }
void* Getpwnam(const char* name) {
  if (!name || strcmp(name, "android") == 0 || strcmp(name, "root") == 0 ||
      strcmp(name, "shell") == 0)
    return &PasswdEntry();
  return nullptr;
}
int Getpwuid_r(uint32_t, void* pwd, char*, uint64_t length, void** result) {
  if (result) {
    if (pwd && length >= sizeof(GuestPasswd)) {
      memcpy(pwd, &PasswdEntry(), sizeof(GuestPasswd));
      *result = pwd;
    } else {
      *result = &PasswdEntry();
    }
  }
  return 0;
}
// How many cores to admit to.
//
// The number is not cosmetic: an engine sizes its worker pools from it, so it
// decides how much of the asset pipeline runs at once. A shipped Android
// binary has whatever races a shipped Android binary has, and more workers
// means losing them more often -- so this is the first knob to reach for when
// a fault looks like one thread using what another just freed. ARC_CORES=1
// serialises the pools, which is slower and answers the question.
int64_t CoreCount() {
  static const int64_t cores = [] {
    const char* how = getenv("ARC_CORES");
    const int64_t asked = how ? strtoll(how, nullptr, 10) : 0;
    return asked > 0 ? asked : 8;
  }();
  return cores;
}

int64_t Sysconf(int name) {
  // _SC_PAGESIZE (39) and _SC_NPROCESSORS_ONLN (97) on Bionic.
  if (name == 39) return 4096;
  if (name == 97) return CoreCount();
  // _SC_NPROCESSORS_CONF (96) is the same question about cores that exist
  // rather than cores that are online, and a pool sized from it would ignore
  // the knob above.
  if (name == 96) return CoreCount();
  return -1;
}
// Zero for everything, and AT_HWCAP is the one to leave that way on purpose.
// OpenSSL's aarch64 code reads it to decide whether to use the ARMv8 crypto
// extensions, and those routines are hand-written `aese`/`aesmc`/`pmull`
// assembly the lifter cannot emit. Advertising the bits would send the TLS
// stack straight into a function that does not exist; reporting none leaves it
// on the NEON and C paths, which lift.
uint64_t Getauxval(uint64_t) { return 0; }
// syscall(2). Almost everything that reaches here is something a host cannot
// answer, and -1 is the honest reply -- but not for getrandom. A TLS stack
// that cannot find getrandom as a symbol calls it by number instead, and -1
// there leaves the random generator unseeded, which fails SSL_connect before
// it writes anything: the connection opens, nothing is ever sent, and the
// caller reports that it has no internet.
int64_t Syscall(int64_t number, ...) {
  if (getenv("ARC_TRACE_SYSCALL")) {
    Arm64Ctx* c = arc_current_guest_context();
    fprintf(stderr,
            "[syscall] n=%lld a1=%#llx a2=%#llx a3=%#llx a4=%#llx "
            "lr=%#llx sp=%#llx\n",
            static_cast<long long>(number),
            c ? (unsigned long long)c->x[1] : 0,
            c ? (unsigned long long)c->x[2] : 0,
            c ? (unsigned long long)c->x[3] : 0,
            c ? (unsigned long long)c->x[4] : 0,
            c ? (unsigned long long)c->x[30] : 0,
            c ? (unsigned long long)c->sp : 0);
  }
  // The guest reaches this from its PLT: `bl syscall` with the number in x0
  // and the remaining arguments following the normal AArch64 C convention.
  // The native thunk always forwards twelve integer arguments, so reading them
  // here is both possible and complete. Marshal them back into a context and
  // let the same kernel-side implementation that answers `svc #0` decode the
  // call. That matters for more than code reuse: a futex wait must actually
  // block, and the direct-svc path is the one that knows how.
  va_list ap;
  va_start(ap, number);
  Arm64Ctx context = {};
  for (int i = 0; i < 8; ++i)
    context.x[i] = va_arg(ap, uint64_t);
  va_end(ap);
  context.x[8] = static_cast<uint64_t>(number);
  const int64_t result = arc_svc(&context);
  // Bionic's public syscall() wrapper converts a raw negative errno into -1
  // and sets errno. The guest calls the wrapper, not the raw instruction, so
  // this is the boundary where that translation belongs.
  if (result < 0 && result >= -4095) {
    errno = static_cast<int>(-result);
    return -1;
  }
  return result;
}
int Uname(void* buf) {
  // Linux struct utsname: six 65-byte fields. Zeroed is a valid answer.
  if (buf) memset(buf, 0, 6 * 65);
  return 0;
}
int Fork() { return -1; }
int Execl(const char*, ...) { return -1; }
int Waitpid(int, int*, int) { return -1; }
int Kill(int, int) { return 0; }
void Openlog(const char*, int, int) {}
void Closelog() {}
// No `syslog` here on purpose. It is variadic, and a variadic guest call
// cannot be marshalled by the fixed-arity thunk: its arguments are spread
// across the general registers, the vector registers and the stack by rules
// the thunk has no way to recover. shim_varargs registers it as a
// context-taking native, which can. A second fixed-arity binding for the same
// name resolved only because the varargs layer happens to be asked first, and
// would have marshalled the call wrongly the day that order changed.
void AndroidSetAbortMessage(const char* msg) {
  fprintf(stderr, "abort: %s\n", msg ? msg : "");
}

// Signal delivery has no desktop counterpart worth reproducing: the host
// process is not going to receive SIGPIPE from a loopback socket, and the
// engine's handlers exist for Android's process lifecycle.
// An alternate signal stack exists so a handler can run after the real stack
// overflows. Nothing here delivers signals, so there is nothing to run and
// nowhere for it to run: succeeding is the honest answer, and refusing would
// send the engine down an error path for a facility it never uses.
int Sigaltstack(const void*, void*) { return 0; }
int Setitimer(int, const void*, void*) { return 0; }
int Prctl(int, ...) { return 0; }

// Enough of a filesystem to answer "is there room". Reporting zero free blocks
// would have a title conclude the device is full before it starts.
struct GuestStatfs {
  uint64_t f_type, f_bsize, f_blocks, f_bfree, f_bavail;
  uint64_t f_files, f_ffree;
  uint64_t f_fsid;
  uint64_t f_namelen, f_frsize, f_flags, f_spare[4];
};
int Statfs(const char*, GuestStatfs* out) {
  if (!out) return -1;
  memset(out, 0, sizeof(*out));
  out->f_bsize = 4096;
  out->f_frsize = 4096;
  out->f_blocks = 64ull << 20;  // 256 GB at 4 KB blocks
  out->f_bfree = 32ull << 20;
  out->f_bavail = 32ull << 20;
  out->f_files = 1u << 20;
  out->f_ffree = 1u << 20;
  out->f_namelen = 255;
  return 0;
}

// epoll is how a Linux program waits on descriptors, and a title's own client
// can be built entirely around it: libcurl's multi interface uses epoll on
// Linux, and a TLS stack waiting there is invisible to select() and poll()
// because it never calls either. A wait that answers "nothing is ready"
// forever is not a benign stub in that case -- the connection is open, the
// client is waiting for the socket to become writable, and it never stops
// waiting.
//
// So this is a real implementation: a set of watched descriptors per instance,
// answered by the same poll that answers poll() and select().
constexpr int kEpollFirst = 0x7000001;
constexpr int kEpollInstances = 32;
constexpr uint32_t kEpollIn = 0x001;
constexpr uint32_t kEpollPri = 0x002;
constexpr uint32_t kEpollOut = 0x004;
constexpr uint32_t kEpollErr = 0x008;
constexpr uint32_t kEpollHup = 0x010;
constexpr uint32_t kEpollRdhup = 0x2000;

// The guest's `struct epoll_event`. On aarch64 it is naturally aligned, so the
// four-byte mask is followed by four bytes of padding and then the eight-byte
// data word -- twelve bytes would be the x86-64 layout, which is packed.
struct GuestEpollEvent {
  uint32_t events;
  uint32_t padding;
  uint64_t data;
};

struct EpollInstance {
  bool open = false;
  std::map<int, GuestEpollEvent> watched;  // by guest descriptor
};

std::mutex g_epoll_lock;
EpollInstance g_epolls[kEpollInstances];

int NewEpoll() {
  std::lock_guard<std::mutex> held(g_epoll_lock);
  for (int i = 0; i < kEpollInstances; ++i) {
    if (g_epolls[i].open) continue;
    g_epolls[i].open = true;
    g_epolls[i].watched.clear();
    return kEpollFirst + i;
  }
  errno = EMFILE;
  return -1;
}

int EpollCreate(int) { return NewEpoll(); }
int EpollCreate1(int) { return NewEpoll(); }

int EpollCtl(int epfd, int op, int fd, void* event) {
  const int index = epfd - kEpollFirst;
  if (index < 0 || index >= kEpollInstances) {
    errno = kGuestEInval;
    return -1;
  }
  std::lock_guard<std::mutex> held(g_epoll_lock);
  EpollInstance& set = g_epolls[index];
  if (!set.open) {
    errno = kGuestEInval;
    return -1;
  }
  if (op == 2) {  // EPOLL_CTL_DEL
    set.watched.erase(fd);
    return 0;
  }
  if (op == 1 || op == 3) {  // ADD, MOD
    GuestEpollEvent entry{};
    if (event) memcpy(&entry, event, sizeof entry);
    entry.padding = 0;
    set.watched[fd] = entry;
    return 0;
  }
  errno = kGuestEInval;
  return -1;
}

int EpollWait(int epfd, void* events, int maxevents, int timeout) {
  const int index = epfd - kEpollFirst;
  if (index < 0 || index >= kEpollInstances || maxevents <= 0 || !events) {
    errno = kGuestEInval;
    return -1;
  }
  // A snapshot: the wait must not hold the lock while it blocks, and another
  // thread adding a descriptor mid-wait is normal.
  std::vector<std::pair<int, GuestEpollEvent>> watched;
  {
    std::lock_guard<std::mutex> held(g_epoll_lock);
    if (!g_epolls[index].open) {
      errno = kGuestEInval;
      return -1;
    }
    watched.assign(g_epolls[index].watched.begin(),
                   g_epolls[index].watched.end());
  }
  if (watched.empty()) {
    if (timeout > 0)
      std::this_thread::sleep_for(std::chrono::milliseconds(timeout));
    return 0;
  }
  std::vector<WSAPOLLFD> host;
  std::vector<GuestEpollEvent> wanted;
  for (const auto& item : watched) {
    // Only descriptors this host can actually poll. A stand-in handle -- an
    // eventfd, a pipe the shim never made -- would fail the whole call, and
    // one unsupported watch is not a reason to refuse to watch the rest.
    if (!ShimHostSocket(item.first)) continue;
    WSAPOLLFD poll{};
    poll.fd = HostSocketFor(item.first);
    if (item.second.events & (kEpollIn | kEpollRdhup)) poll.events |= POLLRDNORM;
    if (item.second.events & kEpollPri) poll.events |= POLLRDBAND;
    if (item.second.events & kEpollOut) poll.events |= POLLWRNORM;
    host.push_back(poll);
    wanted.push_back(item.second);
  }
  if (host.empty()) {
    if (timeout > 0)
      std::this_thread::sleep_for(std::chrono::milliseconds(timeout));
    return 0;
  }
  EnsureWinsock();
  const int rc = ::WSAPoll(host.data(), static_cast<ULONG>(host.size()),
                           timeout);
  if (rc < 0) {
    ShimNetErrno();
    return -1;
  }
  auto* out = static_cast<GuestEpollEvent*>(events);
  int ready = 0;
  for (size_t i = 0; i < host.size() && ready < maxevents; ++i) {
    if (!host[i].revents) continue;
    GuestEpollEvent entry = wanted[i];
    entry.events = 0;
    if (host[i].revents & (POLLRDNORM | POLLRDBAND)) entry.events |= kEpollIn;
    if (host[i].revents & POLLWRNORM) entry.events |= kEpollOut;
    // Errors and hangups are reported whether or not they were asked for.
    if (host[i].revents & POLLERR) entry.events |= kEpollErr;
    if (host[i].revents & POLLHUP) entry.events |= kEpollHup;
    if (host[i].revents & POLLNVAL) entry.events |= kEpollErr;
    out[ready++] = entry;
  }
  if (TracingNet() && (ready || timeout != 0)) {
    KeepErrno keep;
    fprintf(stderr, "[net] epoll_wait n=%zu t=%d -> %d (first %#x)\n",
            host.size(), timeout, ready, ready ? out[0].events : 0u);
  }
  return ready;
}
int Eventfd(unsigned, int) { return 0x7000002; }

int Sigaction(int sig, const void* act, void* old) {
  if (act) {
    const unsigned char* bytes = static_cast<const unsigned char*>(act);
    const uint32_t flags = *reinterpret_cast<const uint32_t*>(bytes);
    const uint64_t handler =
        *reinterpret_cast<const uint64_t*>(bytes + 8);
    arc_signal_set_handler(sig, handler);
    static const bool trace = getenv("ARC_TRACE_SIGNALS") != nullptr;
    if (trace)
      fprintf(stderr, "[sigaction] sig=%d flags=%#x handler=%#llx\n", sig,
              flags, static_cast<unsigned long long>(handler));
  }
  (void)old;
  return 0;
}
int Sigaddset(void*, int) { return 0; }
int Sigdelset(void*, int) { return 0; }
int Sigemptyset(void*) { return 0; }
int Sigfillset(void*) { return 0; }
int Sigprocmask(int, const void*, void*) { return 0; }
// setjmp and longjmp cannot be forwarded to the host's, for the same reason a
// guest callback cannot be called as a host function: the state they capture
// belongs to the wrong machine. The host's would save host registers and a
// host frame, while the guest expects its own callee-saved registers and stack
// pointer to come back -- and the Microsoft build takes a second, undeclared
// frame argument that our thunk has no way to supply, so it was being handed
// whatever the guest happened to leave in that register.
//
// Nor can a faithful one live here. A host setjmp called inside this function
// would capture *this* frame, which is gone the moment it returns; jumping to
// it later is undefined. Doing it properly means the lifter emitting the
// setjmp inline in the calling function, so the frame it captures is the one
// that will still be there.
//
// The lifter does it properly now: it emits the setjmp inline in the lifted
// function that calls it, so the frame captured is the caller's -- the one
// that has to still be live when the jump lands. These remain only for a call
// made through a function pointer, which no engine seen so far does, and they
// cannot arm anything for the reason above.
//
// This file's earlier answer -- succeed, and report that no jump has happened
// -- was reasoned from the shape of the code rather than measured, on the
// argument that a decoder arms an error handler and then succeeds. It does
// not: Family Guy takes the failure path and calls longjmp for real.
int Setjmp(void*) { return 0; }

[[noreturn]] void Longjmp(void*, int value) {
  char msg[96];
  snprintf(msg, sizeof(msg),
           "the guest called longjmp(%d); no jump target was recorded", value);
  arc_trap(nullptr, msg);
  abort();  // only reached with no recovery point armed
}

[[noreturn]] void Siglongjmp(void* buf, int value) { Longjmp(buf, value); }
// Bionic's fortified fd_set macros. A program built with _FORTIFY_SOURCE -- as
// every Android shared library is -- does not touch the mask itself: it calls
// `__FD_SET_chk(fd, set, sizeof set)` and `__FD_ISSET_chk(fd, set, sizeof set)`
// instead, and the libc does the arithmetic after checking the descriptor
// against the set's size.
//
// So an empty stand-in for these is not a harmless lie: the caller believes the
// descriptor was added, `select` is given an empty set, and the wait is for
// something that was never asked about. That is precisely how a TLS client
// connects successfully, waits for its own socket to become writable, and waits
// until it times out -- with the connection open the whole time. A Bionic
// fd_set is a 1024-bit mask, indexed by descriptor, on every architecture.
void FdSetChk(int fd, void* set, uint64_t size) {
  if (!set || fd < 0 || static_cast<uint64_t>(fd) >= size * 8) return;
  auto* bits = static_cast<uint64_t*>(set);
  bits[fd / 64] |= uint64_t{1} << (fd % 64);
}
int FdIssetChk(int fd, const void* set, uint64_t size) {
  if (!set || fd < 0 || static_cast<uint64_t>(fd) >= size * 8) return 0;
  const auto* bits = static_cast<const uint64_t*>(set);
  return (bits[fd / 64] >> (fd % 64)) & 1 ? 1 : 0;
}
void FdClrChk(int fd, void* set, uint64_t size) {
  if (!set || fd < 0 || static_cast<uint64_t>(fd) >= size * 8) return;
  auto* bits = static_cast<uint64_t*>(set);
  bits[fd / 64] &= ~(uint64_t{1} << (fd % 64));
}
void FdZeroChk(void* set, uint64_t size) {
  if (set) memset(set, 0, static_cast<size_t>(size));
}

// POSIX regex and fts. Nothing on a game's hot path uses these; they exist so
// the engine's rarely-taken diagnostic paths link.
int Regcomp(void*, const char*, int) { return -1; }
int Regexec(const void*, const char*, uint64_t, void*, int) { return 1; }
uint64_t Regerror(int, const void*, char* buf, uint64_t n) {
  if (buf && n) buf[0] = '\0';
  return 0;
}
void Regfree(void*) {}
void* FtsOpen(char* const*, int, void*) { return nullptr; }
void* FtsRead(void*) { return nullptr; }
int FtsClose(void*) { return 0; }
int Fnmatch(const char*, const char*, int) { return 1; }

long Random() { return rand(); }
void Srand48(long seed) { srand(static_cast<unsigned>(seed)); }

// --- libjnigraphics --------------------------------------------------------
// A bitmap on the Java side. Nothing hands us one on desktop, so these fail
// cleanly rather than pretending a lock succeeded and returning a wild pointer.
int AndroidBitmapGetInfo(void*, void*, void*) { return -1; }
int AndroidBitmapLockPixels(void*, void*, void** pixels) {
  if (pixels) *pixels = nullptr;
  return -1;
}
int AndroidBitmapUnlockPixels(void*, void*) { return -1; }

// --- libc tails -------------------------------------------------------------
// Everything below is a POSIX spelling the Windows CRT does not have, or an
// Android system-service entry point with no desktop equivalent. They are
// deliberately small: a wrong answer here would be worse than a clean failure,
// so each is either an exact forward or an admission that the facility is
// absent.
int SystemPropertyGet(const char* key, char* value) {
  (void)key;
  if (value) value[0] = '\0';
  return 0;
}
int Setenv(const char* name, const char* value, int overwrite) {
  if (!name) return -1;
#if defined(_WIN32)
  if (!overwrite && getenv(name)) return 0;
  return _putenv_s(name, value ? value : "");
#else
  return setenv(name, value ? value : "", overwrite);
#endif
}
int Unsetenv(const char* name) {
  if (!name) return -1;
#if defined(_WIN32)
  return _putenv_s(name, "");
#else
  return unsetenv(name);
#endif
}
struct GuestDlInfo {
  const char* fname;
  void* fbase;
  const char* sname;
  void* saddr;
};
int Dladdr(const void*, GuestDlInfo* info) {
  if (info) memset(info, 0, sizeof(*info));
  return 0;  // no name, which is what a statically linked host can honestly say
}
int PthreadAtfork(void*, void*, void*) { return 0; }
int PthreadSigmask(int, const void*, void*) { return 0; }
unsigned Getegid() { return 0; }
uint32_t InetAddr(const char* text) {
  struct in_addr a{};
  if (!text || ::inet_pton(AF_INET, text, &a) != 1) return 0xffffffffu;
  return a.s_addr;
}
char* InetNtoa(struct in_addr address) {
  static char text[INET_ADDRSTRLEN];
  return ::inet_ntop(AF_INET, &address, text, sizeof text) ? text : nullptr;
}
size_t Strlcpy(char* dst, const char* src, size_t size) {
  const size_t len = src ? strlen(src) : 0;
  if (dst && size) {
    const size_t n = len < size - 1 ? len : size - 1;
    if (n) memcpy(dst, src, n);
    dst[n] = '\0';
  }
  return len;
}
long Lrand48() { return rand(); }
int ClockGetres(int, void* resolution) {
  if (resolution) memset(resolution, 0, sizeof(uint64_t) * 2);
  return 0;
}
int Setpriority(int, int, int) { return 0; }
int Dup(int fd) {
#if defined(_WIN32)
  return _dup(fd);
#else
  return dup(fd);
#endif
}
int Msync(void* address, size_t length, int) {
#if defined(_WIN32)
  return FlushViewOfFile(address, length) ? 0 : -1;
#else
  return msync(address, length, MS_SYNC);
#endif
}
long long Getline(char** line, size_t* capacity, FILE* stream) {
  if (!line || !capacity || !stream) return -1;
  if (!*line || *capacity == 0) {
    *capacity = 256;
    *line = static_cast<char*>(malloc(*capacity));
    if (!*line) {
      *capacity = 0;
      return -1;
    }
  }
  size_t used = 0;
  int ch = EOF;
  while ((ch = fgetc(stream)) != EOF) {
    if (used + 1 >= *capacity) {
      const size_t grow = *capacity * 2;
      char* next = static_cast<char*>(realloc(*line, grow));
      if (!next) return -1;
      *line = next;
      *capacity = grow;
    }
    (*line)[used++] = static_cast<char>(ch);
    if (ch == '\n') break;
  }
  if (used == 0 && ch == EOF) return -1;
  (*line)[used] = '\0';
  return static_cast<long long>(used);
}
FILE* Popen(const char* command, const char* mode) {
#if defined(_WIN32)
  return _popen(command, mode);
#else
  return popen(command, mode);
#endif
}
int Pclose(FILE* stream) {
#if defined(_WIN32)
  return _pclose(stream);
#else
  return pclose(stream);
#endif
}
void* MemsetChk(void* dst, int c, size_t n, size_t) { return memset(dst, c, n); }
int VsprintfChk(char* dst, int, size_t size, const char* format, va_list ap) {
  return vsnprintf(dst, size, format, ap);
}
char* StrncpyChk(char* dst, const char* src, size_t n, size_t) {
  return strncpy(dst, src, n);
}
char* StrncpyChk2(char* dst, const char* src, size_t n, size_t, size_t) {
  return strncpy(dst, src, n);
}
// Bionic's pthread_attr_t (bits/pthread_types.h, LP64):
//   uint32 flags; void* stack_base; size_t stack_size; size_t guard_size;
//   int32 sched_policy; int32 sched_priority; char reserved[16];
// pthread_getattr_np followed by pthread_attr_getstack is how il2cpp finds the
// main thread's stack bottom. Returning success while leaving the attribute
// zeroed makes it conclude the bottom is absurd and call abort().
struct GuestPthreadAttr {
  uint32_t flags;
  uint32_t pad;
  void* stack_base;
  size_t stack_size;
  size_t guard_size;
  int32_t sched_policy;
  int32_t sched_priority;
  char reserved[16];
};
static_assert(sizeof(GuestPthreadAttr) == 56, "Bionic pthread_attr_t size");

void CurrentStackBounds(void** base, size_t* size) {
  void* low = nullptr;
  size_t span = 0;
#if defined(_WIN32)
  ULONG_PTR lo = 0;
  ULONG_PTR hi = 0;
  GetCurrentThreadStackLimits(&lo, &hi);
  low = reinterpret_cast<void*>(lo);
  span = static_cast<size_t>(hi - lo);
#else
  pthread_attr_t attr;
  if (pthread_getattr_np(pthread_self(), &attr) == 0) {
    pthread_attr_getstack(&attr, &low, &span);
    pthread_attr_destroy(&attr);
  }
#endif
  if (base) *base = low;
  if (size) *size = span;
}

int PthreadGetattrNp(void*, void* attr) {
  if (!attr) return 22;
  auto* out = static_cast<GuestPthreadAttr*>(attr);
  memset(out, 0, sizeof(*out));
  CurrentStackBounds(&out->stack_base, &out->stack_size);
  out->guard_size = 4096;
  return 0;
}
int PthreadAttrGetstack(const void* attr, void** base, size_t* size) {
  if (!attr) return 22;
  const auto* in = static_cast<const GuestPthreadAttr*>(attr);
  if (base) *base = in->stack_base;
  if (size) *size = in->stack_size;
  return 0;
}
int PthreadAttrSetstack(void* attr, void* base, size_t size) {
  if (!attr) return 22;
  auto* out = static_cast<GuestPthreadAttr*>(attr);
  out->stack_base = base;
  out->stack_size = size;
  return 0;
}
int Futimens(int fd, const void*) {
#if defined(_WIN32)
  return _commit(fd);
#else
  return 0;
#endif
}
int Utime(const char*, const void*) { return 0; }
int SemGetvalue(void*, int* value) {
  if (value) *value = 0;
  return 0;
}
int SemTimedwait(void*, const void*) { return -1; }
// A positional read must not disturb the descriptor's own offset, and the only
// way to do that here is seek, read, seek back. That sequence is not atomic, so
// two threads reading the same descriptor at once can each land on the other's
// offset and read the wrong bytes -- which on a title that streams a multi-
// gigabyte asset store from a shared descriptor shows up much later as a
// corrupted record, not as a short read. Serialising per descriptor keeps the
// sequence whole without moving every read onto its own handle.
std::mutex g_positional_lock[64];

std::mutex& PositionalLock(int fd) {
  return g_positional_lock[static_cast<unsigned>(fd) % 64];
}

long long Pread(int fd, void* buffer, size_t count, long long offset) {
#if defined(_WIN32)
  std::lock_guard<std::mutex> held(PositionalLock(fd));
  const __int64 old = _lseeki64(fd, 0, SEEK_CUR);
  if (old < 0 || _lseeki64(fd, offset, SEEK_SET) < 0) return -1;
  const int n = _read(fd, buffer, static_cast<unsigned>(count));
  _lseeki64(fd, old, SEEK_SET);
  return n;
#else
  return pread(fd, buffer, count, offset);
#endif
}
long long Pwrite(int fd, const void* buffer, size_t count, long long offset) {
#if defined(_WIN32)
  std::lock_guard<std::mutex> held(PositionalLock(fd));
  const __int64 old = _lseeki64(fd, 0, SEEK_CUR);
  if (old < 0 || _lseeki64(fd, offset, SEEK_SET) < 0) return -1;
  const int n = _write(fd, buffer, static_cast<unsigned>(count));
  _lseeki64(fd, old, SEEK_SET);
  return n;
#else
  return pwrite(fd, buffer, count, offset);
#endif
}
int Getuid() { return 0; }
int Getrusage(int, void* usage) {
  // Bionic's struct rusage is two timevals followed by fourteen longs. The
  // engine only ever asks for a zeroed baseline here; leaving it untouched
  // would hand back uninitialised stack words instead of "no time used".
  if (usage) memset(usage, 0, 16 * 2 + 8 * 14);
  return 0;
}
int InetAton(const char* text, void* out) {
  unsigned a = 0, b = 0, c = 0, d = 0;
  char tail = 0;
  if (!text || !out) return 0;
  if (sscanf(text, "%u.%u.%u.%u%c", &a, &b, &c, &d, &tail) != 4) return 0;
  if (a > 255 || b > 255 || c > 255 || d > 255) return 0;
  auto* p = static_cast<unsigned char*>(out);
  p[0] = static_cast<unsigned char>(a);
  p[1] = static_cast<unsigned char>(b);
  p[2] = static_cast<unsigned char>(c);
  p[3] = static_cast<unsigned char>(d);
  return 1;
}
// The guest's iovec is pointer-then-length, the same shape as on Linux. The
// host socket API is not vectorised at this layer, so a loop is the honest
// translation; the short-write rule is the same one writev has always had.
struct GuestIovec {
  void* base;
  size_t len;
};
long long Writev(int fd, const void* vector, int count) {
  if (!vector || count <= 0) return 0;
  const auto* iov = static_cast<const GuestIovec*>(vector);
  long long total = 0;
  for (int i = 0; i < count; ++i) {
    long long n = Send(fd, iov[i].base, iov[i].len, 0);
#if defined(_WIN32)
    if (n < 0) n = _write(fd, iov[i].base, static_cast<unsigned>(iov[i].len));
#endif
    if (n < 0) return total ? total : -1;
    total += n;
    if (static_cast<uint64_t>(n) < iov[i].len) break;
  }
  return total;
}
long long Readv(int fd, const void* vector, int count) {
  if (!vector || count <= 0) return 0;
  const auto* iov = static_cast<const GuestIovec*>(vector);
  long long total = 0;
  for (int i = 0; i < count; ++i) {
    long long n = Recv(fd, iov[i].base, iov[i].len, 0);
#if defined(_WIN32)
    if (n < 0) n = _read(fd, iov[i].base, static_cast<unsigned>(iov[i].len));
#endif
    if (n < 0) return total ? total : -1;
    total += n;
    if (static_cast<uint64_t>(n) < iov[i].len) break;
  }
  return total;
}
int PthreadGetschedparam(void*, int* policy, void* param) {
  if (policy) *policy = 0;
  if (param) memset(param, 0, sizeof(int));
  return 0;
}
int SchedGetPriorityMax(int) { return 0; }
int SchedGetPriorityMin(int) { return 0; }
void Tzset() {
#if defined(_WIN32)
  _tzset();
#else
  tzset();
#endif
}
int IfIndextoname(unsigned, char*) { return 0; }
void* Getservbyname(const char*, const char*) { return nullptr; }
int* GetHErrno() {
  static thread_local int err = 0;
  return &err;
}
void* CmsgNxthdr(void*, void*) { return nullptr; }
int Execvp(const char*, char* const*) { return -1; }
int Tcgetattr(int, void*) { return -1; }
int Tcsetattr(int, int, const void*) { return 0; }
void GcovDump() {}
void GcovFlush() {}
// Abseil's cordz sampler has a thread-local initialisation wrapper. There is no
// cordz sampler in this port, so the wrapper has no work to do; resolving it
// keeps the PLT from falling through to its resolver.
void AbseilCordzTlsInit() {}
void DeferredActionTlsInit() {}
const void* SystemPropertyFind(const char*) { return nullptr; }
int SystemPropertyRead(const void*, char*, char*) { return 0; }
void* Gethostbyaddr(const void*, unsigned, int) { return nullptr; }
int AndroidLogBufWrite(int, int, const char*, const char*) { return 0; }
long double Ldexpl(long double value, int exponent) {
  return ldexp(static_cast<double>(value), exponent);
}
long double Frexpl(long double value, int* exponent) {
  return frexp(static_cast<double>(value), exponent);
}
long long Sendmmsg(int, const void*, unsigned, int) { return -1; }
long long Recvmmsg(int, void*, unsigned, int, void*) { return -1; }
long long Sendmsg(int, const void*, int) { return -1; }
long long Recvmsg(int, void*, int) { return -1; }
int Sigsuspend(const void*) {
  // The collector's handler calls this to wait for the stop count to advance.
  // The host has no signal frame to return from, so a short sleep is the
  // equivalent of blocking: it keeps the suspended thread off the CPU while
  // still noticing a resume promptly. A pending signal is consumed here so the
  // handler is not re-entered.
  arc_set_wait_reason("sigsuspend");
  if (arc_signal_pending_current()) arc_signal_take_current();
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  arc_set_wait_reason(nullptr);
  errno = 4 /* EINTR */;
  return -1;
}
int Execve(const char*, char* const*, char* const*) { return -1; }
void PthreadCleanupPush(void*, void*) {}
void PthreadCleanupPop(void*) {}

// --- Android sensors and loopers -------------------------------------------
//
// Unity asks the platform whether a device has an accelerometer/gyroscope while
// it builds the display and input managers. There is no sensor on a desktop,
// and no Android looper either, so the honest answer is a manager that exists
// but reports an empty sensor list. The earlier behaviour -- a guard-page trap
// on ASensorManager_getInstance -- stopped startup before the renderer ever
// reached its first frame.
int g_sensor_manager_token;
int g_sensor_queue_token;

void* SensorManagerGetInstance() { return &g_sensor_manager_token; }
void* SensorManagerGetInstanceForPackage(const char*) {
  return &g_sensor_manager_token;
}
int SensorManagerGetSensorList(void*, const void** list) {
  if (list) *list = nullptr;
  return 0;
}
void* SensorManagerGetDefaultSensor(void*, int) { return nullptr; }
void* SensorManagerGetDefaultSensorEx(void*, int, int) { return nullptr; }
void* SensorManagerCreateEventQueue(void*, void*, int, void*, void*) {
  return &g_sensor_queue_token;
}
int SensorManagerDestroyEventQueue(void*, void*) { return 0; }
int SensorEventQueueEnableSensor(void*, void*) { return 0; }
int SensorEventQueueDisableSensor(void*, void*) { return 0; }
int SensorEventQueueSetEventRate(void*, void*, int) { return 0; }
int SensorEventQueueHasEvents(void*) { return 0; }
long long SensorEventQueueGetEvents(void*, void*, unsigned long long) {
  return 0;
}
int SensorGetType(const void*) { return 0; }
float SensorGetResolution(const void*) { return 0.0f; }
int SensorGetMinDelay(const void*) { return 0; }
const char* SensorGetName(const void*) { return ""; }
const char* SensorGetVendor(const void*) { return "androidrecomp"; }
const char* SensorGetStringType(const void*) { return "android.sensor.unknown"; }

int g_looper_token;
thread_local int t_looper_token;
void* LooperForThread() { return &t_looper_token; }
void* LooperPrepare(int) { return &t_looper_token; }
void LooperAcquire(void*) {}
void LooperRelease(void*) {}
void LooperWake(void*) {}
int LooperPollOnce(int, int, int* out_fd, int* out_events, void** out_data) {
  if (out_fd) *out_fd = -1;
  if (out_events) *out_events = 0;
  if (out_data) *out_data = nullptr;
  return -3;  // ALOOPER_POLL_TIMEOUT
}
int LooperPollAll(int timeout, int* out_fd, int* out_events, void** out_data) {
  return LooperPollOnce(timeout, 0, out_fd, out_events, out_data);
}
int LooperAddFd(void*, int, int, int, void*, void*) {
  return 1;  // ALOOPER_POLL_CALLBACK
}
int LooperRemoveFd(void*, int) { return 0; }

struct Entry {
  const char* name;
  void* fn;
};

#define E(sym, fn) {sym, reinterpret_cast<void*>(&fn)}
const Entry kTable[] = {
    E("socket", Socket),            E("bind", Bind),
    E("connect", Connect),          E("listen", Listen),
    E("accept", Accept),            E("send", Send),
    E("recv", Recv),                E("sendto", Sendto),
    E("recvfrom", Recvfrom),        E("shutdown", Shutdown),
    E("setsockopt", Setsockopt),    E("getsockopt", Getsockopt),
    E("getsockname", Getsockname),  E("getpeername", Getpeername),
    E("gethostname", Gethostname),  E("select", Select),
    E("inet_ntop", InetNtop),       E("inet_pton", InetPton),
    E("getaddrinfo", Getaddrinfo),  E("freeaddrinfo", Freeaddrinfo),
    E("gai_strerror", GaiStrerror), E("getnameinfo", Getnameinfo),
    E("gethostbyname", Gethostbyname),
    E("if_nametoindex", IfNametoindex),
    E("socketpair", Socketpair),    E("sendfile", Sendfile),
    E("poll", Poll),                E("ioctl", Ioctl),

    E("dlopen", Dlopen),            E("dlsym", Dlsym),
    E("dlclose", Dlclose),          E("dlerror", Dlerror),
    E("dl_iterate_phdr", DlIteratePhdr),

    E("getpid", Getpid),            E("gettid", Gettid),
    E("getuid", Getuid),            E("geteuid", Geteuid),
    E("getpwuid", Getpwuid),        E("getpwuid_r", Getpwuid_r),
    E("getpwnam", Getpwnam),
    E("getrusage", Getrusage),
    E("sysconf", Sysconf),          E("getauxval", Getauxval),
    E("syscall", Syscall),          E("uname", Uname),
    E("fork", Fork),                E("execl", Execl),
    E("waitpid", Waitpid),          E("kill", Kill),
    E("openlog", Openlog),          E("closelog", Closelog),
    E("android_set_abort_message", AndroidSetAbortMessage),

    E("sigaltstack", Sigaltstack),  E("setitimer", Setitimer),
    E("prctl", Prctl),              E("statfs", Statfs),
    E("epoll_create", EpollCreate), E("epoll_create1", EpollCreate1),
    E("epoll_ctl", EpollCtl),       E("epoll_wait", EpollWait),
    E("epoll_pwait", EpollWait),
    E("eventfd", Eventfd),

    E("sigaction", Sigaction),      E("sigaddset", Sigaddset),
    E("sigdelset", Sigdelset),      E("sigemptyset", Sigemptyset),
    E("sigfillset", Sigfillset),    E("sigprocmask", Sigprocmask),
    E("setjmp", Setjmp),           E("longjmp", Longjmp),
    E("siglongjmp", Siglongjmp),
    E("__FD_SET_chk", FdSetChk),    E("__FD_ISSET_chk", FdIssetChk),
    E("__FD_CLR_chk", FdClrChk),    E("__FD_ZERO_chk", FdZeroChk),

    E("regcomp", Regcomp),          E("regexec", Regexec),
    E("regerror", Regerror),        E("regfree", Regfree),
    E("fts_open", FtsOpen),         E("fts_read", FtsRead),
    E("fts_close", FtsClose),       E("fnmatch", Fnmatch),
    E("random", Random),            E("srand48", Srand48),

    E("AndroidBitmap_getInfo", AndroidBitmapGetInfo),
    E("AndroidBitmap_lockPixels", AndroidBitmapLockPixels),
    E("AndroidBitmap_unlockPixels", AndroidBitmapUnlockPixels),

    E("ASensorManager_getInstance", SensorManagerGetInstance),
    E("ASensorManager_getInstanceForPackage", SensorManagerGetInstanceForPackage),
    E("ASensorManager_getSensorList", SensorManagerGetSensorList),
    E("ASensorManager_getDefaultSensor", SensorManagerGetDefaultSensor),
    E("ASensorManager_getDefaultSensorEx", SensorManagerGetDefaultSensorEx),
    E("ASensorManager_createEventQueue", SensorManagerCreateEventQueue),
    E("ASensorManager_destroyEventQueue", SensorManagerDestroyEventQueue),
    E("ASensorEventQueue_enableSensor", SensorEventQueueEnableSensor),
    E("ASensorEventQueue_disableSensor", SensorEventQueueDisableSensor),
    E("ASensorEventQueue_setEventRate", SensorEventQueueSetEventRate),
    E("ASensorEventQueue_hasEvents", SensorEventQueueHasEvents),
    E("ASensorEventQueue_getEvents", SensorEventQueueGetEvents),
    E("ASensor_getType", SensorGetType),
    E("ASensor_getResolution", SensorGetResolution),
    E("ASensor_getMinDelay", SensorGetMinDelay),
    E("ASensor_getName", SensorGetName),
    E("ASensor_getVendor", SensorGetVendor),
    E("ASensor_getStringType", SensorGetStringType),
    E("ALooper_forThread", LooperForThread),
    E("ALooper_prepare", LooperPrepare),
    E("ALooper_acquire", LooperAcquire),
    E("ALooper_release", LooperRelease),
    E("ALooper_wake", LooperWake),
    E("ALooper_pollOnce", LooperPollOnce),
    E("ALooper_pollAll", LooperPollAll),
    E("ALooper_addFd", LooperAddFd),
    E("ALooper_removeFd", LooperRemoveFd),

    E("__system_property_get", SystemPropertyGet),
    E("__system_property_find", SystemPropertyFind),
    E("__system_property_read", SystemPropertyRead),
    E("__android_log_buf_write", AndroidLogBufWrite),
    E("gethostbyaddr", Gethostbyaddr),
    E("_ZTH15gDeferredAction", DeferredActionTlsInit),
    E("setenv", Setenv),            E("unsetenv", Unsetenv),
    E("dladdr", Dladdr),            E("pthread_atfork", PthreadAtfork),
    E("pthread_sigmask", PthreadSigmask), E("getegid", Getegid),
    E("inet_addr", InetAddr),       E("inet_ntoa", InetNtoa),
    E("inet_aton", InetAton),       E("getservbyname", Getservbyname),
    E("if_indextoname", IfIndextoname), E("__get_h_errno", GetHErrno),
    E("__cmsg_nxthdr", CmsgNxthdr),
    E("sendmmsg", Sendmmsg),        E("recvmmsg", Recvmmsg),
    E("writev", Writev),            E("readv", Readv),
    E("pthread_getschedparam", PthreadGetschedparam),
    E("sched_get_priority_max", SchedGetPriorityMax),
    E("sched_get_priority_min", SchedGetPriorityMin),
    E("tzset", Tzset),              E("execvp", Execvp),
    E("tcgetattr", Tcgetattr),      E("tcsetattr", Tcsetattr),
    E("__gcov_dump", GcovDump),     E("__gcov_flush", GcovFlush),
    E("_ZTHN4absl12lts_2024011613cord_internal17cordz_next_sampleE",
      AbseilCordzTlsInit),
    E("ldexpl", Ldexpl),            E("frexpl", Frexpl),
    E("strlcpy", Strlcpy),          E("lrand48", Lrand48),
    E("clock_getres", ClockGetres), E("setpriority", Setpriority),
    E("dup", Dup),                  E("msync", Msync),
    E("getline", Getline),          E("popen", Popen),
    E("pclose", Pclose),            E("__memset_chk", MemsetChk),
    E("__vsprintf_chk", VsprintfChk), E("__strncpy_chk", StrncpyChk),
    E("__strncpy_chk2", StrncpyChk2),
    E("pthread_getattr_np", PthreadGetattrNp),
    E("pthread_attr_getstack", PthreadAttrGetstack),
    E("pthread_attr_setstack", PthreadAttrSetstack),
    E("futimens", Futimens),        E("utime", Utime),
    E("sem_getvalue", SemGetvalue), E("sem_timedwait", SemTimedwait),
    E("pread", Pread),              E("pwrite", Pwrite),
    E("sendmsg", Sendmsg),          E("recvmsg", Recvmsg),
    E("sigsuspend", Sigsuspend),    E("execve", Execve),
    E("__pthread_cleanup_push", PthreadCleanupPush),
    E("__pthread_cleanup_pop", PthreadCleanupPop),
};
#undef E

}  // namespace

int64_t HostCoreCount() { return CoreCount(); }

void ShimRegisterPlugin(const char* soname, ArcPluginFn generic) {
  RegisterPlugin(soname, generic);
}

void ShimRegisterPluginSymbol(const char* soname, const char* symbol,
                              ArcPluginFn fn) {
  RegisterPluginSymbol(soname, symbol, fn);
}

void ShimRegisterPluginResolver(const char* soname, ArcPluginResolve resolve) {
  RegisterPluginResolver(soname, resolve);
}

uint64_t ShimGuestSymbol(const char* name) {
  std::lock_guard<std::mutex> g(g_images_lock);
  for (const ElfImage* img : g_images)
    if (uint64_t a = img->Lookup(name)) return a;
  return 0;
}

// The socket descriptor map, for the file layer and for select. Declared in
// shim.h; defined here because this is where the map lives.
uint64_t ShimHostSocket(int fd) {
  std::lock_guard<std::mutex> held(g_socket_map_lock);
  const auto it = g_guest_to_host.find(fd);
  return it == g_guest_to_host.end() ? 0
                                     : static_cast<uint64_t>(it->second);
}

int ShimGuestSocket(uint64_t host) {
  std::lock_guard<std::mutex> held(g_socket_map_lock);
  const auto it = g_host_to_guest.find(static_cast<SOCKET>(host));
  return it == g_host_to_guest.end() ? -1 : it->second;
}

void ShimForgetGuestSocket(int fd) {
  std::lock_guard<std::mutex> held(g_socket_map_lock);
  const auto it = g_guest_to_host.find(fd);
  if (it == g_guest_to_host.end()) return;
  g_host_to_guest.erase(it->second);
  g_guest_to_host.erase(it);
}

void ShimTraceSocketIo(const char* what, int fd, int64_t want, int64_t got,
                       const void* buf) {
  TraceIo(what, fd, want, got);
  TraceBody(what, buf, got);
}

int ShimCloseEpoll(int fd) {
  const int index = fd - kEpollFirst;
  if (index < 0 || index >= kEpollInstances) return -1;
  std::lock_guard<std::mutex> held(g_epoll_lock);
  if (!g_epolls[index].open) return -1;
  g_epolls[index].open = false;
  g_epolls[index].watched.clear();
  return 0;
}


// Translate the last Winsock error into errno, in the numbers the guest's libc
// uses. Returns the value it set, so a caller can test it without a second
// lookup. Shared with the socket paths in shim_file.cpp.
int ShimNetErrno() {
#if defined(_WIN32)
  int e = kGuestEInval;
  switch (::WSAGetLastError()) {
    case WSAEWOULDBLOCK:   e = kGuestEAgain; break;
    case WSAEINPROGRESS:   e = kGuestEInProgress; break;
    case WSAEALREADY:      e = kGuestEAlready; break;
    case WSAEISCONN:       e = kGuestEIsConn; break;
    case WSAECONNREFUSED:  e = kGuestEConnRefused; break;
    case WSAECONNRESET:    e = kGuestEConnReset; break;
    case WSAECONNABORTED:  e = kGuestEPipe; break;
    case WSAETIMEDOUT:     e = kGuestETimedOut; break;
    case WSAENETUNREACH:   e = kGuestENetUnreach; break;
    case WSAEHOSTUNREACH:  e = kGuestEHostUnreach; break;
    case WSAEINTR:         e = kGuestEIntr; break;
    case 0:                e = 0; break;
    default:               break;
  }
  errno = e;
  return e;
#else
  return errno;
#endif
}

void ShimRegisterImage(const ElfImage* image) {
  std::lock_guard<std::mutex> g(g_images_lock);
  g_images.push_back(image);
}

// TEMPORARY probe: attribute the boot-time memcpy storm. Gated by
// ARC_MEMCPY_TRACE; remove once the caller is understood.
static void* ProbeMemcpy(void* dst, const void* src, size_t count) {
  static const bool on = getenv("ARC_MEMCPY_TRACE") != nullptr;
  static unsigned long long calls;
  static unsigned long long bytes;
  const unsigned long long index = calls++;
  bytes += count;
  if (on && (index < 80 || (index % 2000) == 0)) {
    Arm64Ctx* c = arc_current_guest_context();
    fprintf(stderr,
            "[memcpy] n=%llu totalMB=%llu dst=%p src=%p lr=%#llx caller=%#llx "
            "sp=%#llx\n",
            static_cast<unsigned long long>(count),
            static_cast<unsigned long long>(bytes >> 20), dst, src,
            c ? static_cast<unsigned long long>(c->x[30]) : 0,
            c ? static_cast<unsigned long long>(arc_ld64(c->sp)) : 0,
            c ? static_cast<unsigned long long>(c->sp) : 0);
  }
  return memcpy(dst, src, count);
}

uint64_t ShimResolveSys(const char* name) {
  if (strcmp(name, "memcpy") == 0 && getenv("ARC_MEMCPY_TRACE"))
    return reinterpret_cast<uint64_t>(&ProbeMemcpy);
  for (const Entry& e : kTable)
    if (strcmp(e.name, name) == 0) return reinterpret_cast<uint64_t>(e.fn);
  return 0;
}

size_t ShimSysCount() { return sizeof(kTable) / sizeof(kTable[0]); }

}  // namespace arc
