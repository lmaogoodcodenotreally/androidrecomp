// The Android platform surface API, as much of it as a desktop port needs.
//
// Unity's renderer is written against EGL, even when the window belongs to the
// host: it asks for the current display, surface and context, queries the
// surface size, and occasionally creates a pbuffer. On Windows the context is
// SDL's, created with WGL, so the honest answer is a set of stand-ins that
// describe the context that is already current rather than a second EGL
// implementation. A call that would change state -- make current, swap -- is a
// no-op; the host owns the window and presents after `nativeRender`.
//
// ANativeWindow is the same story one level down: a Surface is a Java object
// with no desktop counterpart, so the port hands back an opaque token and
// answers size queries from the frame it actually opened.

#include "shim.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <atomic>

#if defined(ARC_HAVE_SDL2)
#include <SDL.h>
#endif

namespace arc {
namespace {

using EGLBoolean = uint32_t;
using EGLint = int32_t;

constexpr EGLint kEglSuccess = 0x3000;
constexpr EGLint kEglWidth = 0x3057;
constexpr EGLint kEglHeight = 0x3056;
constexpr EGLint kEglConfigId = 0x3028;
constexpr EGLint kEglRedSize = 0x3024;
constexpr EGLint kEglGreenSize = 0x3023;
constexpr EGLint kEglBlueSize = 0x3022;
constexpr EGLint kEglAlphaSize = 0x3021;
constexpr EGLint kEglDepthSize = 0x3025;
constexpr EGLint kEglStencilSize = 0x3026;
constexpr EGLint kEglSurfaceType = 0x3033;
constexpr EGLint kEglRenderableType = 0x3040;
constexpr EGLint kEglOpenGLES2Bit = 0x0004;
constexpr EGLint kEglWindowBit = 0x0004;
constexpr EGLint kEglPbufferBit = 0x0001;

constexpr uintptr_t kDisplay = 1;
constexpr uintptr_t kContext = 2;
constexpr uintptr_t kSurface = 3;
constexpr uintptr_t kNativeWindow = 4;

// The window and context the host created. Registered at `Window::Open`, before
// the engine is loaded, so every thread that asks for a context gets the real
// pair rather than the fake handles the engine passes around.
void* g_window = nullptr;
void* g_context = nullptr;

// Presentation belongs to the engine here. It issues `eglSwapBuffers` on the
// thread that holds the context, which is its own render thread and not the
// host's frame loop; the host cannot take the context back to swap without
// stalling that thread mid-frame, and every attempt fails while it is busy.
// So the guest's swap is the present, and the host only decides whether one is
// allowed -- a covered window must not wait on the compositor.
std::atomic<bool> g_swap_allowed{true};
std::atomic<bool> g_swapped{false};

int ScreenWidth() {
  const char* s = getenv("ARC_SCREEN_WIDTH");
  return s && *s ? atoi(s) : 1280;
}
int ScreenHeight() {
  const char* s = getenv("ARC_SCREEN_HEIGHT");
  return s && *s ? atoi(s) : 720;
}

// Which thread is asking for the context is the whole question here: the
// engine may render on a thread of its own, and a context that is current
// somewhere else is not current there. Every call that hands the context out or
// takes it away says who called and what the driver currently believes.
void TraceEgl(const char* what, const void* argument) {
#if defined(ARC_HAVE_SDL2)
  static const bool on = getenv("ARC_TRACE_EGL") != nullptr;
  if (!on) return;
  fprintf(stderr, "[egl] %-14s arg=%p tid=%lu current=%p\n", what, argument,
          static_cast<unsigned long>(SDL_ThreadID()),
          static_cast<const void*>(SDL_GL_GetCurrentContext()));
#else
  (void)what;
  (void)argument;
#endif
}

EGLint EglGetError() { return kEglSuccess; }
void* EglGetDisplay(void*) { return reinterpret_cast<void*>(kDisplay); }
EGLBoolean EglInitialize(void*, EGLint* major, EGLint* minor) {
  if (major) *major = 1;
  if (minor) *minor = 4;
  return 1;
}
EGLBoolean EglTerminate(void*) { return 1; }
const char* EglQueryString(void*, EGLint name) {
  switch (name) {
    case 0x3053:  // EGL_VENDOR
      return "androidrecomp";
    case 0x3054:  // EGL_VERSION
      return "1.4 androidrecomp";
    case 0x3055:  // EGL_EXTENSIONS
      return "EGL_KHR_surfaceless_context EGL_KHR_gl_colorspace";
    case 0x308D:  // EGL_CLIENT_APIS
      return "OpenGL_ES";
    default:
      return "";
  }
}
EGLBoolean EglChooseConfig(void*, const EGLint*, void** configs,
                           EGLint config_size, EGLint* num_config) {
  if (num_config) *num_config = 1;
  if (configs && config_size > 0) configs[0] = reinterpret_cast<void*>(1);
  return 1;
}
EGLBoolean EglGetConfigAttrib(void*, void*, EGLint attribute, EGLint* value) {
  if (!value) return 1;
  switch (attribute) {
    case kEglRedSize:
    case kEglGreenSize:
    case kEglBlueSize:
      *value = 8;
      break;
    case kEglAlphaSize:
      *value = 8;
      break;
    case kEglDepthSize:
      *value = 24;
      break;
    case kEglStencilSize:
      *value = 8;
      break;
    case kEglConfigId:
      *value = 1;
      break;
    case kEglSurfaceType:
      *value = kEglWindowBit | kEglPbufferBit;
      break;
    case kEglRenderableType:
      *value = kEglOpenGLES2Bit;
      break;
    default:
      *value = 0;
      break;
  }
  return 1;
}
void* EglCreateWindowSurface(void*, void*, void*, const EGLint*) {
  return reinterpret_cast<void*>(kSurface);
}
void* EglCreatePbufferSurface(void*, void*, const EGLint* attribs) {
  (void)attribs;
  return reinterpret_cast<void*>(kSurface);
}
void* EglCreateContext(void*, void*, void*, const EGLint*) {
  TraceEgl("createContext", nullptr);
  return reinterpret_cast<void*>(kContext);
}
EGLBoolean EglDestroyContext(void*, void*) { return 1; }
EGLBoolean EglDestroySurface(void*, void*) { return 1; }
EGLBoolean EglMakeCurrent(void*, void*, void*, void* context) {
  TraceEgl("makeCurrent", context);
#if defined(ARC_HAVE_SDL2)
  // The engine hands the one context between its own threads: a thread makes it
  // current, issues its GL calls, and releases before another thread takes it.
  // The desktop driver enforces exactly that rule, and a no-op here is not
  // harmless -- every call on the engine's render thread would land on no
  // context at all, so `glCreateShader` answers zero, the program links to
  // nothing, and the frame draws a black screen with a full set of calls
  // behind it. Forward the hand-off instead of pretending it happened.
  if (g_window) {
    if (SDL_GL_MakeCurrent(static_cast<SDL_Window*>(g_window),
                           context ? static_cast<SDL_GLContext>(g_context)
                                   : nullptr) != 0) {
      static const bool log = getenv("ARC_TRACE_EGL") != nullptr;
      if (log) fprintf(stderr, "[egl] makeCurrent failed: %s\n", SDL_GetError());
      return 0;
    }
  }
#endif
  return 1;
}
void* EglGetCurrentContext() {
  TraceEgl("getCurrent", nullptr);
  return reinterpret_cast<void*>(kContext);
}
void* EglGetCurrentSurface(EGLint) { return reinterpret_cast<void*>(kSurface); }
EGLBoolean EglQuerySurface(void*, void*, EGLint attribute, EGLint* value) {
  if (!value) return 1;
  switch (attribute) {
    case kEglWidth:
      *value = ScreenWidth();
      break;
    case kEglHeight:
      *value = ScreenHeight();
      break;
    default:
      *value = 0;
      break;
  }
  return 1;
}
EGLBoolean EglSurfaceAttrib(void*, void*, EGLint, EGLint) { return 1; }
EGLBoolean EglSwapBuffers(void*, void*) {
  TraceEgl("swapBuffers", nullptr);
#if defined(ARC_HAVE_SDL2)
  if (!g_window) return 1;
  if (!g_swap_allowed.load(std::memory_order_relaxed)) {
    // Covered: submit the frame without waiting on a compositor that has no
    // pixels to show. Waiting is what turns a hidden window into a
    // multi-second frame and a "Not Responding" title.
    static auto flush = reinterpret_cast<void (*)()>(
        SDL_GL_GetProcAddress("glFlush"));
    if (flush) flush();
    return 1;
  }
  if (SDL_GL_GetCurrentContext() == static_cast<SDL_GLContext>(g_context)) {
    SDL_GL_SwapWindow(static_cast<SDL_Window*>(g_window));
    g_swapped.store(true, std::memory_order_relaxed);
  }
#endif
  return 1;
}
EGLBoolean EglSwapInterval(void*, EGLint) { return 1; }
void* EglGetProcAddress(const char* name) {
#if defined(ARC_HAVE_SDL2)
  if (!name) return nullptr;
  const uint64_t address =
      reinterpret_cast<uint64_t>(SDL_GL_GetProcAddress(name));
  return reinterpret_cast<void*>(ShimHandOut(address, name));
#else
  (void)name;
  return nullptr;
#endif
}

void* NativeWindowFromSurface(void*, void*) {
  return reinterpret_cast<void*>(kNativeWindow);
}
void NativeWindowAcquire(void*) {}
void NativeWindowRelease(void*) {}
int NativeWindowGetWidth(void*) { return ScreenWidth(); }
int NativeWindowGetHeight(void*) { return ScreenHeight(); }
int NativeWindowGetFormat(void*) { return 1; }  // WINDOW_FORMAT_RGBA_8888
int NativeWindowSetBuffersGeometry(void*, int, int, int) { return 0; }
void* NativeWindowToSurface(void*) { return nullptr; }

struct Entry {
  const char* name;
  void* fn;
};
#define E(n, f) {n, reinterpret_cast<void*>(&f)}
const Entry kTable[] = {
    E("eglGetError", EglGetError),
    E("eglGetDisplay", EglGetDisplay),
    E("eglInitialize", EglInitialize),
    E("eglTerminate", EglTerminate),
    E("eglQueryString", EglQueryString),
    E("eglChooseConfig", EglChooseConfig),
    E("eglGetConfigAttrib", EglGetConfigAttrib),
    E("eglCreateWindowSurface", EglCreateWindowSurface),
    E("eglCreatePbufferSurface", EglCreatePbufferSurface),
    E("eglCreateContext", EglCreateContext),
    E("eglDestroyContext", EglDestroyContext),
    E("eglDestroySurface", EglDestroySurface),
    E("eglMakeCurrent", EglMakeCurrent),
    E("eglGetCurrentContext", EglGetCurrentContext),
    E("eglGetCurrentSurface", EglGetCurrentSurface),
    E("eglQuerySurface", EglQuerySurface),
    E("eglSurfaceAttrib", EglSurfaceAttrib),
    E("eglSwapBuffers", EglSwapBuffers),
    E("eglSwapInterval", EglSwapInterval),
    E("eglGetProcAddress", EglGetProcAddress),
    E("ANativeWindow_fromSurface", NativeWindowFromSurface),
    E("ANativeWindow_acquire", NativeWindowAcquire),
    E("ANativeWindow_release", NativeWindowRelease),
    E("ANativeWindow_getWidth", NativeWindowGetWidth),
    E("ANativeWindow_getHeight", NativeWindowGetHeight),
    E("ANativeWindow_getFormat", NativeWindowGetFormat),
    E("ANativeWindow_setBuffersGeometry", NativeWindowSetBuffersGeometry),
    E("ANativeWindow_toSurface", NativeWindowToSurface),
};
#undef E

}  // namespace

void ShimSetGfxWindow(void* window, void* context) {
  g_window = window;
  g_context = context;
}

void ShimGfxAllowSwap(bool allow) { g_swap_allowed.store(allow); }

bool ShimGfxPresents() { return g_swapped.load(std::memory_order_relaxed); }

uint64_t ShimResolveEGL(const char* name) {
  for (const Entry& e : kTable)
    if (strcmp(e.name, name) == 0) return reinterpret_cast<uint64_t>(e.fn);
  return 0;
}

}  // namespace arc
