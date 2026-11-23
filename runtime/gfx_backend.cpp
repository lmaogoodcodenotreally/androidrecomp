// ANGLE backend: the guest speaks EGL/GLES2, ANGLE translates onto a modern
// desktop API. Kept in its own translation unit so the desktop-GL path, which
// is still the default, pays nothing for it.

#include "gfx_backend.h"

#if defined(_WIN32)

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <cstdio>
#include <cstring>

namespace arc {
namespace {

HMODULE g_egl = nullptr;
HMODULE g_gles = nullptr;
void* g_display = nullptr;
void* g_surface = nullptr;
void* g_context = nullptr;

using GetPlatformDisplayEXT = void* (*)(unsigned, void*, const int*);
using InitializeFn = int (*)(void*, int*, int*);
using ChooseConfigFn = int (*)(void*, const int*, void**, int, int*);
using CreateWindowSurfaceFn = void* (*)(void*, void*, void*, const int*);
using CreateContextFn = void* (*)(void*, void*, void*, const int*);
using MakeCurrentFn = int (*)(void*, void*, void*, void*);
using GetProcFn = void* (*)(const char*);
using SwapFn = int (*)(void*, void*);
using SwapIntervalFn = int (*)(void*, int);
using GetErrorFn = unsigned (*)();

GetPlatformDisplayEXT get_platform_display = nullptr;
InitializeFn initialize = nullptr;
ChooseConfigFn choose_config = nullptr;
CreateWindowSurfaceFn create_surface = nullptr;
CreateContextFn create_context = nullptr;
MakeCurrentFn make_current = nullptr;
GetProcFn get_proc = nullptr;
SwapFn swap = nullptr;
SwapIntervalFn swap_interval = nullptr;
GetErrorFn get_error = nullptr;

// EGL / ANGLE constants (from EGL/eglext.h and ANGLE's eglext_angle.h).
constexpr unsigned kNone = 0x3038;
constexpr unsigned kDefaultDisplay = 0;
constexpr unsigned kPlatformAngle = 0x3202;
constexpr unsigned kPlatformType = 0x3203;
constexpr unsigned kTypeD3d11 = 0x3208;
constexpr unsigned kTypeVulkan = 0x3450;
constexpr unsigned kSurfaceType = 0x3033;
constexpr unsigned kWindowBit = 0x0004;
constexpr unsigned kRenderableType = 0x3040;
constexpr unsigned kOpenGlEs2Bit = 0x0004;
constexpr unsigned kRedSize = 0x3024;
constexpr unsigned kGreenSize = 0x3025;
constexpr unsigned kBlueSize = 0x3026;
constexpr unsigned kAlphaSize = 0x3021;
constexpr unsigned kDepthSize = 0x3025 + 8;  // 0x302D
constexpr unsigned kStencilSize = 0x3031;
constexpr unsigned kContextClientVersion = 0x3098;

template <typename T>
T Load(HMODULE module, const char* name) {
  if (!module) return nullptr;
  return reinterpret_cast<T>(reinterpret_cast<void*>(
      GetProcAddress(module, name)));
}

}  // namespace

bool GfxInitAngle(const char* dll_dir, const char* backend, void* native,
                  std::string* err) {
  std::string egl_path = std::string(dll_dir) + "\\libEGL.dll";
  std::string gles_path = std::string(dll_dir) + "\\libGLESv2.dll";
  g_egl = LoadLibraryA(egl_path.c_str());
  g_gles = LoadLibraryA(gles_path.c_str());
  if (!g_egl || !g_gles) {
    if (err) *err = "ANGLE DLLs not found in " + std::string(dll_dir);
    GfxAngleShutdown();
    return false;
  }

  get_platform_display = Load<GetPlatformDisplayEXT>(g_egl,
                                                     "eglGetPlatformDisplayEXT");
  initialize = Load<InitializeFn>(g_egl, "eglInitialize");
  choose_config = Load<ChooseConfigFn>(g_egl, "eglChooseConfig");
  create_surface = Load<CreateWindowSurfaceFn>(g_egl, "eglCreateWindowSurface");
  create_context = Load<CreateContextFn>(g_egl, "eglCreateContext");
  make_current = Load<MakeCurrentFn>(g_egl, "eglMakeCurrent");
  get_proc = Load<GetProcFn>(g_egl, "eglGetProcAddress");
  swap = Load<SwapFn>(g_egl, "eglSwapBuffers");
  swap_interval = Load<SwapIntervalFn>(g_egl, "eglSwapInterval");
  get_error = Load<GetErrorFn>(g_egl, "eglGetError");
  if (!get_platform_display || !get_proc || !swap) {
    if (err) *err = "ANGLE EGL entry points missing";
    GfxAngleShutdown();
    return false;
  }

  const unsigned type = std::strcmp(backend, "vulkan") == 0
                            ? kTypeVulkan
                            : kTypeD3d11;
  const int display_attribs[] = {static_cast<int>(kPlatformType),
                                 static_cast<int>(type),
                                 static_cast<int>(kNone)};
  g_display = get_platform_display(kPlatformAngle,
                                   reinterpret_cast<void*>(kDefaultDisplay),
                                   display_attribs);
  if (!g_display) {
    if (err) *err = "eglGetPlatformDisplayEXT failed for ANGLE backend";
    GfxAngleShutdown();
    return false;
  }

  int major = 0, minor = 0;
  if (!initialize || !initialize(g_display, &major, &minor)) {
    if (err) *err = "eglInitialize failed";
    GfxAngleShutdown();
    return false;
  }

  // Ask only for what the engine cannot live without. Strict depth/stencil
  // requests leave Chromium's ANGLE builds with no matching config at all.
  const int config_attribs[] = {
      static_cast<int>(kSurfaceType),    static_cast<int>(kWindowBit),
      static_cast<int>(kRenderableType), static_cast<int>(kOpenGlEs2Bit),
      static_cast<int>(kNone)};
  void* config = nullptr;
  int count = 0;
  if (!choose_config || !choose_config(g_display, config_attribs, &config, 1,
                                       &count) ||
      count == 0) {
    if (err) *err = "eglChooseConfig found no GLES2 config";
    GfxAngleShutdown();
    return false;
  }

  g_surface = create_surface ? create_surface(g_display, config, native, nullptr)
                             : nullptr;
  if (!g_surface) {
    if (err) *err = "eglCreateWindowSurface failed";
    GfxAngleShutdown();
    return false;
  }

  const int context_attribs[] = {static_cast<int>(kContextClientVersion), 2,
                                 static_cast<int>(kNone)};
  g_context = create_context ? create_context(g_display, config, nullptr,
                                              context_attribs)
                             : nullptr;
  if (!g_context) {
    if (err) *err = "eglCreateContext failed";
    GfxAngleShutdown();
    return false;
  }

  if (!make_current || !make_current(g_display, g_surface, g_surface,
                                     g_context)) {
    if (err) *err = "eglMakeCurrent failed";
    GfxAngleShutdown();
    return false;
  }

  // Do not let ANGLE's D3D11 swap chain pace the frame loop; the host presents
  // explicitly and an occluded window must not block it.
  if (swap_interval) swap_interval(g_display, 0);

  std::fprintf(stderr, "[gfx] ANGLE %s on EGL %d.%d\n", backend, major, minor);
  return true;
}

void* GfxAngleProc(const char* name) {
  return get_proc ? get_proc(name) : nullptr;
}

bool GfxAngleActive() { return g_context != nullptr; }

bool GfxAngleSwap() {
  const bool ok = g_display && g_surface && swap && swap(g_display, g_surface) != 0;
  static int logged;
  if (++logged <= 3)
    std::fprintf(stderr, "[gfx] swap ok=%d eglError=%#x\n", ok,
                 get_error ? get_error() : 0u);
  return ok;
}

void GfxAngleShutdown() {
  g_context = nullptr;
  g_surface = nullptr;
  g_display = nullptr;
  get_platform_display = nullptr;
  initialize = nullptr;
  choose_config = nullptr;
  create_surface = nullptr;
  create_context = nullptr;
  make_current = nullptr;
  get_proc = nullptr;
  swap = nullptr;
  swap_interval = nullptr;
  get_error = nullptr;
  if (g_egl) FreeLibrary(g_egl);
  if (g_gles) FreeLibrary(g_gles);
  g_egl = nullptr;
  g_gles = nullptr;
}

}  // namespace arc

#else

namespace arc {
bool GfxInitAngle(const char*, const char*, void*, std::string* err) {
  if (err) *err = "ANGLE backend is Windows-only";
  return false;
}
void* GfxAngleProc(const char*) { return nullptr; }
bool GfxAngleActive() { return false; }
bool GfxAngleSwap() { return false; }
void GfxAngleShutdown() {}
}  // namespace arc

#endif
