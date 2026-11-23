#include "window.h"
#include "gfx_backend.h"
#include "shim.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
// Occlusion probes use the window-manager entry points; the link spec is
// explicit about its libraries, so name this one here rather than in two rsp
// files that would then drift.
#pragma comment(lib, "user32.lib")
#endif

#if defined(ARC_HAVE_SDL2)
#include <SDL.h>
#include <SDL_syswm.h>
#endif

namespace arc {

#if defined(ARC_HAVE_SDL2)

namespace {
// How far one wheel notch drags, and how far one frame of that drag moves.
// The ratio is the number of frames a notch takes to play out: four, which is
// slow enough that the engine sees a drag rather than a jump, and quick enough
// that a list does not feel like it is being winched.
constexpr int kWheelNotch = 120;
constexpr int kWheelStep = 30;

void Put16(std::FILE* f, unsigned v) {
  std::fputc(v & 0xff, f);
  std::fputc((v >> 8) & 0xff, f);
}
void Put32(std::FILE* f, std::uint32_t v) {
  Put16(f, v & 0xffff);
  Put16(f, (v >> 16) & 0xffff);
}

// A frame is worth more than a draw-call count when the question is whether
// the engine has reached a menu or is still painting a loading quad. The host
// owns the back buffer just before Present, so read it there and write an
// uncompressed BMP: no dependency, and the result opens in every viewer.
bool SaveFrame(const char* path, int width, int height) {
  using ReadPixels = void (*)(int, int, int, int, unsigned, unsigned, void*);
  auto read = reinterpret_cast<ReadPixels>(
      arc::GfxAngleActive() ? arc::GfxAngleProc("glReadPixels")
                            : SDL_GL_GetProcAddress("glReadPixels"));
  if (!read || width <= 0 || height <= 0) return false;
  if (arc::GfxAngleActive()) {
    // GLES2 keeps the read buffer separate from the draw buffer; ANGLE does not
    // default it to BACK the way a desktop compatibility context does.
    using ReadBuffer = void (*)(unsigned);
    if (auto rb = reinterpret_cast<ReadBuffer>(
            arc::GfxAngleProc("glReadBuffer")))
      rb(0x0405);  // GL_BACK
  }
  const size_t stride = (static_cast<size_t>(width) * 3 + 3) & ~size_t(3);
  std::vector<unsigned char> rgba(static_cast<size_t>(width) * height * 4);
  std::vector<unsigned char> bgr(stride * height);
  read(0, 0, width, height, 0x1908 /* GL_RGBA */, 0x1401 /* GL_UNSIGNED_BYTE */,
       rgba.data());
  for (int y = 0; y < height; ++y) {
    const unsigned char* in = rgba.data() + static_cast<size_t>(y) * width * 4;
    unsigned char* out = bgr.data() + static_cast<size_t>(y) * stride;
    for (int x = 0; x < width; ++x) {
      out[x * 3 + 0] = in[x * 4 + 2];
      out[x * 3 + 1] = in[x * 4 + 1];
      out[x * 3 + 2] = in[x * 4 + 0];
    }
  }
  std::FILE* f = std::fopen(path, "wb");
  if (!f) return false;
  const std::uint32_t image = static_cast<std::uint32_t>(stride * height);
  std::fputc('B', f);
  std::fputc('M', f);
  Put32(f, 14 + 40 + image);
  Put16(f, 0);
  Put16(f, 0);
  Put32(f, 14 + 40);
  Put32(f, 40);
  Put32(f, static_cast<std::uint32_t>(width));
  Put32(f, static_cast<std::uint32_t>(height));
  Put16(f, 1);
  Put16(f, 24);
  Put32(f, 0);
  Put32(f, image);
  Put32(f, 2835);
  Put32(f, 2835);
  Put32(f, 0);
  Put32(f, 0);
  std::fwrite(bgr.data(), 1, bgr.size(), f);
  std::fclose(f);
  return true;
}
}  // namespace

Window::~Window() { Close(); }

bool Window::Open(const char* title, int width, int height, std::string* err,
                  int major, int minor) {
  if (SDL_Init(SDL_INIT_VIDEO) != 0) {
    if (err) *err = SDL_GetError();
    return false;
  }

  // ARC_GFX=angle-d3d11 / angle-vulkan routes the guest's GLES2 calls through
  // ANGLE onto a modern desktop API. The default stays desktop GL, which is
  // resolved directly through the compatibility context below.
  const char* gfx = std::getenv("ARC_GFX");
  const bool angle = gfx && *gfx && std::strncmp(gfx, "angle-", 6) == 0;

  if (!angle && major) {
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, major);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, minor);
  }

  if (!angle) {
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK,
                        SDL_GL_CONTEXT_PROFILE_COMPATIBILITY);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
  }

  Uint32 flags = SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI;
  flags |= angle ? 0 : SDL_WINDOW_OPENGL;
  SDL_Window* w = SDL_CreateWindow(title, SDL_WINDOWPOS_CENTERED,
                                   SDL_WINDOWPOS_CENTERED, width, height,
                                   flags);
  if (!w) {
    if (err) *err = SDL_GetError();
    return false;
  }
  window_ = w;
#if defined(_WIN32)
  SDL_SysWMinfo wm;
  SDL_VERSION(&wm.version);
  if (SDL_GetWindowWMInfo(w, &wm)) hwnd_ = wm.info.win.window;
#endif
  SDL_GL_GetDrawableSize(w, &width_, &height_);

  if (angle) {
#if defined(_WIN32)
    SDL_SysWMinfo info;
    SDL_VERSION(&info.version);
    if (!SDL_GetWindowWMInfo(w, &info)) {
      if (err) *err = "SDL_GetWindowWMInfo failed";
      Close();
      return false;
    }
    const char* dll_dir = std::getenv("ARC_ANGLE_DIR");
    if (!dll_dir || !*dll_dir)
      dll_dir = "C:\\Program Files (x86)\\Steam\\bin\\cef\\cef.win64";
    std::string angle_err;
    if (!GfxInitAngle(dll_dir, gfx + 6, info.info.win.window, &angle_err)) {
      if (err) *err = angle_err;
      Close();
      return false;
    }
    // A non-null marker: ANGLE owns the context, so Close must not hand this
    // to SDL_GL_DeleteContext.
    context_ = reinterpret_cast<void*>(1);
    return true;
#else
    if (err) *err = "ANGLE backend is Windows-only";
    Close();
    return false;
#endif
  }

  SDL_GLContext ctx = SDL_GL_CreateContext(w);
  if (!ctx) {
    if (err) *err = SDL_GetError();
    Close();
    return false;
  }
  context_ = ctx;
  // The engine's EGL shim hands this pair out on whichever thread makes the
  // context current; the guest runs its GL on a thread of its own.
  ShimSetGfxWindow(w, ctx);
  // The frame loop is the pacer here. Waiting for the compositor as well
  // doubles the latency and lets an occluded window block the swap for
  // seconds; the guest already asked for vsync through ARC_FAKE_VSYNC.
  // ARC_SWAP_INTERVAL=1 is a pacing knob: an unpaced loop can starve the
  // engine's asset workers on a small job system.
  const char* swap = std::getenv("ARC_SWAP_INTERVAL");
  SDL_GL_SetSwapInterval(swap ? std::atoi(swap) : 0);
  return true;
}

bool Window::MakeCurrent(std::string* err) {
  if (!window_ || !context_) {
    if (err) *err = "no window";
    return false;
  }
  if (GfxAngleActive()) return true;  // ANGLE made the context current at Open
  if (SDL_GL_MakeCurrent(static_cast<SDL_Window*>(window_),
                         static_cast<SDL_GLContext>(context_)) != 0) {
    if (err) *err = SDL_GetError();
    return false;
  }
  return true;
}

void Window::ReleaseCurrent() {
  if (GfxAngleActive()) return;
  if (window_)
    SDL_GL_MakeCurrent(static_cast<SDL_Window*>(window_), nullptr);
}

bool Window::PumpEvents() {
  SDL_Event e;
  bool open = true;
  while (SDL_PollEvent(&e)) {
    switch (e.type) {
      case SDL_QUIT:
        open = false;
        break;
      case SDL_WINDOWEVENT:
        if (e.window.event == SDL_WINDOWEVENT_SIZE_CHANGED) {
          SDL_GL_GetDrawableSize(static_cast<SDL_Window*>(window_), &width_,
                                 &height_);
          resized_ = true;
        } else if (e.window.event == SDL_WINDOWEVENT_CLOSE) {
          open = false;
        }
        break;
      case SDL_KEYDOWN:
        if (e.key.keysym.sym == SDLK_ESCAPE) open = false;
        break;
      case SDL_MOUSEBUTTONDOWN:
        if (e.button.button == SDL_BUTTON_LEFT) {
          // A real press arriving mid-wheel-drag has to end that drag first.
          // The engine tracks one finger: two presses with no release between
          // them leave it holding something forever, and the next tap goes
          // nowhere -- which looks exactly like input having died.
          if (wheel_active_) {
            wheel_active_ = false;
            wheel_pending_ = 0;
            pointers_.push_back({Pointer::Up, wheel_x_, wheel_y_, last_x_,
                                 last_y_});
          }
          last_x_ = e.button.x;
          last_y_ = e.button.y;
          down_ = true;
          pointers_.push_back({Pointer::Down, last_x_, last_y_, last_x_,
                               last_y_});
        }
        break;
      case SDL_MOUSEBUTTONUP:
        if (e.button.button == SDL_BUTTON_LEFT && down_) {
          down_ = false;
          pointers_.push_back({Pointer::Up, e.button.x, e.button.y, last_x_,
                               last_y_});
          last_x_ = e.button.x;
          last_y_ = e.button.y;
        }
        break;
      case SDL_MOUSEMOTION:
        // Only while held. A drag is the gesture the engine has an entry
        // point for; a hovering cursor is not an event Android can produce,
        // and feeding one in makes a tap out of every pass over the window.
        if (down_) {
          // Coalesced, because the host makes one guest call per queued
          // pointer and a guest call is not cheap. A mouse reporting at
          // hundreds of hertz produces dozens of motions per frame, and
          // delivering each one separately is what makes dragging a building
          // crawl -- the work is in the delivery, not in the movement.
          //
          // Android does not deliver them individually either: a touch queue
          // is drained per frame. Keep the oldest origin and the newest
          // destination, so the move the engine sees covers the same distance
          // it actually travelled.
          if (!pointers_.empty() && pointers_.back().kind == Pointer::Move) {
            pointers_.back().x = e.motion.x;
            pointers_.back().y = e.motion.y;
          } else {
            pointers_.push_back({Pointer::Move, e.motion.x, e.motion.y,
                                 last_x_, last_y_});
          }
          last_x_ = e.motion.x;
          last_y_ = e.motion.y;
        }
        break;
      case SDL_MOUSEWHEEL: {
        // Never while a real button is held: the engine has one finger, and
        // the drag already in progress is the one the player is making.
        if (down_) break;
        // Positive y is away from the user, which wants the content to move
        // the way a finger dragging downwards would move it.
        if (!wheel_active_ && wheel_pending_ == 0)
          SDL_GetMouseState(&wheel_x_, &wheel_y_);
        wheel_pending_ += e.wheel.y * kWheelNotch;
        break;
      }
      default:
        break;
    }
  }

  // One step of the synthetic drag per pump, which is one per frame. Never
  // while a real button is held: the engine tracks a single finger, and a
  // second one appearing mid-drag is not something a mouse can mean.
  if (!down_) {
    if (wheel_pending_ != 0 && !wheel_active_) {
      wheel_active_ = true;
      pointers_.push_back({Pointer::Down, wheel_x_, wheel_y_, wheel_x_,
                           wheel_y_});
      last_x_ = wheel_x_;
      last_y_ = wheel_y_;
    } else if (wheel_pending_ != 0) {
      int step = wheel_pending_;
      if (step > kWheelStep) step = kWheelStep;
      if (step < -kWheelStep) step = -kWheelStep;
      const int from_y = wheel_y_;
      wheel_y_ += step;
      wheel_pending_ -= step;
      pointers_.push_back({Pointer::Move, wheel_x_, wheel_y_, wheel_x_,
                           from_y});
      last_x_ = wheel_x_;
      last_y_ = wheel_y_;
    } else if (wheel_active_) {
      wheel_active_ = false;
      pointers_.push_back({Pointer::Up, wheel_x_, wheel_y_, last_x_, last_y_});
      last_x_ = wheel_x_;
      last_y_ = wheel_y_;
    }
  }
  return open;
}

bool Window::NextPointer(Pointer* out) {
  if (pointers_.empty()) return false;
  *out = pointers_.front();
  pointers_.pop_front();
  return true;
}

bool Window::Occluded() const {
#if defined(_WIN32)
  HWND hwnd = static_cast<HWND>(hwnd_);
  if (!hwnd) return false;
  if (IsIconic(hwnd) || !IsWindowVisible(hwnd)) return true;
  RECT client;
  if (!GetClientRect(hwnd, &client)) return false;
  const int w = client.right - client.left;
  const int h = client.bottom - client.top;
  if (w <= 0 || h <= 0) return true;
  // A compositor with no exposed pixels is allowed to hold the swap until the
  // window is shown again; that is what turns a covered window into a
  // multi-second frame and a "Not Responding" title. Probe five points: if
  // none belongs to this window, no part of it is exposed.
  const POINT probes[5] = {{w / 2, h / 2},         {w / 4, h / 4},
                           {3 * w / 4, h / 4},     {w / 4, 3 * h / 4},
                           {3 * w / 4, 3 * h / 4}};
  int exposed = 0;
  for (const POINT& probe : probes) {
    POINT screen = probe;
    if (!ClientToScreen(hwnd, &screen)) continue;
    const HWND at = WindowFromPoint(screen);
    if (at == hwnd || IsChild(hwnd, at)) ++exposed;
  }
  if (exposed == 0) return true;
  // Cloaked covers the virtual-desktop and suspended-UWP cases, which are
  // visible as far as IsWindowVisible is concerned.
  using DwmGetWindowAttributeFn = HRESULT(WINAPI*)(HWND, DWORD, PVOID, DWORD);
  static DwmGetWindowAttributeFn get = [] {
    HMODULE dwm = LoadLibraryW(L"dwmapi.dll");
    return dwm ? reinterpret_cast<DwmGetWindowAttributeFn>(
                     GetProcAddress(dwm, "DwmGetWindowAttribute"))
               : nullptr;
  }();
  if (get) {
    DWORD cloaked = 0;
    if (SUCCEEDED(get(hwnd, 14 /* DWMWA_CLOAKED */, &cloaked,
                      sizeof cloaked)) &&
        cloaked)
      return true;
  }
  return false;
#else
  return false;
#endif
}

bool Window::SkipPresent() const {
  // The environment cannot change under a running process, and these two are
  // consulted on every frame. UCRT's getenv walks the environment block under
  // a lock, which was measurable in the render loop; resolve them once.
  static const bool always = [] {
    const char* v = std::getenv("ARC_PRESENT_ALWAYS");
    return v && *v && std::strcmp(v, "0") != 0;
  }();
  if (always) return false;
  static const bool skip_disabled = [] {
    const char* v = std::getenv("ARC_PRESENT_SKIP_OCCLUDED");
    return v && *v && std::strcmp(v, "0") == 0;
  }();
  if (skip_disabled) return false;
  if (occl_ticks_ > 0) {
    --occl_ticks_;
    return occl_cached_;
  }
  occl_cached_ = Occluded();
  occl_ticks_ = 7;
  return occl_cached_;
}

void Window::FlushGl() {
  if (GfxAngleActive()) {
    if (auto flush = reinterpret_cast<void (*)()>(GfxAngleProc("glFlush")))
      flush();
    return;
  }
  if (auto flush = reinterpret_cast<void (*)()>(
          SDL_GL_GetProcAddress("glFlush")))
    flush();
}

bool Window::Present() {
  if (!window_) return false;
  // Diagnostics are fixed for the process lifetime; resolve them once rather
  // than taking the environment lock on every frame.
  static const char* const shot = std::getenv("ARC_SCREENSHOT");
  if (shot && *shot) {
    static int frame = 0;
    static const int target = [] {
      const char* want = std::getenv("ARC_SCREENSHOT_FRAME");
      return want && *want ? std::atoi(want) : 0;
    }();
    static const int every = [] {
      const char* text = std::getenv("ARC_SCREENSHOT_EVERY");
      return text && *text ? std::atoi(text) : 0;
    }();
    if (every > 0 && frame > 0 && frame % every == 0) {
      char path[512];
      std::snprintf(path, sizeof path, "%s.%d.bmp", shot, frame);
      if (SaveFrame(path, width_, height_))
        std::fprintf(stderr, "[shot] frame %d -> %s (%dx%d)\n", frame, path,
                     width_, height_);
    }
    if (frame == target && SaveFrame(shot, width_, height_))
      std::fprintf(stderr, "[shot] frame %d -> %s (%dx%d)\n", frame, shot,
                   width_, height_);
    ++frame;
  }
  static const bool present_log = std::getenv("ARC_PRESENT_LOG") != nullptr;
  static bool skipping = false;
  if (SkipPresent()) {
    ShimGfxAllowSwap(false);
    if (!skipping && present_log) {
      skipping = true;
      std::fprintf(stderr,
                   "[present] no exposed client area; skipping swaps\n");
    }
    skipping = true;
    // Submit the frame but do not wait: the driver's queue would otherwise
    // grow for the whole time the window is covered.
    FlushGl();
    return false;
  }
  ShimGfxAllowSwap(true);
  if (skipping) {
    skipping = false;
    if (present_log)
      std::fprintf(stderr, "[present] exposed; presenting again\n");
  }
  // The engine presents from the thread that holds the context. A swap it has
  // already issued is this frame; swapping again from here would only race it
  // for the context, which it holds for most of the frame.
  if (ShimGfxPresents()) return true;
  if (GfxAngleActive()) {
    static int fb_logged;
    if (++fb_logged <= 3) {
      int fb = -1;
      if (auto gi = reinterpret_cast<void (*)(unsigned, int*)>(
              arc::GfxAngleProc("glGetIntegerv")))
        gi(0x8CA6 /* GL_FRAMEBUFFER_BINDING */, &fb);
      std::fprintf(stderr, "[gfx] framebuffer bound at present = %d\n", fb);
    }
    GfxAngleSwap();
    return true;
  }
  // The engine hands the context back and forth between its own threads and
  // releases it after each turn, so the host cannot assume it is still current
  // here. Take it back for the swap; if a render thread is mid-turn, wait a few
  // scheduler quanta for it to finish rather than swapping with nothing bound.
  if (SDL_GL_GetCurrentContext() != static_cast<SDL_GLContext>(context_)) {
    bool held = false;
    for (int attempt = 0; attempt < 8 && !held; ++attempt) {
      held = SDL_GL_MakeCurrent(static_cast<SDL_Window*>(window_),
                                static_cast<SDL_GLContext>(context_)) == 0;
      if (!held) SDL_Delay(0);
    }
    if (!held) {
      static int logged = 0;
      if (present_log && ++logged <= 3)
        std::fprintf(stderr, "[present] context busy; skipping swap\n");
      return false;
    }
  }
  SDL_GL_SwapWindow(static_cast<SDL_Window*>(window_));
  return true;
}

void Window::SetTitle(const char* title) {
  if (!window_ || !title) return;
  SDL_SetWindowTitle(static_cast<SDL_Window*>(window_), title);
}

void Window::Close() {
  const bool angle = GfxAngleActive();
  if (angle) GfxAngleShutdown();
  if (context_ && !angle) {
    SDL_GL_DeleteContext(static_cast<SDL_GLContext>(context_));
  }
  context_ = nullptr;
  if (window_) {
    SDL_DestroyWindow(static_cast<SDL_Window*>(window_));
    window_ = nullptr;
    SDL_Quit();
  }
}

bool Window::TakeResized() {
  bool was = resized_;
  resized_ = false;
  return was;
}

std::string Window::Describe() const {
  if (!context_) return "no context";
  if (GfxAngleActive()) return "ANGLE GLES2 (desktop backend)";
  int major = 0, minor = 0, profile = 0;
  SDL_GL_GetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, &major);
  SDL_GL_GetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, &minor);
  SDL_GL_GetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, &profile);
  const char* kind = profile == SDL_GL_CONTEXT_PROFILE_ES     ? "ES"
                     : profile == SDL_GL_CONTEXT_PROFILE_CORE ? "core"
                                                              : "compatibility";
  return "GL " + std::to_string(major) + "." + std::to_string(minor) + " " +
         kind;
}

#else  // no SDL2

Window::~Window() = default;
bool Window::Open(const char*, int, int, std::string* err, int, int) {
  if (err) *err = "built without SDL2";
  return false;
}
bool Window::MakeCurrent(std::string* err) {
  if (err) *err = "built without SDL2";
  return false;
}
void Window::ReleaseCurrent() {}
bool Window::PumpEvents() { return false; }
bool Window::NextPointer(Pointer*) { return false; }
bool Window::Present() { return false; }
bool Window::Occluded() const { return false; }
void Window::SetTitle(const char*) {}
void Window::Close() {}
bool Window::TakeResized() { return false; }
std::string Window::Describe() const { return "built without SDL2"; }

#endif

}  // namespace arc
