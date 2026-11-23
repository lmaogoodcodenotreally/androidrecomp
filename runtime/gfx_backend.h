#pragma once

#include <string>

namespace arc {

// ANGLE is a GLES2 implementation that translates to a modern desktop API. The
// guest Unity build is EGL/GLES2-only, so this is the only way to reach
// D3D11/D3D12/Vulkan without changing the lifted program.
//
// `dll_dir` holds libEGL.dll and libGLESv2.dll. `backend` is "d3d11" or
// "vulkan". `native` is the Win32 HWND behind the SDL window.
bool GfxInitAngle(const char* dll_dir, const char* backend, void* native,
                  std::string* err);

// GLES2 entry point from the active ANGLE context. Returns null for the
// desktop-GL backend, whose caller then uses SDL_GL_GetProcAddress.
void* GfxAngleProc(const char* name);

bool GfxAngleActive();
bool GfxAngleSwap();
void GfxAngleShutdown();

}  // namespace arc
