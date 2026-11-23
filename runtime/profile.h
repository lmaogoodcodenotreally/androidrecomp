// Statistical sampling of the host threads, for the one question a call
// counter cannot answer: where the two seconds in a two-second frame went.
//
// A lifted function is a plain C function, so a sample of a thread's
// instruction pointer is a sample of the guest code that thread was running.
// The names come from a linker map: linking with /MAP and passing the map in
// ARC_SAMPLE_MAP turns an offset into fn0_<guest offset>, which names the
// lifted body directly. With no map the report still has the module and
// offset, enough to separate a driver stall from guest work.
//
// Off unless ARC_SAMPLE is set to an interval in milliseconds. This suspends
// every thread of the process to read its context, so it is a diagnostic, not
// something to leave on in a normal run.
#pragma once

namespace arc {

void StartSampler();
void StopSampler();

}  // namespace arc
