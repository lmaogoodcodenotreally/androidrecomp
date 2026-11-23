// A JNIEnv the guest can call through, with nothing behind it.
//
// The engine was written against a JVM: `init` and its neighbours take a
// JNIEnv* and immediately dereference it, because in JNI an environment *is* a
// pointer to a table of function pointers. Passing null is why the first call
// to `init` faulted reading address zero.
//
// There is no JVM here and there will not be one, so every slot is a stub. What
// matters is that the shape is right -- a pointer to a pointer to a table of
// 256 callable addresses -- and that each stub is distinguishable, so a run
// reports *which* slots a title actually reaches. Writing 233 named
// implementations before knowing which half-dozen are needed would be a lot of
// speculative work; the log tells us instead.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// A JNIEnv* for the guest, in guest terms: an address it can put in x0.
uint64_t arc_jni_env(void);

// The JavaVM, which is what `JNI_OnLoad` is handed. A library caches it and
// reaches every other thread's environment through it later, so this is the
// one pointer a host must give the guest before anything else asks.
uint64_t arc_jni_vm(void);

// The object an entry point is invoked on. Its fields are answered by the
// accessors rather than read out of it, so this only has to be a valid handle.
uint64_t arc_jni_object(void);

// A jstring holding this text, for an entry point whose Java declaration takes
// a String -- a path, a locale, a launch argument.
uint64_t arc_jni_string(const char* text);

// A stand-in `android.view.MotionEvent` carrying one touch.
//
// Unity on Android never sees a touch directly: Java's UnityPlayer receives a
// MotionEvent and hands it to the registered `nativeInjectEvent`, whose body
// reads the event back through JNI. With no Java here, the host has to be the
// one that calls nativeInjectEvent -- and the event it passes has to answer
// the questions that body asks (getAction, getX, getPointerCount, ...). This
// makes one, in the same arena every other handle comes from.
//
// `action` is the full Android action, including the pointer index bits:
// ACTION_DOWN 0, ACTION_UP 1, ACTION_MOVE 2, ACTION_CANCEL 3. Times are
// milliseconds on the monotonic clock (`arc::ShimMonotonicMillis`).
uint64_t arc_jni_motion_event(int action, float x, float y, int64_t down_time,
                              int64_t event_time);

// Registers every slot with the native bridge, so an indirect branch into the
// table is recognised as a call out to the host rather than a missing function.
void arc_jni_register(void);

// Natives the guest registered through RegisterNatives. Unity's player entry
// points are registered, not exported, so this is how a port finds initJni,
// nativeRender and the rest. `signature` may be null to match by name alone.
uint64_t arc_jni_lookup_native(const char* name, const char* signature);
size_t arc_jni_native_count(void);
void arc_jni_dump_natives(void);

// A port's answer to a Java method the title calls.
//
// The built-in table answers what the engine itself asks of Java: strings,
// paths, intents, the handful of questions about the device. A title's own
// Java half -- a library that talks to a service, say -- is not that, and a
// blank answer there is indistinguishable from a service that said nothing.
//
// A handler registered here is consulted before the table, matched on the
// class, name and signature the method id was resolved with. `receiver` is the
// object the call was made on, or the class for a static one; a Java object
// that remembers something it was constructed with carries it there (see
// `arc_jni_ctor_text`). `args` are the call's arguments in order, already
// unwrapped from whichever JNI calling convention the site used; a string
// argument is one of this environment's handles and `arc_jni_text` reads it.
// Returning non-zero means the call is answered and `result` goes back in x0.
typedef int (*ArcJavaMethodFn)(const char* owner, const char* name,
                               const char* signature, uint64_t receiver,
                               const uint64_t* args, size_t argc,
                               uint64_t* result);
void arc_jni_register_method(const char* owner, const char* name,
                             const char* signature, ArcJavaMethodFn fn);

// The text behind one of the environment's handles, or null if the handle is
// not one. A class handle and a string handle both carry text.
const char* arc_jni_text(uint64_t handle);

// What an object was constructed with, when the first argument of its
// constructor named something -- a string, or an id passed as a number.
//
// A Java object built to carry a request usually holds the identity of the
// caller's conversation -- a callback id, a token -- and is handed it at
// construction. The body that would later read it is Java, which is not here,
// so the only place that identity can be kept is the handle. It is kept
// beside the handle rather than as its text: an object is not a string, and
// reading one as the other would make every `IsInstanceOf` answer false.
const char* arc_jni_ctor_text(uint64_t handle);

// The class the object was constructed from. Kept with the argument above
// because a request is identified by both: the class says which command it is
// and the argument says which conversation it belongs to.
const char* arc_jni_ctor_class(uint64_t handle);

// The bytes behind a `byte[]` handle, or null if the handle is not one. A
// title that speaks a binary protocol to its Java half hands the payload over
// this way, and there is nothing textual to read.
const uint8_t* arc_jni_bytes(uint64_t handle, size_t* length);

// Hands managed code this text as a System.String, the way a Java callback
// would. The other half of answering a Java call is often the title's own C#
// being told the answer, and that conversation is in managed types: a
// `char*` of ours is not a string object to IL2CPP, and passing one where a
// string was expected is a fault rather than an empty answer.
uint64_t arc_guest_string(const char* utf8);

// Invokes a managed static method by name, with string arguments, through
// IL2CPP's exported reflection API. Returns 0 on success, negative if the
// assembly, class or method was not found, or 1 if the method threw.
// This is what a port uses to deliver `arc_guest_string` to the title.
int arc_call_managed(const char* assembly, const char* name_space,
                     const char* klass, const char* method, size_t argc,
                     const char* const* utf8_args);

// How many slots were called, and a report of which.
size_t arc_jni_call_count(void);
void arc_jni_report(void);

// Whether an address is one of the stand-in handles, so a fault on one can be
// reported as such rather than as an unattributable number.
int arc_jni_owns(uint64_t address);

#ifdef __cplusplus
}
#endif
