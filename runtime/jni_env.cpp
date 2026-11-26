#include "jni_env.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#endif

#include "arm64_context.h"
#include "shim.h"

namespace {

// The environment's table, and eight more stubs behind it for the JavaVM's.
// The two are separate interfaces the guest reaches identically -- a table of
// function pointers, indexed and branched through -- so they share one pool of
// stubs and the VM's indices are simply offset past the environment's.
constexpr size_t kEnvSlots = 256;
constexpr size_t kVmSlots = 8;
constexpr size_t kVmBase = kEnvSlots;
constexpr size_t kSlots = kEnvSlots + kVmSlots;

// The table the guest sees, and the pointer that names it. A JNIEnv* is the
// address of `g_table`, not of the table itself -- one more level of
// indirection than feels natural, and the reason a null environment faults on
// a read rather than on a call.
uint64_t g_slots[kSlots];
uint64_t g_table = reinterpret_cast<uint64_t>(g_slots);

// The invocation interface, which is how the engine reaches the VM rather than
// one environment: three reserved entries, then DestroyJavaVM,
// AttachCurrentThread, DetachCurrentThread, GetEnv and
// AttachCurrentThreadAsDaemon. A JavaVM* is the address of `g_vm_table`, the
// same double indirection as above.
uint64_t g_vm_slots[kVmSlots];
uint64_t g_vm_table = reinterpret_cast<uint64_t>(g_vm_slots);

// Counters, not state the guest depends on -- but written from every thread
// the engine starts, so they are atomic rather than merely approximate.
std::atomic<size_t> g_hits[kSlots];
std::atomic<size_t> g_total;

// Native methods the guest registered through RegisterNatives.
//
// Unity does not export its player entry points as Java_* symbols; it builds a
// JNINativeMethod table and hands it to the VM during JNI_OnLoad. The shim used
// to discard that call, which is right for a boot that only needs the engine's
// constructors and wrong for a port that has to drive a frame. Keeping the
// triples gives the host a name to call by; the class name is not part of the
// registration call and the shim's FindClass handles are opaque, so name plus
// signature is what identifies a method.
struct RegisteredNative {
  std::string name;
  std::string signature;
  uint64_t fn = 0;
};
std::mutex g_natives_lock;
std::vector<RegisteredNative> g_natives;

// --- the arena -------------------------------------------------------------
// Handles are blocks from a zeroed arena, reserved low in the address space. A
// handle is opaque in principle, but code with a 32-bit lineage stores one in
// an int, and a raw heap pointer does not survive that -- it reappears in
// arithmetic as a large negative number.
constexpr size_t kArenaSize = 32u << 20;
constexpr size_t kBlock = 64u << 10;
unsigned char* g_arena;
size_t g_arena_used;

unsigned char* ReserveLowArena() {
#if defined(_WIN32)
  for (uintptr_t at = 0x10000000; at < 0x60000000; at += 0x1000000) {
    void* p = VirtualAlloc(reinterpret_cast<void*>(at), kArenaSize,
                           MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (p) return static_cast<unsigned char*>(p);
  }
#endif
  return static_cast<unsigned char*>(calloc(1, kArenaSize));
}

// Locked, because the engine reaches this from every thread it starts. Two
// threads racing here do not merely miscount: both leave with the same block,
// and a handle that two owners are writing over is a pointer that reads back
// as something no one stored -- which surfaces far away as a branch to an
// address that was never in any mapped image.
std::mutex g_arena_lock;
size_t g_active_slot = 0;
bool g_trace_next_alloc = false;

// What an object was constructed with. See `arc_jni_ctor_text`, which is the
// only reader. Declared up here because the arena recycles and `Allocate` has
// to forget the previous occupant.
//
// The class is kept beside the argument because the two together are the
// identity: a command object is a class that names the command and an argument
// that names the conversation. Either alone would match something else -- the
// arena has held a `java.lang.String` constructed from the word "Log" at an
// address that later belonged to a request.
struct CtorRecord {
  std::string klass;
  std::string first;
};
std::mutex g_ctor_lock;
std::map<uint64_t, CtorRecord> g_ctor_text;

// --- the synthetic MotionEvent ---------------------------------------------
//
// A touch reaches Unity through Java on Android: UnityPlayer.onTouchEvent
// passes the MotionEvent to the registered `nativeInjectEvent`, and the native
// body reads it back through JNI. There is no Java here, so the event has to
// be a stand-in that answers the same questions. Its fields live beside the
// handle for the same reason a constructor's argument does: the handle is an
// arena block, and `IsInstanceOf` reads a handle's text as its class name --
// which is exactly what the event must be, "android/view/MotionEvent".
struct MotionEventState {
  int action = 0;
  float x = 0, y = 0;
  int64_t down_time = 0, event_time = 0;
};
std::mutex g_motion_lock;
std::map<uint64_t, MotionEventState> g_motion_events;

bool MotionEventStateFor(uint64_t handle, MotionEventState* out) {
  std::lock_guard<std::mutex> held(g_motion_lock);
  const auto it = g_motion_events.find(handle);
  if (it == g_motion_events.end()) return false;
  if (out) *out = it->second;
  return true;
}

uint64_t Allocate() {
  std::lock_guard<std::mutex> held(g_arena_lock);
  if (!g_arena) {
    g_arena = ReserveLowArena();
    if (!g_arena) return 0;
  }
  if (g_arena_used + kBlock > kArenaSize) g_arena_used = 0;  // wrap; nothing frees
  const uint64_t p = reinterpret_cast<uint64_t>(g_arena + g_arena_used);
  g_arena_used += kBlock;
  // The arena wraps after five hundred blocks and nothing is freed, so this
  // address has held other things. An identity recorded for the last occupant
  // would otherwise answer for this one -- it is how a callback id was read as
  // the string "Log", left over from a `java.lang.String`.
  {
    std::lock_guard<std::mutex> ctor_held(g_ctor_lock);
    g_ctor_text.erase(p);
  }
  {
    std::lock_guard<std::mutex> motion_held(g_motion_lock);
    g_motion_events.erase(p);
  }
  // Cleared, because the arena wraps and nothing frees: a handle handed back
  // for a question we could not answer is meant to read as empty, and an
  // uncleared block reads as whatever string last occupied it. That is worse
  // than a wrong answer -- it is a *plausible* one, from somewhere else in
  // the program. It arrived here as a proxy host named "LandDataVersion",
  // which the network stack then spent every request trying to resolve.
  memset(reinterpret_cast<void*>(p), 0, kBlock);
  if (g_trace_next_alloc && g_active_slot) {
    g_trace_next_alloc = false;
    fprintf(stderr, "[jni] next alloc %#llx from slot %zu\n",
            static_cast<unsigned long long>(p), g_active_slot);
  }
  return p;
}

// A string handle carries its text, because something will measure it. An
// empty one is not harmless: the engine takes the length, subtracts one, and
// hands the result to memcpy.
//
// The last word of the block is a marker that says this handle is a string.
// GetStringUTFChars has a fallback for opaque handles that would otherwise
// read as empty -- the boot.config Scanner is the case it exists for -- and
// without the marker that fallback cannot tell a legitimate empty string (a
// Locale with no language tag, say) from a handle that never carried text at
// all. The fallback then hands back whichever file was read last, which is how
// a language tag became the whole boot config.
constexpr uint64_t kTextHandleMagic = 0x5458542D48414E44ull;  // "TXTHAND"

uint64_t AllocateText(const char* text) {
  const uint64_t p = Allocate();
  if (p && text) {
    const size_t n = strlen(text);
    const size_t cap = kBlock - 2 * sizeof(uint64_t);
    const size_t copy = n < cap ? n : cap - 1;
    memcpy(reinterpret_cast<void*>(p), text, copy);
    reinterpret_cast<unsigned char*>(p)[copy] = 0;
  }
  if (p) {
    *reinterpret_cast<uint64_t*>(p + kBlock - sizeof(uint64_t)) =
        kTextHandleMagic;
  }
  if (p && g_active_slot && (!text || !*text))
    fprintf(stderr, "[jni] empty text block %#llx from slot %zu\n",
            static_cast<unsigned long long>(p), g_active_slot);
  return p;
}

// Unity's Java bootstrap reads bin/Data/boot.config through AssetManager.open
// and java.util.Scanner before it ever reaches native code. A generic opaque
// handle makes that file read back empty, and Unity treats an empty boot
// config as a fatal startup error. The host already knows where the extracted
// assets live, so answer those two calls from the file itself.
std::mutex g_asset_text_lock;
std::string g_last_asset_text;
std::string g_last_utf8;

std::string ReadAssetFile(const char* name) {
  const char* root = arc::ShimAssetRoot();
  if (!root || !*root || !name) return {};
  const char* p = name;
  while (*p == '/') ++p;
  if (strncmp(p, "assets/", 7) == 0) p += 7;
  std::ifstream in(std::filesystem::path(root) / p, std::ios::binary);
  if (!in) return {};
  std::string text((std::istreambuf_iterator<char>(in)),
                   std::istreambuf_iterator<char>());
  return text;
}

// --- the object the engine is actually asking about ------------------------
//
// A JNI entry point is handed the object its Java declaration named, and reads
// its configuration out of it field by field. Answering every read with zero
// is why initialisation could not proceed: the engine was promised a device
// description, found nothing, and did arithmetic on the nothing.
//
// The field names are read out of the dex -- see tools/dex_contract.py. The
// values are ours to choose; what matters is that they are the right shape and
// that a string is never empty.
struct Field {
  const char* name;
  char kind;  // 's' text, 'i' integer, 'f' float, 'z' boolean
  const char* text;
  int64_t number;
  double real;
};

const Field kFields[] = {
    {"appPath", 's', "/assets", 0, 0},
    // Answered from the asset root, not from here -- see DirectoryForName.
    {"writablePath", 's', ".", 0, 0},
    {"appBundle", 's', "androidrecomp.host", 0, 0},
    // Three components, not two. The engine parses these, and with "1.0" its
    // search for the second separator finds nothing -- the span computed from
    // that comes out negative and goes straight into memcpy as -1, which reads
    // until it leaves the heap. The value is ours to choose; the shape is not.
    {"appVersion", 's', "1.0.0", 0, 0},
    {"versionName", 's', "1.0.0", 0, 0},
    {"version", 's', "1.0.0", 0, 0},
    {"name", 's', "Mario Kart Tour", 0, 0},
    {"url", 's', "http://127.0.0.1:8443/", 0, 0},
    {"sessionTracker", 's', "0", 0, 0},
    {"clientVersion", 's', "1.0.0", 0, 0},
    {"deviceName", 's', "androidrecomp", 0, 0},
    {"deviceVersion", 's', "1.0.0", 0, 0},
    {"manufacturerName", 's', "generic", 0, 0},
    {"widthPixels", 'i', nullptr, 1280, 0},
    {"heightPixels", 'i', nullptr, 720, 0},
    {"densityDpi", 'i', nullptr, 320, 0},
    {"language", 's', "en", 0, 0},
    {"locale", 's', "en_US", 0, 0},
    {"width", 'i', nullptr, 1280, 0},
    {"height", 'i', nullptr, 720, 0},
    {"orientation", 'i', nullptr, 0, 0},
    {"model", 'i', nullptr, 0, 0},
    {"dpiFromCategory", 'i', nullptr, 320, 0},
    {"miNavBarHeight", 'i', nullptr, 0, 0},
    {"density", 'f', nullptr, 0, 2.0},
    {"densityDPI", 'f', nullptr, 0, 320.0},
    {"mbExternalStorageUnusable", 'z', nullptr, 0, 0},
    // The API level, which read as zero -- not a version any Android
    // ever had, so every check against it took a branch meant for no
    // real device. 21 is the oldest level that can load an arm64
    // library, which keeps the engine on its most self-contained paths
    // rather than ones gated behind newer Java APIs nothing here has.
    {"sdk_int", 'i', nullptr, 21, 0},
    // Zero already meant "no override"; naming it stops it being
    // reported as missing and records that the value is deliberate.
    {"screenOverride", 'i', nullptr, 0, 0},
};
constexpr size_t kFieldCount = sizeof(kFields) / sizeof(kFields[0]);

// Field ids are small and unlike handles, so a mix-up between the two is loud
// rather than plausible.
constexpr uint64_t kFieldIdBase = 0x0F1E1D00;

const Field* FieldFromId(uint64_t id) {
  const uint64_t index = id - kFieldIdBase;
  return index < kFieldCount ? &kFields[index] : nullptr;
}

// Fields the engine asked for that this table has no answer for. Each reads as
// zero of its type -- an empty string, a zero count -- which is a plausible
// value rather than a visible failure, so the engine carries it away and comes
// apart somewhere else entirely. Naming them is the difference between that and
// a line saying which one to add.
std::mutex g_field_lock;
std::deque<std::string> g_unknown_fields;

uint64_t FieldIdFor(const char* name, const char* sig) {
  if (name) {
    for (size_t i = 0; i < kFieldCount; ++i)
      if (strcmp(kFields[i].name, name) == 0) return kFieldIdBase + i;
    // Record the signature with the name. A missing field has to be added back
    // as some particular type, and the caller already said which -- guessing
    // between an int, a boolean and a string is a wrong answer two times in
    // three, and each wrong answer reads as a plausible zero somewhere later.
    const std::string entry = std::string(name) + "  " + (sig ? sig : "?");
    std::lock_guard<std::mutex> held(g_field_lock);
    bool seen = false;
    for (const std::string& s : g_unknown_fields)
      if (s == entry) { seen = true; break; }
    if (!seen) g_unknown_fields.push_back(entry);
  }
  // Unknown field: still a usable id, and it reads as zero of its type.
  return kFieldIdBase + kFieldCount;
}

// --- methods ---------------------------------------------------------------
//
// A method id is opaque, so it may as well be an index into the names the
// engine asked for. That is worth keeping, because a call through an id is
// otherwise answerable only with a blank handle -- and a method that was
// supposed to return a path, answered with an empty string, is how a bundle
// path becomes "/".
//
// The class and the signature are kept with the name. The same method name
// means different things on different classes, and a title that hands Java a
// *binary* argument (NPF's `UnityBridge.execute(byte[])` is the one that
// matters here) cannot be understood from the name alone. Keeping them costs
// one string per lookup and makes a call trace read `class.method(sig)`,
// which is the difference between "something asked for execute" and "which
// execute, on what, returning what".
//
// A deque and not a vector: names are handed out as pointers and another
// thread may be adding one, which would move a vector's contents.
constexpr uint64_t kMethodIdBase = 0x0E2D1000;
std::mutex g_method_lock;

struct MethodEntry {
  std::string name;
  std::string owner;
  std::string signature;
};

std::deque<MethodEntry> g_method_entries;
std::deque<std::string> g_method_unanswered;

// Java methods to report as missing, from ARC_JNI_ABSENT -- a comma-separated
// list of names. Matched on the method name alone, which is how the rest of
// this bridge identifies a method too.
bool MethodIsAbsent(const char* name) {
  static const char* const list = getenv("ARC_JNI_ABSENT");
  if (!list || !*list) return false;
  const size_t n = strlen(name);
  for (const char* p = list; *p;) {
    const char* comma = strchr(p, ',');
    const size_t len = comma ? static_cast<size_t>(comma - p) : strlen(p);
    if (len == n && strncmp(p, name, n) == 0) return true;
    p = comma ? comma + 1 : p + len;
  }
  return false;
}

uint64_t MethodIdFor(const char* owner, const char* name,
                     const char* signature) {
  const std::string wanted = name ? name : "?";
  std::lock_guard<std::mutex> held(g_method_lock);
  for (size_t i = 0; i < g_method_entries.size(); ++i) {
    if (g_method_entries[i].name == wanted) {
      // The first lookup creates the id; a later one for the same name only
      // fills in what the first one did not know, so an early id resolved
      // before its class was recorded is not silently re-pointed.
      if (g_method_entries[i].owner.empty() && owner)
        g_method_entries[i].owner = owner;
      if (g_method_entries[i].signature.empty() && signature)
        g_method_entries[i].signature = signature;
      return kMethodIdBase + i;
    }
  }
  g_method_entries.push_back(
      {wanted, owner ? owner : "", signature ? signature : ""});
  return kMethodIdBase + g_method_entries.size() - 1;
}

const MethodEntry* MethodAt(uint64_t id) {
  const uint64_t index = id - kMethodIdBase;
  if (id < kMethodIdBase) return nullptr;
  return index < g_method_entries.size() ? &g_method_entries[index] : nullptr;
}

const char* MethodName(uint64_t id) {
  std::lock_guard<std::mutex> held(g_method_lock);
  const MethodEntry* e = MethodAt(id);
  return e ? e->name.c_str() : nullptr;
}

const char* MethodOwner(uint64_t id) {
  std::lock_guard<std::mutex> held(g_method_lock);
  const MethodEntry* e = MethodAt(id);
  return e && !e->owner.empty() ? e->owner.c_str() : nullptr;
}

const char* MethodSignature(uint64_t id) {
  std::lock_guard<std::mutex> held(g_method_lock);
  const MethodEntry* e = MethodAt(id);
  return e && !e->signature.empty() ? e->signature.c_str() : nullptr;
}

// The return type of a JNI method id, read from its signature: the character
// after `(args)`. Where an answer belongs depends on it -- a float leaves in v0
// and everything else in x0 -- and the signature is the only place the engine
// recorded that. Slot numbers are an accident of the order names happened to be
// resolved in, so they cannot carry it.
char MethodReturnType(uint64_t id) {
  const char* signature = MethodSignature(id);
  if (!signature) return 0;
  const char* close = strrchr(signature, ')');
  return close && close[1] ? close[1] : 0;
}

// Every JNI slot check used to call getenv, which locks the CRT environment
// on each of the millions of calls a frame makes. Read the flag once.
static bool TraceJni() {
  static const bool on = std::getenv("ARC_TRACE_JNI") != nullptr;
  return on;
}
void NoteUnanswered(const char* name) {
  if (!name) return;
  std::lock_guard<std::mutex> held(g_method_lock);
  for (const std::string& s : g_method_unanswered)
    if (s == name) return;
  g_method_unanswered.push_back(name);
  // The report at the end of a run lists these, and a title that is parked on
  // a download never gets to the end of a run. Naming each one the first time
  // it is asked turns "stuck at 0%" into "Java was asked for this and got a
  // blank", which is the whole question when Java is not here.
  static const bool trace = getenv("ARC_TRACE_JNI_MISS") != nullptr;
  if (trace) fprintf(stderr, "[jni] no answer for %s\n", name);
}

// Every method actually invoked, as opposed to merely looked up. The
// difference matters: a name can be resolved to an id during setup and then
// never called, and reading a value into the answer for a call that never
// happened is a way to spend a long time being wrong.
std::deque<std::string> g_method_called;

void NoteCalled(const char* name) {
  if (!name) return;
  std::lock_guard<std::mutex> held(g_method_lock);
  for (const std::string& s : g_method_called)
    if (s == name) return;
  g_method_called.push_back(name);
}

// What a variadic call actually carried, when the answer had to come from an
// argument and none was recognised. "Not one of our handles" is the whole
// failure and it is invisible otherwise: the call falls through to the table,
// misses, and is reported as a method with no answer rather than as a method
// whose argument we could not read. ARC_TRACE_PREFS=1.
void TraceVaArgs(const char* name, Arm64Ctx* c) {
  static const bool on = getenv("ARC_TRACE_PREFS") != nullptr;
  if (!on) return;
  fprintf(stderr, "[pref] %s  self=%#llx%s \"%.60s\"\n", name,
          static_cast<unsigned long long>(c->x[1]),
          arc_jni_owns(c->x[1]) ? " (ours)" : "",
          arc_jni_owns(c->x[1]) ? reinterpret_cast<const char*>(c->x[1]) : "");
  if (!c->x[3]) return;
  std::vector<unsigned char> cursor(arc::ShimVaListSize());
  memcpy(cursor.data(), reinterpret_cast<const void*>(c->x[3]), cursor.size());
  for (int i = 0; i < 4; ++i) {
    const uint64_t v = arc::ShimVaNextInt(cursor.data());
    const bool ours = v && arc_jni_owns(v);
    fprintf(stderr, "        arg%d = %#llx%s \"%.60s\"\n", i,
            static_cast<unsigned long long>(v), ours ? " (ours)" : "",
            ours ? reinterpret_cast<const char*>(v) : "");
  }
}

// What a method returning a string should answer with. The engine builds paths
// out of these, so an empty one is worse than a wrong one: it silently becomes
// the filesystem root.
struct MethodText {
  const char* name;
  const char* text;
};

const MethodText kMethodText[] = {
    {"getCocos2dxPackageName", "androidrecomp.host"},
    {"getPackageName", "androidrecomp.host"},
    // A document, not a name: the engine parses this one as JSON and says so
    // when it cannot ("Failed to construct JsonMap from string ...").
    {"getDeviceInfo", "{}"},
    {"getDeviceModel", "androidrecomp"},
    {"getCurrentLanguage", "en"},
    {"getLanguage", "en"},
    // java.util.Locale.getDefault().toLanguageTag() is how the engine reads
    // the device locale before IL2CPP starts. An empty answer is not "no
    // preference" here: IL2CPP hands it to CultureInfo, which rejects it and
    // throws while the runtime is still initialising. A well-formed BCP-47
    // tag is the answer that lets startup continue.
    {"toLanguageTag", "en-US"},
    {"getCountry", "US"},
    {"getVersion", "1.0"},
    // A locale variant and an absent preference are legitimately empty, and
    // saying so is different from having no answer.
    {"getVariant", ""},
    {"getStringPreference", ""},
    // The device identity the title registers with its server under. Android
    // answers this from settings that survive a reinstall, so it has to be
    // stable across runs on this host too -- a new one each launch reads to
    // the server as a new device every time. A fixed string is stable by
    // construction; ARC_DEVICE_ID replaces it when one host has to look like
    // several devices.
    {"getUUID", "androidrecomp-00000000-0000-0000-0000-000000000000"},
    {"getDeviceId", "androidrecomp-00000000-0000-0000-0000-000000000000"},
    // A device with no mobile network, described honestly: no carrier, and a
    // model name that is this host rather than a phone it is pretending to
    // be. Empty is a real answer to both; absent is not, and an SDK that gets
    // a blank handle where it expected a string measures it and gets zero
    // either way -- but only one of the two is recorded as answered.
    {"getCarrier", ""},
    {"getDeviceString", "androidrecomp"},
    // No proxy. Nimble asks Java for one and puts whatever comes back in
    // front of every request it makes, so this has to be empty rather than
    // absent -- and empty rather than left to a blank handle, because the
    // question is asked once and the answer is used for the whole run.
    {"getHttpProxy", ""},
    // Nimble's network status, which it reads as a Java enum: getStatus()
    // hands back the constant and ordinal() turns it into a number. Three is
    // "reachable over wifi", and it is the answer a machine on a wire should
    // give -- a host that says "unknown" leaves the SDK waiting for a network
    // it has, so nothing is ever sent and nothing ever fails.
    {"getStatus", "3"},
};

// The first argument in a guest va_list that is one of our own handles.
//
// Which position a jstring or a byte[] occupies depends on the Java signature,
// which we do not have -- but a handle we made is recognisable, and no other
// argument will be. So "the first one that is ours" is both the answer and the
// only question we can actually ask.
// An object array a handle can actually be.
//
// Everything else here is a block of text, and an "array" was that text with
// its length measured by strlen -- which is right for the byte arrays the
// engine fills and wrong for an array *of objects*, which the SDK reads
// element by element. Handing it a string there is not a smaller answer, it
// is a zero-length one: the caller collects nothing, then reads the elements
// it was promised and runs off the end of its own vector.
//
// So: an array handle starts with a NUL, so anything that reads it as text
// still sees an empty string, and carries its count and its elements after
// a magic word that ordinary text will not begin with.
constexpr uint64_t kArrayMagic = 0x0A11A77A7C0DEULL;
constexpr size_t kArrayMagicOffset = 8;
constexpr size_t kArrayCountOffset = 16;
constexpr size_t kArrayFirstOffset = 24;

bool IsObjectArray(uint64_t handle) {
  if (!handle || !arc_jni_owns(handle)) return false;
  uint64_t magic = 0;
  memcpy(&magic, reinterpret_cast<const void*>(handle + kArrayMagicOffset),
         sizeof magic);
  return magic == kArrayMagic;
}

uint64_t ArrayCount(uint64_t handle) {
  uint64_t n = 0;
  memcpy(&n, reinterpret_cast<const void*>(handle + kArrayCountOffset),
         sizeof n);
  return n;
}

uint64_t ArrayElement(uint64_t handle, uint64_t i) {
  if (i >= ArrayCount(handle)) return 0;
  uint64_t v = 0;
  memcpy(&v,
         reinterpret_cast<const void*>(handle + kArrayFirstOffset + i * 8),
         sizeof v);
  return v;
}

uint64_t AllocateObjectArray(const uint64_t* items, size_t n) {
  const uint64_t p = Allocate();
  if (!p) return 0;
  if (kArrayFirstOffset + n * 8 > kBlock) n = 0;
  const uint64_t magic = kArrayMagic, count = n;
  memcpy(reinterpret_cast<void*>(p + kArrayMagicOffset), &magic, sizeof magic);
  memcpy(reinterpret_cast<void*>(p + kArrayCountOffset), &count, sizeof count);
  for (size_t i = 0; i < n; ++i)
    memcpy(reinterpret_cast<void*>(p + kArrayFirstOffset + i * 8), &items[i],
           sizeof items[i]);
  return p;
}

// A byte array is *bytes*, not text. The engine's own arrays are ASCII paths
// and log lines, and measuring one with strlen was right for those; a protobuf
// command or an HTTP response is not text at all -- it carries NULs, and
// strlen stops at the first one. So a byte array carries its own length behind
// a magic word at the end of the block.
//
// The bytes still start at offset zero, and a block that holds them still ends
// with the text marker, because some of the engine's own helpers hand a filled
// byte array to something that reads it as a string. Keeping both views of the
// same block is what makes the array usable as an array without breaking it as
// a string.
constexpr uint64_t kBytesMagic = 0x42595445A44A45ull;  // "BYTEAJE"
constexpr size_t kBytesLengthOffset = kBlock - 3 * sizeof(uint64_t);
constexpr size_t kBytesMagicOffset = kBlock - 2 * sizeof(uint64_t);
constexpr size_t kBytesCapacity = kBlock - 3 * sizeof(uint64_t) - 1;

bool IsByteArray(uint64_t handle) {
  if (!handle || !arc_jni_owns(handle)) return false;
  uint64_t magic = 0;
  memcpy(&magic, reinterpret_cast<const void*>(handle + kBytesMagicOffset),
         sizeof magic);
  return magic == kBytesMagic;
}

uint64_t ByteArrayLength(uint64_t handle) {
  uint64_t n = 0;
  memcpy(&n, reinterpret_cast<const void*>(handle + kBytesLengthOffset),
         sizeof n);
  return n;
}

void SetByteArrayLength(uint64_t handle, uint64_t n) {
  if (n > kBytesCapacity) n = kBytesCapacity;
  memcpy(reinterpret_cast<void*>(handle + kBytesLengthOffset), &n, sizeof n);
}

uint64_t AllocateByteArray(uint64_t n) {
  const uint64_t p = Allocate();
  if (!p) return 0;
  if (n > kBytesCapacity) n = kBytesCapacity;
  const uint64_t magic = kBytesMagic, length = n;
  memcpy(reinterpret_cast<void*>(p + kBytesMagicOffset), &magic, sizeof magic);
  memcpy(reinterpret_cast<void*>(p + kBytesLengthOffset), &length,
         sizeof length);
  *reinterpret_cast<uint64_t*>(p + kBlock - sizeof(uint64_t)) =
      kTextHandleMagic;
  return p;
}

// --- the platform trust store ---------------------------------------------
//
// Unity's native TLS client does not carry a CA bundle of its own. It asks the
// platform, and the platform's answer is Java:
//
//   TrustManagerFactory.getDefaultAlgorithm()  -> "X509"
//   TrustManagerFactory.getInstance(algorithm) -> factory
//   factory.init(keystore)                     -> (the default store)
//   factory.getTrustManagers()                 -> TrustManager[]
//   X509TrustManager.getAcceptedIssuers()      -> X509Certificate[]
//   X509Certificate.getEncoded()               -> DER
//
// and the verifier then trusts exactly those issuers. On a device the last two
// read the system CA store. There is no Java here, so the store is the bundle
// the host names in SSL_CERT_FILE (or ARC_TRUST_CA): a port that points its
// traffic at a local server points this at that server's CA, and the same code
// path works unchanged.
//
// Answering these with opaque blocks, as the generic table did, made
// getTrustManagers() a zero-length array. The verifier then had no issuer to
// trust, every handshake failed as "not signed by a trusted CA", and a title
// whose boot begins with a catalog check never got its answer -- which is what
// a dialog saying the host could not be reached actually was.
std::mutex g_trust_lock;
bool g_trust_ready = false;
std::vector<std::vector<unsigned char>> g_trust_certs;  // DER, one per issuer
std::vector<uint64_t> g_trust_cert_handles;
uint64_t g_trust_manager = 0;
uint64_t g_trust_issuers = 0;

// A PEM bundle is the shape the host already has: the same file the OpenSSL
// side of the engine reads. Only the certificate blocks matter, so the base64
// body between the markers is the whole parser.
std::vector<std::vector<unsigned char>> LoadPemCertificates(const char* path) {
  std::vector<std::vector<unsigned char>> out;
  if (!path || !*path) return out;
  std::ifstream in(std::filesystem::path(path), std::ios::binary);
  if (!in) return out;
  const std::string text((std::istreambuf_iterator<char>(in)),
                         std::istreambuf_iterator<char>());
  static const char kBegin[] = "-----BEGIN CERTIFICATE-----";
  static const char kEnd[] = "-----END CERTIFICATE-----";
  size_t at = 0;
  while (true) {
    const size_t begin = text.find(kBegin, at);
    if (begin == std::string::npos) break;
    const size_t body = begin + sizeof kBegin - 1;
    const size_t end = text.find(kEnd, body);
    if (end == std::string::npos) break;
    at = end + sizeof kEnd - 1;
    std::vector<unsigned char> der;
    int acc = 0, bits = 0;
    for (size_t i = body; i < end; ++i) {
      const unsigned char ch = static_cast<unsigned char>(text[i]);
      int v;
      if (ch >= 'A' && ch <= 'Z') v = ch - 'A';
      else if (ch >= 'a' && ch <= 'z') v = ch - 'a' + 26;
      else if (ch >= '0' && ch <= '9') v = ch - '0' + 52;
      else if (ch == '+') v = 62;
      else if (ch == '/') v = 63;
      else continue;  // whitespace and the trailing '='
      acc = (acc << 6) | v;
      bits += 6;
      if (bits >= 8) {
        bits -= 8;
        der.push_back(static_cast<unsigned char>((acc >> bits) & 0xFF));
      }
    }
    if (!der.empty()) out.push_back(std::move(der));
  }
  return out;
}

// Built once. The handles live in the same arena as every other answer, so
// nothing frees them and the same array can be handed out again.
void EnsureTrustStore() {
  std::lock_guard<std::mutex> held(g_trust_lock);
  if (g_trust_ready) return;
  g_trust_ready = true;
  const char* path = std::getenv("ARC_TRUST_CA");
  if (!path || !*path) path = std::getenv("SSL_CERT_FILE");
  g_trust_certs = LoadPemCertificates(path);
  // A handle that carries its class name is an instance of the class of that
  // name -- which is the only hierarchy this bridge can answer for.
  g_trust_manager = AllocateText("javax/net/ssl/X509TrustManager");
  g_trust_cert_handles.reserve(g_trust_certs.size());
  for (size_t i = 0; i < g_trust_certs.size(); ++i)
    g_trust_cert_handles.push_back(
        AllocateText("java/security/cert/X509Certificate"));
  g_trust_issuers = AllocateObjectArray(g_trust_cert_handles.data(),
                                        g_trust_cert_handles.size());
  static const bool trace = TraceJni();
  if (trace)
    std::fprintf(stderr, "[jni] trust store %s: %zu certificate(s)\n",
                 path ? path : "(none)", g_trust_certs.size());
}

// The Nimble SDK's callbacks, which on Android are delivered by Java.
//
// Its C++ half hands Java an object carrying a small integer id, asks Java to
// go and do something, and returns. Java does the work and calls back through
// BaseNativeCallback.nativeCallback(id, result), which finds the id in a map
// and resumes the C++ side. There is no Java here, so nothing ever calls
// back: the C++ waits for an answer that cannot arrive.
//
// That is not a corner case. The anonymous login -- the one that gets a
// player into the game without an account -- asks for a Google SafetyNet
// attestation this way and stops there. No request is made, so nothing fails;
// the Nexus service simply never leaves status Unknown and the title reports
// it as an error.
//
// So the host delivers the callback itself. On another thread and after a
// beat, because that is what Java does and because the C++ side is still
// inside the call that asked: answering synchronously would re-enter it.
uint64_t g_native_callback;

void DeliverNimbleCallback(long id, const char* result) {
  if (!g_native_callback) {
    g_native_callback = arc::ShimGuestSymbol(
        "Java_com_ea_nimble_bridge_BaseNativeCallback_nativeCallback");
    if (!g_native_callback) return;
  }
  // Two elements, which is what the SDK's bridge reads back out: the value
  // and the error beside it. An attestation this host cannot produce is an
  // empty one, and no error -- which is what a device without Play Services
  // reports and what the SDK is written to carry on from.
  const uint64_t items[2] = {AllocateText(result ? result : ""),
                             AllocateText("")};
  const uint64_t answer = AllocateObjectArray(items, 2);
  const uint64_t fn = g_native_callback;
  const uint64_t env = arc_jni_env();
  const uint64_t cls = arc_jni_object();
  const uint64_t args[4] = {env, cls, static_cast<uint64_t>(id), answer};
  // ARC_NIMBLE_CALLBACK=thread delivers it the way Java would, from another
  // thread and a beat later. That is the honest shape and it is also the one
  // that crashes at the moment, so the default is the blunt one: answer
  // before returning, on the thread that asked.
  static const char* how = getenv("ARC_NIMBLE_CALLBACK");
  if (how && strcmp(how, "thread") == 0) {
    std::thread([fn, args] {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      arc_call_guest(fn, args, 4);
    }).detach();
    return;
  }
  arc_call_guest(fn, args, 4);
}

// The id the SDK put in a callback object. Ours carry their text, and for one
// of these the text is the id in decimal -- the same trick an enum's ordinal
// uses, for the same reason: the number belongs to the handle, not to the
// call that later hands it back.
long CallbackIdOf(uint64_t handle) {
  if (!handle || !arc_jni_owns(handle)) return -1;
  const char* text = reinterpret_cast<const char*>(handle);
  if (!*text) return -1;
  char* end = nullptr;
  const long v = strtol(text, &end, 10);
  return (end && *end == '\0') ? v : -1;
}

// All of them, in order, for a call whose interesting argument is not the
// first -- a logger's message behind its title, say.
std::vector<uint64_t> OwnedArgs(Arm64Ctx* c) {
  std::vector<uint64_t> found;
  if (!c->x[3]) return found;
  std::vector<unsigned char> cursor(arc::ShimVaListSize());
  memcpy(cursor.data(), reinterpret_cast<const void*>(c->x[3]), cursor.size());
  for (int i = 0; i < 6; ++i) {
    const uint64_t v = arc::ShimVaNextInt(cursor.data());
    if (v && arc_jni_owns(v)) found.push_back(v);
  }
  return found;
}

uint64_t FirstOwnedArg(Arm64Ctx* c) {
  if (!c->x[3]) return 0;
  std::vector<unsigned char> cursor(arc::ShimVaListSize());
  memcpy(cursor.data(), reinterpret_cast<const void*>(c->x[3]), cursor.size());
  for (int i = 0; i < 6; ++i) {
    const uint64_t v = arc::ShimVaNextInt(cursor.data());
    if (v && arc_jni_owns(v)) return v;
  }
  return 0;
}

// The arguments of a Java call, as the JNI entry point received them.
//
// JNI spells every call three ways and the call site chooses: `...Method`
// passes the arguments directly (in registers, on this ABI), `...MethodV`
// passes a va_list, and `...MethodA` passes a jvalue[]. The JNI index says
// which, so the arguments can be read without guessing -- which matters,
// because the one call that decides the *identity* of every later Java call
// (Unity's `ReflectionHelper.getMethodID`) arrives through the varargs form.
// The three spellings are consecutive per family, but the families do not
// start at the same offset in the table: instance calls begin at 34,
// nonvirtual at 64 and static at 114, so the modulus differs between them. The
// index alone still says which form a call used.
//
// Reading a `...V` call as a jvalue[] is not a near miss: the argument is a
// va_list, not an array of eight-byte slots, so every argument comes back as
// whatever those bytes happen to be. Nabe's `AndroidUtil.Copy(String, String)`
// arrives this way, and two garbage paths read as "the pack could not be
// copied".
int CallForm(size_t index) {
  const size_t base = index >= 114 ? 114 : index >= 64 ? 64 : 34;
  if (index - base >= 30) return -1;
  return static_cast<int>((index - base) % 3);
}

std::vector<uint64_t> CallArgs(Arm64Ctx* c, size_t index, size_t count) {
  std::vector<uint64_t> args;
  const int form = CallForm(index);
  if (form < 0) return args;
  if (form == 0) {
    for (size_t i = 0; i < count && 3 + i < 8; ++i) args.push_back(c->x[3 + i]);
    return args;
  }
  if (form == 1) {
    if (!c->x[3]) return args;
    std::vector<unsigned char> cursor(arc::ShimVaListSize());
    memcpy(cursor.data(), reinterpret_cast<const void*>(c->x[3]),
           cursor.size());
    for (size_t i = 0; i < count; ++i)
      args.push_back(arc::ShimVaNextInt(cursor.data()));
    return args;
  }
  // jvalue[]: a union of eight bytes per element, and the integer/pointer
  // member is where every argument the engine passes lives.
  if (!c->x[3]) return args;
  for (size_t i = 0; i < count; ++i) {
    uint64_t v = 0;
    memcpy(&v, reinterpret_cast<const void*>(c->x[3] + i * 8), sizeof v);
    args.push_back(v);
  }
  return args;
}

// A handle that carries text, or nothing. Class handles and strings both do;
// a blank stand-in does not, and refusing to read one as a string is the
// difference between "the caller named nothing" and "the caller named
// whatever was in the arena at that address".
const char* HandleText(uint64_t handle) {
  if (!handle || !arc_jni_owns(handle)) return nullptr;
  return reinterpret_cast<const char*>(handle);
}

// What an object was constructed with. See `arc_jni_ctor_text`: a request
// object is handed its conversation's identity as the first argument of its
// constructor, and with no Java to keep it the handle has to.
//
// This is a side table and not the handle's own text on purpose. A Java object
// is not a string, and `IsInstanceOf` reads a handle's text as its class name;
// a `File` constructed from a path that then claimed to be that path would
// stop being an instance of `File`. Only a caller that asks for the
// constructor's argument gets it.
void NoteCtorText(uint64_t handle, uint64_t klass, uint64_t first,
                  bool has_first) {
  const char* class_text = HandleText(klass);
  if (!class_text || !*class_text) return;
  std::lock_guard<std::mutex> held(g_ctor_lock);
  CtorRecord& record = g_ctor_text[handle];
  record.klass = class_text;
  if (!has_first) return;
  if (const char* first_text = HandleText(first)) {
    record.first = first_text;
  } else if (first < 1000000) {
    // An id handed over as an integer rather than as a string. It names the
    // same conversation either way, and a caller that asks which one has no
    // other way to find out.
    char text[24];
    snprintf(text, sizeof text, "%llu",
             static_cast<unsigned long long>(first));
    record.first = text;
  }
}

const char* CtorClassFor(uint64_t handle) {
  std::lock_guard<std::mutex> held(g_ctor_lock);
  const auto it = g_ctor_text.find(handle);
  return it == g_ctor_text.end() ? nullptr : it->second.klass.c_str();
}

const char* CtorTextFor(uint64_t handle) {
  std::lock_guard<std::mutex> held(g_ctor_lock);
  const auto it = g_ctor_text.find(handle);
  if (it == g_ctor_text.end() || it->second.first.empty()) return nullptr;
  return it->second.first.c_str();
}

// A port's answers to its title's own Java classes.
//
// The table below answers the engine. A title's Java half is a different
// subject: it belongs to the title, and the port is where the title lives. The
// identity of a call is already recorded with its method id, so a handler can
// be matched on exactly what was resolved, and this file never has to learn a
// class name it was not told.
struct JavaMethodHandler {
  std::string owner;
  std::string name;
  std::string signature;
  ArcJavaMethodFn fn;
};

std::mutex g_java_method_lock;
std::vector<JavaMethodHandler> g_java_methods;

ArcJavaMethodFn FindJavaMethod(const char* owner, const char* name,
                               const char* signature) {
  if (!name) return nullptr;
  std::lock_guard<std::mutex> held(g_java_method_lock);
  for (const JavaMethodHandler& h : g_java_methods) {
    if (h.name != name) continue;
    if (!h.owner.empty() && (!owner || h.owner != owner)) continue;
    if (!h.signature.empty() && (!signature || h.signature != signature))
      continue;
    return h.fn;
  }
  return nullptr;
}

// An exported guest symbol, by name, remembered once found. The loader's
// lookup walks every image's dynamic table, which is not something to do for
// each string handed back to managed code. A miss is not remembered: the
// runtime may simply not have finished loading when a port first asks.
uint64_t GuestExport(const char* name) {
  static std::mutex held_mutex;
  static std::map<std::string, uint64_t> found;
  {
    std::lock_guard<std::mutex> held(held_mutex);
    const auto it = found.find(name);
    if (it != found.end()) return it->second;
  }
  const uint64_t address = arc::ShimGuestSymbol(name);
  if (!address) return 0;
  std::lock_guard<std::mutex> held(held_mutex);
  found.emplace(name, address);
  return address;
}

// A managed string, made by the runtime itself. `il2cpp_string_new` is the
// only way to get an object the collector understands from host text.
uint64_t GuestStringFor(const char* utf8) {
  static const uint64_t create = GuestExport("il2cpp_string_new");
  if (!create) return 0;
  const uint64_t args[1] = {reinterpret_cast<uint64_t>(utf8 ? utf8 : "")};
  return arc_call_guest(create, args, 1);
}

// The image a managed assembly's classes are looked up in. Unity names it
// after the assembly, so the open is a name lookup; a title whose assembly is
// named differently from the file (a stripped or merged build) is found by
// walking the domain instead, because a class lookup that silently answers
// null is a conversation this bridge cannot finish.
uint64_t ManagedImage(const char* assembly) {
  const uint64_t domain_get = GuestExport("il2cpp_domain_get");
  const uint64_t assembly_open = GuestExport("il2cpp_domain_assembly_open");
  const uint64_t assembly_image = GuestExport("il2cpp_assembly_get_image");
  if (!domain_get || !assembly_open || !assembly_image) return 0;
  const uint64_t no_args[1] = {0};
  const uint64_t domain = arc_call_guest(domain_get, no_args, 1);
  if (!domain) return 0;
  const uint64_t open_args[2] = {domain, reinterpret_cast<uint64_t>(assembly)};
  uint64_t found = arc_call_guest(assembly_open, open_args, 2);
  const uint64_t list_assemblies = GuestExport("il2cpp_domain_get_assemblies");
  const uint64_t image_name = GuestExport("il2cpp_image_get_name");
  if (found) {
    const uint64_t image_args[1] = {found};
    return arc_call_guest(assembly_image, image_args, 1);
  }
  if (!list_assemblies || !image_name) return 0;
  size_t count = 0;
  const uint64_t list_args[2] = {domain,
                                 reinterpret_cast<uint64_t>(&count)};
  const uint64_t list = arc_call_guest(list_assemblies, list_args, 2);
  if (!list) return 0;
  const size_t want = strlen(assembly);
  for (size_t i = 0; i < count; ++i) {
    const uint64_t entry = arc_ld64(list + i * 8);
    if (!entry) continue;
    const uint64_t image_args[1] = {entry};
    const uint64_t image = arc_call_guest(assembly_image, image_args, 1);
    if (!image) continue;
    const uint64_t name_args[1] = {image};
    const char* name =
        reinterpret_cast<const char*>(arc_call_guest(image_name, name_args, 1));
    if (!name) continue;
    if (strncmp(name, assembly, want) != 0) continue;
    const char after = name[want];
    if (after == '\0' || after == '.') return image;
  }
  return 0;
}

// What a method returning an integer should answer with. Zero is the default
// and is usually harmless; these are the ones where it is a specific, wrong
// claim about the device.
struct MethodNumber {
  const char* name;
  int64_t value;
};

const MethodNumber kMethodNumbers[] = {
    // Free disk space, in bytes. The engine is about to download content, so
    // it checks for room before it starts loading and quietly refuses below
    // about 40 MB. Zero means "the disk is full": the loading manager goes to
    // a phase that never initialises the font system, every font cache lookup
    // afterwards answers null, and the first one whose caller does not check
    // takes the process down. Four gigabytes free is an ordinary answer.
    {"getFreeDiskSpace", 4LL << 30},
    // Memory, in bytes. Zero is the answer that says "this device is out",
    // and a title that believes it drops to a reduced-texture path or refuses
    // to load at all. It is also not a number to be shy with: the engine sizes
    // its asset caches from it, and a small answer means a cache that evicts
    // while the game is still reading what it evicted. Three gigabytes is an
    // ordinary answer for a device that could run this, and this host has more.
    {"getAvailableMemory", 3LL << 30},
    {"getFreeMemory", 3LL << 30},
    // A layout inset in pixels. Nothing here has a notch or a rounded corner
    // to inset for, so none is the right answer rather than a missing one.
    {"getHorizontalMargin", 0},
    // GetUptime is not here: it is a clock, and a clock that answers the same
    // number twice is not one. See Uptime() below.
};

// Milliseconds since the host started, which is what the engine is measuring
// when it asks. A constant would be worse than useless: the callers that
// matter subtract two readings, and a constant makes every interval zero --
// so a frame takes no time, a timeout never expires, and a rate limiter lets
// everything through at once.
int64_t Uptime() {
  // The same clock CLOCK_MONOTONIC answers from, because on Android they are
  // the same clock: SystemClock.uptimeMillis() is CLOCK_MONOTONIC in
  // milliseconds. This used to keep its own origin and add a minute to it, on
  // the reasoning that a device has usually been up a while -- which is true
  // of a device and false of a stopwatch two pieces of the same program are
  // sharing. Code that started a timer through one and read it through the
  // other measured a minute of elapsed time immediately, so every network
  // timeout expired the moment it was set.
  return static_cast<int64_t>(arc::ShimMonotonicMillis());
}

int64_t NumberForMethod(uint64_t id, bool* answered) {
  *answered = false;
  const char* name = MethodName(id);
  if (!name) return 0;
  NoteCalled(name);
  if (strcmp(name, "GetUptime") == 0 || strcmp(name, "getUptime") == 0) {
    *answered = true;
    return Uptime();
  }
  for (const MethodNumber& m : kMethodNumbers)
    if (strcmp(m.name, name) == 0) {
      *answered = true;
      return m.value;
    }
  NoteUnanswered(name);
  return 0;
}

// Preferences, which are read by key rather than by method: getSharedPreference
// is one Java method with forty-odd different questions behind it, so the
// answer has to come from the argument.
//
// The one that matters is where the game's own server lives. On Android that
// is patched into the library by hash and offset before installation; here it
// is a host setting, which is the point of doing this natively at all. Set
// ARC_SERVER to the base URL -- "http://127.0.0.1:9000" for a loopback sidecar
// -- and the engine asks that server for its directory instead of nothing.
// Every key asked for and what it was answered with. getSharedPreference is
// reported as one unanswered method however many questions went through it,
// which is the least useful thing the report could say about the call that
// decides where the game looks for its server. ARC_TRACE_PREFS=1.
void TracePreference(const char* key, const char* value) {
  static const bool on = getenv("ARC_TRACE_PREFS") != nullptr;
  if (on)
    fprintf(stderr, "[pref] %-40s -> %s\n", key, value ? value : "(no answer)");
}

// Where a preference actually lives.
//
// Everything a title and its SDKs write down between runs comes through two
// Java calls -- one to set a value, one to read it -- and a host that answers
// neither has no memory at all. Every run is then a first run: no account, no
// token, no anonymous id, no age check, nothing carried over. The SDK's own
// log says so plainly and repeatedly ("Data not found in persistence") and
// the title reads the sum of it as a failure to sign in.
//
// A flat file of key=value beside the title's own data, which is where
// Android would have put it. One store rather than one per component: the
// keys are already distinct enough to have collided by now if they were
// going to, and a scheme that puts each component's file somewhere different
// has to know the component, which the Java call does not tell us.
std::mutex g_prefs_lock;
std::map<std::string, std::string> g_prefs;
bool g_prefs_loaded = false;

std::string PrefsPath() {
  const char* root = arc::ShimAssetRoot();
  return std::string(root && *root ? root : ".") + "/prefs.txt";
}

void LoadPrefs() {
  if (g_prefs_loaded) return;
  g_prefs_loaded = true;
  std::ifstream in(PrefsPath(), std::ios::binary);
  std::string key;
  // "key length value", with the length counted rather than the value
  // escaped. A stored value is whatever the title wrote -- a token, a JSON
  // document, something with a newline in it -- and a format that has to
  // escape is a format that can hand back something subtly different from
  // what it was given.
  while (in >> key) {
    size_t n = 0;
    if (!(in >> n)) break;
    in.get();  // the single space before the value
    std::string value(n, 0);
    in.read(&value[0], static_cast<std::streamsize>(n));
    in.get();  // the newline after it
    g_prefs[key] = value;
  }
}

void SavePrefs() {
  std::ofstream out(PrefsPath(), std::ios::binary | std::ios::trunc);
  for (const auto& kv : g_prefs)
    out << kv.first << " " << kv.second.size() << " " << kv.second << std::endl;
}

void SetPreference(const char* key, const char* value) {
  if (!key || !*key) return;
  std::lock_guard<std::mutex> held(g_prefs_lock);
  LoadPrefs();
  g_prefs[key] = value ? value : "";
}

void SyncPreferences() {
  std::lock_guard<std::mutex> held(g_prefs_lock);
  if (g_prefs_loaded) SavePrefs();
}

const char* StoredPreference(const char* key) {
  std::lock_guard<std::mutex> held(g_prefs_lock);
  LoadPrefs();
  auto it = g_prefs.find(key);
  return it == g_prefs.end() ? nullptr : it->second.c_str();
}

const char* PreferenceValue(const char* key) {
  if (!key || !*key) return nullptr;
  // What was written down last time wins over any default below it: a title
  // that stored a value expects to read that value back, and a default that
  // overrode it would make the store look broken in exactly the cases it
  // exists for.
  if (const char* v = StoredPreference(key)) {
    TracePreference(key, v);
    return v;
  }
  // A title's preference keys are its own. Tapped Out asks for none of the
  // four below -- it asks for DLCSource, DLCLocation, DLCSecretKey and
  // ServerEnvironment -- and the next title will ask for others again.
  // ARC_PREF_<key> answers any of them, which is what turns a "(no answer)"
  // in the trace into something that can be tried without a rebuild.
  {
    char name[96];
    if (snprintf(name, sizeof name, "ARC_PREF_%s", key) <
        static_cast<int>(sizeof name)) {
      if (const char* v = getenv(name)) {
        TracePreference(key, v);
        return v;
      }
    }
  }
  static const char* const kServerKeys[] = {"MayhemServerURL", "MayhemURL",
                                            "ServerURL", "DirectorURL"};
  for (const char* k : kServerKeys)
    if (strcmp(k, key) == 0) {
      static const char* v = getenv("ARC_SERVER");
      TracePreference(key, v);
      return v;
    }
  // The stored language, which is a preference rather than the device setting
  // LocaleOverride answers -- the engine reads both and they are allowed to
  // differ. Same knob, because a host that has one has no reason to have two.
  if (strcmp(key, "language") == 0) {
    static const char* v = getenv("ARC_LANG");
    TracePreference(key, v ? v : "en");
    return v ? v : "en";
  }
  TracePreference(key, nullptr);
  return nullptr;
}

// The engine's own log, which on Android goes to Java rather than to
// __android_log. These are the names it writes through; anything else that
// returns void stays quiet.
bool IsLogMethod(const char* name) {
  static const char* const kNames[] = {"Log",      "log",     "LogError",
                                       "logError", "LogInfo", "printLog",
                                       "writeWithTitle", "LogToFile"};
  for (const char* n : kNames)
    if (strcmp(n, name) == 0) return true;
  return false;
}

// What a method returning a boolean should answer with. False is the safe
// default -- "no notification launched us", "no other music is playing" -- and
// it is what an unlisted name gets. These are the ones where false is not
// merely conservative but wrong.
struct MethodFlag {
  const char* name;
  bool value;
};

const MethodFlag kMethodFlags[] = {
    // Handled by ThreadComplete() below rather than here, because "how long
    // did the thread take" is a real variable and answering it instantly is
    // not obviously the honest answer. Listed so the name is not reported as
    // unanswered.
    {"isThreadComplete", true},
    // mkdir is answered in the boolean slot above, by actually creating the
    // directory. Listed so the name is not reported as unanswered.
    {"mkdir", true},
    // Reachability is a question the engine asks Java, not the network: it
    // never opens a socket to find out. Answered false, it never tries at all
    // -- it goes straight to its error state and offers to retry, which is a
    // correct-looking screen produced without a single connect(). The host has
    // a network; say so, and let the connection succeed or fail on its own.
    {"hasConnectivity", true},
    // A preference nobody has ever set is false, and that is a real answer
    // rather than a missing one. Recorded so the report stops listing it.
    {"getBooleanPreference", false},
    // No previous install, so nothing was carried over and there is no saved
    // state to migrate.
    {"isFirstRun", true},
    // Neither is true of a host launched from a command line: there is no
    // notification to have been created or resumed by, and no music app.
    {"wasCreatedViaLocalNotification", false},
    {"wasResumedViaLocalNotification", false},
    {"isOtherMusicPlaying", false},
    // A machine on a wire is not on a metered connection, and the title asks
    // before it downloads anything large.
    {"UsingMobileData", false},
    // On a wire, which for a title deciding whether it may download is the
    // same answer wifi gives.
    {"isNetworkWifi", true},
    // A stock, unmodified install on a machine that is not a phone. These are
    // anti-tamper questions with one true answer here, and "no answer" is not
    // it: a title that cannot tell whether it has been tampered with is
    // entitled to assume the worst.
    {"isDeviceRooted", false},
    {"isAppCracked", false},
    // Whether an override exists for a config value. Nothing here overrides
    // any, and saying so is different from not answering.
    {"configValueExists", false},
    // Nimble asks its Java half whether the Synergy environment -- the
    // directory of EA services it was configured with -- has been fetched.
    // Answered false, its Nexus service never leaves status Unknown, the
    // title reads that as an unrecoverable error, and nothing has been tried:
    // no request is made, so nothing can fail informatively. Here the
    // environment is a fixed loopback address that is always available.
    {"isDataAvailable", true},
    // There is no Play Core on this host and no Play Asset Delivery service to
    // report pack state or call a download listener back. Answering false says
    // the service exists, after which the engine registers a listener and waits
    // for a status callback that nothing will ever deliver -- the download screen
    // holds at zero with the catalog already on disk. This is the same answer a
    // device without Play Core gives, and it selects Unity's own HTTP path.
    {"playCoreApiMissing", true},
};

// Tri-state on purpose: "not in the table" has to stay distinguishable from
// "in the table, answering false", or every unanswered call would be recorded
// as a deliberate no.
// The loading screen polls a Java worker thread and will not advance until it
// says it is finished. There is no Java here and no work outstanding, so the
// truthful answer is "finished" -- but on a device that thread takes real
// time, and the engine advances its own state machine while it waits. Saying
// yes on the very first poll compresses that to nothing, which is a different
// order of events than the engine was written against.
//
// So it is a knob, defaulting to answering immediately: ARC_THREAD_POLLS is
// how many polls to answer "still running" first. This is the sort of thing a
// real device varies and a host cannot infer.
bool ThreadComplete() {
  static const long wait = [] {
    const char* s = getenv("ARC_THREAD_POLLS");
    return s ? strtol(s, nullptr, 10) : 0;
  }();
  static std::atomic<long> polls{0};
  return polls++ >= wait;
}

bool FlagForMethod(Arm64Ctx* c, uint64_t id, bool* answered) {
  *answered = false;
  const char* name = MethodName(id);
  if (!name) return false;
  NoteCalled(name);
  if (strcmp(name, "isThreadComplete") == 0) {
    *answered = true;
    return ThreadComplete();
  }
  // "Is this feature switched off?" is the one unanswered question whose
  // default is not neutral. Answering false says every feature the title asks
  // about is on -- including the seasonal ones, which is how a September boot
  // ends up covered in snow. What the name is matters, so it is logged.
  if (strcmp(name, "isFeatureDisabled") == 0) {
    static const char* const how = getenv("ARC_JNI_FEATURES");
    const bool off = how && strcmp(how, "off") == 0;
    if (getenv("ARC_TRACE_FEATURES")) {
      // A handle is a pointer to its own text, which is what makes the name
      // readable here at all.
      const uint64_t arg = FirstOwnedArg(c);
      const char* which = arg ? reinterpret_cast<const char*>(arg) : nullptr;
      fprintf(stderr, "[feature] isFeatureDisabled(%s) -> %s\n",
              which && *which ? which : "?", off ? "true" : "false");
    }
    *answered = true;
    return off;
  }
  for (const MethodFlag& m : kMethodFlags)
    if (strcmp(m.name, name) == 0) {
      *answered = true;
      return m.value;
    }
  NoteUnanswered(name);
  return false;
}

// The two the engine turns into directories, which have to be real paths on
// this machine rather than anything invented. They come from the asset root
// the host was given: the bundle is that directory, and what the engine writes
// goes beside it rather than into it.
// The device's locale, which is a real property of a real device and not
// something a host can infer. The engine matches the language string against a
// table of twenty-two it knows and takes the index; a string it does not know
// is index zero, which is also what "English" is, and the two are not
// distinguishable downstream.
//
// So it is a knob. ARC_LANG and ARC_LOCALE override what the JNI bridge
// answers, which is how you find out what a title does with a locale other
// than the one you guessed.
const char* LocaleOverride(const char* name) {
  if (strcmp(name, "language") == 0 || strcmp(name, "getLanguage") == 0 ||
      strcmp(name, "getCurrentLanguage") == 0 ||
      strcmp(name, "getApplicationLanguageCode") == 0) {
    static const char* v = getenv("ARC_LANG");
    return v;
  }
  if (strcmp(name, "locale") == 0) {
    static const char* v = getenv("ARC_LOCALE");
    return v;
  }
  return nullptr;
}

// The version the title believes it is, which is a property of the APK and not
// of this kit -- so it is a knob with a neutral default rather than a value
// invented here. ARC_APP_VERSION, in the three components the parser needs.
//
// It is not cosmetic. A title asks its content server for an index and gets
// back a list of entries tagged with the version each belongs to; it then picks
// the one its own version selects. Claiming 1.0.0 against a catalogue that
// starts at 4.x selects nothing, and the loading screen waits forever for a
// download it never asked for -- with the server showing a clean 200 for the
// index and nothing after it.
// Identity the title asked Java for after it had just been told it by the
// server and written it down itself. getCurrentSynergyId is the telemetry id
// read back, and answering nothing there leaves the loading task list waiting
// on a value it is holding in its own preferences.
const char* IdentityOverride(const char* name) {
  if (strcmp(name, "getCurrentSynergyId") == 0 ||
      strcmp(name, "getSynergyId") == 0)
    return StoredPreference("telemetryId");
  // No override, which is a real answer: the id is whatever the server said.
  if (strcmp(name, "getOverrideSynergyId") == 0) return "";
  if (strcmp(name, "getUniqueDeviceID") == 0 ||
      strcmp(name, "getCurrentEADeviceId") == 0 ||
      strcmp(name, "getEADeviceId") == 0 ||
      strcmp(name, "getEAHardwareId") == 0) {
    if (const char* v = StoredPreference("deviceIdentifier")) return v;
    return StoredPreference("androidId");
  }
  return nullptr;
}

const char* VersionOverride(const char* name) {
  if (strcmp(name, "appVersion") == 0 || strcmp(name, "clientVersion") == 0 ||
      strcmp(name, "versionName") == 0 || strcmp(name, "getVersionName") == 0 ||
      strcmp(name, "getVersion") == 0 || strcmp(name, "deviceVersion") == 0 ||
      strcmp(name, "getBundleVersion") == 0 ||
      strcmp(name, "getApplicationVersion") == 0) {
    static const char* v = getenv("ARC_APP_VERSION");
    return v;
  }
  return nullptr;
}

const char* DirectoryForName(const char* name) {
  static std::mutex lock;
  static std::string bundle, storage;
  const char* root = arc::ShimAssetRoot();
  if (!root || !*root) return nullptr;
  std::lock_guard<std::mutex> held(lock);
  // Android hands out Unix paths, and the managed Path class uses Unix rules.
  // An absolute Windows path here is read as a relative one, so Path.Combine
  // appends it instead of replacing -- and a caller that combines two of these
  // answers ends up with the content root twice in the path. The virtual tree
  // in HostPath is the boundary; every directory answer belongs under it.
  if (strcmp(name, "getBundleDir") == 0 || strcmp(name, "getAssetsPath") == 0 ||
      strcmp(name, "getCocos2dxWritablePath") == 0 ||
      strcmp(name, "appPath") == 0) {
    bundle = "/androidrecomp/assets";
    return bundle.c_str();
  }
  if (strcmp(name, "getStorageDir") == 0 ||
      strcmp(name, "getCachePath") == 0 ||
      strcmp(name, "writablePath") == 0 ||
      strcmp(name, "getFilesDir") == 0 ||
      strcmp(name, "getDataDir") == 0 ||
      strcmp(name, "getExternalFilesDir") == 0 ||
      strcmp(name, "getNoBackupFilesDir") == 0 ||
      strcmp(name, "getCacheDir") == 0 ||
      strcmp(name, "getCodeCacheDir") == 0 ||
      strcmp(name, "getExternalCacheDir") == 0 ||
      strcmp(name, "getDownloadCacheDir") == 0 ||
      strcmp(name, "getObbDir") == 0 ||
      strcmp(name, "getExternalStorageDirectory") == 0 ||
      strcmp(name, "getExternalStoragePublicDirectory") == 0) {
    // The same directory, not the one above it. Android hands a title one data
    // directory and it both reads its content out of that and writes beside
    // it: Tapped Out builds "<writablePath>/core/res-core/UberShader.vsh" and
    // "<writablePath>/prefbackup" from the same string. Answering with the
    // parent put every shipped file one directory out of reach, which shows up
    // not as a missing file but as an empty shader that links to nothing and
    // draws a black screen behind a full set of draw calls.
    storage = "/androidrecomp/storage";
    return storage.c_str();
  }
  return nullptr;
}

// ClassLoader.findLibrary asks Java where a native image lives. Unity reaches
// for "il2cpp" before it initialises the scripting backend, and an empty answer
// is not "unknown" to it: it is a library path of length zero, which then goes
// to dlopen and fails the whole backend. Build the name Android would have
// under the staged lib directory instead.
std::string LibraryPathFor(const char* asked) {
  std::string name = asked ? asked : "";
  if (name.empty()) return {};
  const bool prefixed = name.rfind("lib", 0) == 0;
  const bool suffixed = name.size() >= 3 &&
                        name.compare(name.size() - 3, 3, ".so") == 0;
  if (!prefixed) name = "lib" + name;
  if (!suffixed) name += ".so";
  const char* root = arc::ShimLibraryRoot();
  if (root && *root)
    return (std::filesystem::path(root) / name).generic_string();
  return name;
}

const char* TextForMethod(uint64_t id) {  const char* name = MethodName(id);
  if (!name) return nullptr;
  NoteCalled(name);
  // ARC_JNI_<method> answers any string-returning Java method, the same way
  // ARC_PREF_<key> answers any preference. The values these carry -- a client
  // id, a shared secret, an account identifier -- belong to a title and its
  // server, not to this kit, and a table here would be the wrong place to
  // keep them. It also turns the report's "had no value to return" list into
  // something that can be tried one name at a time without a rebuild.
  {
    char var[96];
    if (snprintf(var, sizeof var, "ARC_JNI_%s", name) <
        static_cast<int>(sizeof var)) {
      if (const char* v = getenv(var)) return v;
    }
  }
  if (const char* v = LocaleOverride(name)) return v;
  if (const char* v = IdentityOverride(name)) return v;
  if (const char* v = VersionOverride(name)) return v;
  if (const char* dir = DirectoryForName(name)) return dir;
  for (const MethodText& m : kMethodText)
    if (strcmp(m.name, name) == 0) return m.text;
  NoteUnanswered(name);
  return nullptr;
}

// --- slot behaviour --------------------------------------------------------

struct SlotInfo {
  int index;
  const char* name;
  bool returns_handle;
};

constexpr SlotInfo kKnown[] = {
    {4, "GetVersion", false},           {5, "DefineClass", true},
    {6, "FindClass", true},             {10, "GetSuperclass", true},
    {13, "Throw", false},               {14, "ThrowNew", false},
    {15, "ExceptionOccurred", false},   {16, "ExceptionDescribe", false},
    {17, "ExceptionClear", false},      {19, "PushLocalFrame", false},
    {20, "PopLocalFrame", true},        {21, "NewGlobalRef", true},
    {22, "DeleteGlobalRef", false},     {23, "DeleteLocalRef", false},
    {24, "IsSameObject", false},        {25, "NewLocalRef", true},
    {27, "AllocObject", true},          {28, "NewObject", true},
    {29, "NewObjectV", true},           {30, "NewObjectA", true},
    {31, "GetObjectClass", true},       {32, "IsInstanceOf", false},
    {33, "GetMethodID", true},          {34, "CallObjectMethod", true},
    {35, "CallObjectMethodV", true},    {61, "CallVoidMethod", false},
    {94, "GetFieldID", true},
    {95, "GetObjectField", true},       {96, "GetBooleanField", false},
    {100, "GetIntField", false},        {101, "GetLongField", false},
    {102, "GetFloatField", false},
    // The nine that were missing, and whose absence shifted every index after
    // them nine places low.
    {104, "SetObjectField", false},     {105, "SetBooleanField", false},
    {106, "SetByteField", false},       {107, "SetCharField", false},
    {108, "SetShortField", false},      {109, "SetIntField", false},
    {110, "SetLongField", false},       {111, "SetFloatField", false},
    {112, "SetDoubleField", false},     {113, "GetStaticMethodID", true},
    {115, "CallStaticObjectMethodV", true},
    {37, "CallBooleanMethod", false},
    {38, "CallBooleanMethodV", false},
    {39, "CallBooleanMethodA", false},
    {117, "CallStaticBooleanMethod", false},
    {118, "CallStaticBooleanMethodV", false},
    {119, "CallStaticBooleanMethodA", false},
    {130, "CallStaticIntMethodV", false},
    {142, "CallStaticVoidMethodV", false},
    {144, "GetStaticFieldID", true},    {145, "GetStaticObjectField", true},
    {150, "GetStaticIntField", false},  {163, "NewString", true},
    {164, "GetStringLength", false},    {165, "GetStringChars", true},
    {166, "ReleaseStringChars", false}, {167, "NewStringUTF", true},
    {168, "GetStringUTFLength", false}, {169, "GetStringUTFChars", true},
    {170, "ReleaseStringUTFChars", false},
    {171, "GetArrayLength", false},     {172, "NewObjectArray", true},
    {173, "GetObjectArrayElement", true},
    {174, "SetObjectArrayElement", false},
    {175, "NewBooleanArray", true},     {176, "NewByteArray", true},
    {177, "NewCharArray", true},        {178, "NewShortArray", true},
    {179, "NewIntArray", true},         {180, "NewLongArray", true},
    {181, "NewFloatArray", true},       {182, "NewDoubleArray", true},
    {183, "GetBooleanArrayElements", true},
    {184, "GetByteArrayElements", true},
    {185, "GetCharArrayElements", true},
    {186, "GetShortArrayElements", true},
    {187, "GetIntArrayElements", true},
    {188, "GetLongArrayElements", true},
    {189, "GetFloatArrayElements", true},
    {190, "GetDoubleArrayElements", true},
    {208, "SetByteArrayRegion", false}, {211, "SetIntArrayRegion", false},
    {215, "RegisterNatives", false},    {219, "GetJavaVM", false},
    {222, "GetPrimitiveArrayCritical", true},
    {223, "ReleasePrimitiveArrayCritical", false},
    {226, "NewWeakGlobalRef", true},
    // ExceptionCheck must answer false, or the engine believes a throw is
    // pending after every call and unwinds instead of continuing.
    {228, "ExceptionCheck", false},     {229, "NewDirectByteBuffer", true},

    // The invocation interface, offset past the environment's slots.
    {kVmBase + 3, "DestroyJavaVM", false},
    {kVmBase + 4, "AttachCurrentThread", false},
    {kVmBase + 5, "DetachCurrentThread", false},
    {kVmBase + 6, "GetEnv", false},
    {kVmBase + 7, "AttachCurrentThreadAsDaemon", false},
};

const SlotInfo* Lookup(size_t index) {
  for (const SlotInfo& s : kKnown)
    if (static_cast<size_t>(s.index) == index) return &s;
  return nullptr;
}

const char* NameOf(size_t index) {
  const SlotInfo* s = Lookup(index);
  return s ? s->name : "JNI slot";
}

// Every slot takes the context, so it can read its arguments and put a result
// in the right register. A float return lives in v0, which an
// integer-returning thunk has no way to express.
// Tracks which JNI slot is running, so an allocation made on its behalf can
// say who asked for it. Nesting is real (a slot can call back into guest code
// that reaches another JNI call), so the previous value is restored.
struct SlotScope {
  size_t previous;
  explicit SlotScope(size_t index) : previous(g_active_slot) {
    g_active_slot = index;
  }
  ~SlotScope() { g_active_slot = previous; }
};

// Answers a method called on one of our synthetic MotionEvents.
//
// The body Unity registers as `nativeInjectEvent` reads the event entirely
// through JNI, so every one of these questions is on the path between a host
// mouse and the engine's input state. Where the answer belongs is a property of
// the method's return type -- a float leaves in v0, everything else in x0 -- and
// getting that wrong turns a tap into a point the engine cannot place.
bool AnswerMotionEvent(Arm64Ctx* c, size_t index) {
  MotionEventState state;
  if (!c->x[2] || !MotionEventStateFor(c->x[1], &state)) return false;
  const char* name = MethodName(c->x[2]);
  if (!name) return false;

  // Where the answer goes is a property of the method's return type, and the
  // signature is the only place the engine recorded it. Slot numbers depend on
  // the order names happened to be resolved in, so they cannot carry it; they
  // stay only as the fallback for an id created without a signature.
  const char ret = MethodReturnType(c->x[2]);
  const bool float_result =
      ret ? ret == 'F'
          : (index >= 55 && index <= 57) || (index >= 135 && index <= 137);
  static const bool trace_touch = getenv("ARC_INPUT_TRACE") != nullptr;
  const auto number = [&](double value) {
    if (float_result)
      arc_s_w(c, 0, static_cast<float>(value));
    else
      c->x[0] = static_cast<uint64_t>(static_cast<int64_t>(value));
    // A touch that arrives but is never acted on is either an event the engine
    // did not recognise or a field it read as a different value. Naming the
    // field, its answer and where the answer went tells the two apart.
    if (trace_touch &&
        (strcmp(name, "getAction") == 0 ||
         strcmp(name, "getActionMasked") == 0 || strcmp(name, "getX") == 0 ||
         strcmp(name, "getY") == 0 || strcmp(name, "getSource") == 0 ||
         strcmp(name, "getPointerCount") == 0 ||
         strcmp(name, "getEventTime") == 0 ||
         strcmp(name, "getDownTime") == 0)) {
      fprintf(stderr, "[touch] %s -> %.3f (%s)\n", name, value,
              ret ? (ret == 'F' ? "float" : "int") : "slot-guess");
    }
  };
  // One finger, always index 0. Android reports the full action word here and
  // leaves masking to getActionMasked, so both are answered from the same
  // field rather than one being derived from the other.
  if (strcmp(name, "getAction") == 0) return number(state.action), true;
  if (strcmp(name, "getActionMasked") == 0)
    return number(state.action & 0xff), true;
  if (strcmp(name, "getActionIndex") == 0)
    return number((state.action >> 8) & 0xff), true;
  if (strcmp(name, "getPointerCount") == 0) return number(1), true;
  if (strcmp(name, "getPointerId") == 0) return number(0), true;
  if (strcmp(name, "getX") == 0 || strcmp(name, "getRawX") == 0)
    return number(state.x), true;
  if (strcmp(name, "getY") == 0 || strcmp(name, "getRawY") == 0)
    return number(state.y), true;
  if (strcmp(name, "getEventTime") == 0)
    return number(static_cast<double>(state.event_time)), true;
  if (strcmp(name, "getDownTime") == 0)
    return number(static_cast<double>(state.down_time)), true;
  // InputDevice.SOURCE_TOUCHSCREEN. The engine tests the source to decide
  // whether an event is a touch at all, so zero here is an event from nowhere.
  if (strcmp(name, "getSource") == 0) return number(0x1002), true;
  if (strcmp(name, "getDeviceId") == 0) return number(0), true;
  if (strcmp(name, "getFlags") == 0 || strcmp(name, "getMetaState") == 0 ||
      strcmp(name, "getButtonState") == 0 ||
      strcmp(name, "getEdgeFlags") == 0 ||
      strcmp(name, "getHistorySize") == 0)
    return number(0), true;
  // A finger is the whole contact: full pressure, full size, tool type FINGER.
  // A zero size or pressure makes some engines discard the sample as noise.
  if (strcmp(name, "getPressure") == 0 || strcmp(name, "getSize") == 0)
    return number(1.0), true;
  // The contact ellipse, in pixels. A mouse has no area, but zero is what a
  // cancelled or synthetic event reports, and the engine treats it as noise.
  if (strcmp(name, "getTouchMajor") == 0 ||
      strcmp(name, "getTouchMinor") == 0)
    return number(1.0), true;
  if (strcmp(name, "getToolType") == 0) return number(1), true;
  if (strcmp(name, "getAxisValue") == 0 ||
      strcmp(name, "getOrientation") == 0)
    return number(0.0), true;
  if (strcmp(name, "getPrecisionX") == 0 ||
      strcmp(name, "getPrecisionY") == 0)
    return number(1.0), true;
  // No history: every event is the current sample, so a historical query
  // answers with it rather than with nothing.
  if (strcmp(name, "getHistoricalX") == 0) return number(state.x), true;
  if (strcmp(name, "getHistoricalY") == 0) return number(state.y), true;
  if (strcmp(name, "getHistoricalEventTime") == 0)
    return number(static_cast<double>(state.event_time)), true;
  if (strcmp(name, "getHistoricalSize") == 0 ||
      strcmp(name, "getHistoricalPressure") == 0)
    return number(1.0), true;
  if (strcmp(name, "isRecycled") == 0) return number(0), true;
  if (strcmp(name, "recycle") == 0) return c->x[0] = 0, true;
  if (strcmp(name, "getClass") == 0) {
    c->x[0] = AllocateText("android/view/MotionEvent");
    return true;
  }
  if (strcmp(name, "toString") == 0) {
    c->x[0] = AllocateText("MotionEvent");
    return true;
  }
  return false;
}

void Handle(size_t index, Arm64Ctx* c) {
  SlotScope slot_scope(index);
  if (index < kSlots) ++g_hits[index];
  ++g_total;

  // The three slots that carry a name. A frame that repeats the same
  // lookups is asking for something it never gets, and the only way to
  // know which thing is to read the names it asks by.
  {
    static const bool trace = TraceJni();
    if (trace) {
      const char* what = index == 6 ? "FindClass"
                       : index == 33 ? "GetMethodID"
                       : index == 113 ? "GetStaticMethodID"
                       : index == 167 ? "NewStringUTF"
                       : index == 94 ? "GetFieldID" : nullptr;
      if (what) {
        const uint64_t reg = (index == 6 || index == 167) ? c->x[1] : c->x[2];
        fprintf(stderr, "[jni] %s %.72s\n", what,
                reg ? reinterpret_cast<const char*>(reg) : "(null)");
      }
      // A call through an id says nothing on its own. The name behind the
      // id is the whole content of the event: which Java method the engine
      // reached for, and therefore what it believes it is doing.
      if (index >= 34 && index <= 143) {
        if (const char* called = MethodName(c->x[2])) {
          const char* owner = MethodOwner(c->x[2]);
          const char* sig = MethodSignature(c->x[2]);
          fprintf(stderr, "[jni] %s -> %s%s%s%s lr=%#llx slot=%zu\n", NameOf(index),
                  owner ? owner : "?", owner ? "." : "", called, sig ? sig : "",
                  static_cast<unsigned long long>(ARC_X_R(c, 30)), index);
        }
      }
    }
  }

  // A port's answer, before the built-in table. A call names the class and
  // method its id was resolved with, so whoever registered an answer has
  // already said which methods it understands; nothing below this point needs
  // to know that a title's Java half exists at all.
  if (index >= 34 && index <= 143 && c->x[2]) {
    const char* called = MethodName(c->x[2]);
    if (called) {
      const char* owner = MethodOwner(c->x[2]);
      const char* signature = MethodSignature(c->x[2]);
      if (ArcJavaMethodFn fn = FindJavaMethod(owner, called, signature)) {
        const std::vector<uint64_t> args = CallArgs(c, index, 8);
        uint64_t answer = 0;
        if (fn(owner, called, signature, c->x[1], args.data(), args.size(),
               &answer)) {
          c->x[0] = answer;
          return;
        }
      }
    }
  }

  // A touch event, answered from the fields the host set when it made it. This
  // sits before the built-in table because the same method names mean
  // different things on different receivers -- "getX" on a MotionEvent is a
  // coordinate, and a table keyed by name alone would answer it for anything.
  if (index >= 34 && index <= 143 && AnswerMotionEvent(c, index)) return;

  // References. A reference in this environment *is* the object: the arena
  // never frees and no local frame is ever walked, so a global or local ref is
  // the same handle. Returning a fresh empty block here loses whatever the
  // object was -- findLibrary's answer is immediately passed to NewGlobalRef,
  // and a blank block there turns a library path into the empty string.
  if (index == 20 || index == 21 || index == 25) {
    c->x[0] = c->x[1];
    return;
  }
  if (index == 19) {  // PushLocalFrame
    c->x[0] = 0;
    return;
  }
  if (index == 24) {  // IsSameObject
    c->x[0] = c->x[1] == c->x[2] ? 1 : 0;
    return;
  }

  switch (index) {
    case 31: {  // GetObjectClass(env, object)
      // A class handle is a name, and the only objects here that have one are
      // the synthetic touch events. Everything else keeps the old answer -- a
      // blank block -- because a string handle's text is its contents, not the
      // class of the string.
      if (MotionEventStateFor(c->x[1], nullptr)) {
        c->x[0] = AllocateText("android/view/MotionEvent");
        return;
      }
      c->x[0] = Allocate();
      return;
    }
    case 94: {  // GetFieldID(env, class, name, signature)
      c->x[0] = FieldIdFor(reinterpret_cast<const char*>(c->x[2]),
                           reinterpret_cast<const char*>(c->x[3]));
      return;
    }
    case 95: {  // GetObjectField(env, object, fieldID)
      const Field* f = FieldFromId(c->x[2]);
      if (f && f->kind == 's') {
        // A path field has to name a directory that exists on this machine, so
        // it comes from the asset root rather than from the table.
        if (const char* dir = DirectoryForName(f->name)) {
          c->x[0] = AllocateText(dir);
          return;
        }
        if (const char* v = VersionOverride(f->name)) {
          c->x[0] = AllocateText(v);
          return;
        }
        if (const char* v = LocaleOverride(f->name)) {
          c->x[0] = AllocateText(v);
          return;
        }
      }
      c->x[0] = AllocateText(f && f->kind == 's' ? f->text : "");
      return;
    }
    case 96: {  // GetBooleanField
      const Field* f = FieldFromId(c->x[2]);
      c->x[0] = f && f->kind == 'z' ? static_cast<uint64_t>(f->number) : 0;
      return;
    }
    case 100:    // GetIntField
    case 101: {  // GetLongField
      const Field* f = FieldFromId(c->x[2]);
      c->x[0] = f && f->kind == 'i' ? static_cast<uint64_t>(f->number) : 0;
      return;
    }
    case 102: {  // GetFloatField -- the result belongs in v0, not x0
      const Field* f = FieldFromId(c->x[2]);
      arc_s_w(c, 0, f && f->kind == 'f' ? static_cast<float>(f->real) : 0.0f);
      return;
    }
    case 163: {  // NewString(env, const jchar* chars, jsize len)
      // Unity turns a managed string into a jstring through a UTF-16 buffer
      // and this call, not through NewStringUTF. Left to the default, this
      // slot handed back a blank block -- so the method name and the signature
      // Unity passed to Java's reflection helper both arrived empty, and every
      // Java method it resolved was anonymous. That is why a title handing
      // Java a binary command looked exactly like a title doing nothing.
      const auto* units = reinterpret_cast<const uint16_t*>(c->x[1]);
      const size_t count = static_cast<size_t>(c->x[2]);
      std::string text;
      text.reserve(count);
      for (size_t i = 0; units && i < count; ++i) {
        uint32_t u = units[i];
        if (u >= 0xD800 && u < 0xDC00 && i + 1 < count) {
          const uint32_t low = units[++i];
          if (low >= 0xDC00 && low < 0xE000)
            u = 0x10000 + ((u - 0xD800) << 10) + (low - 0xDC00);
        }
        if (u < 0x80) {
          text.push_back(static_cast<char>(u));
        } else if (u < 0x800) {
          text.push_back(static_cast<char>(0xC0 | (u >> 6)));
          text.push_back(static_cast<char>(0x80 | (u & 0x3F)));
        } else if (u < 0x10000) {
          text.push_back(static_cast<char>(0xE0 | (u >> 12)));
          text.push_back(static_cast<char>(0x80 | ((u >> 6) & 0x3F)));
          text.push_back(static_cast<char>(0x80 | (u & 0x3F)));
        } else {
          text.push_back(static_cast<char>(0xF0 | (u >> 18)));
          text.push_back(static_cast<char>(0x80 | ((u >> 12) & 0x3F)));
          text.push_back(static_cast<char>(0x80 | ((u >> 6) & 0x3F)));
          text.push_back(static_cast<char>(0x80 | (u & 0x3F)));
        }
      }
      const uint64_t made = AllocateText(text.c_str());
      if (TraceJni())
        fprintf(stderr, "[jni] NewString(%zu) -> %#llx text=\"%.60s\"\n", count,
                static_cast<unsigned long long>(made),
                reinterpret_cast<const char*>(made));
      c->x[0] = made;
      return;
    }
    case 167: {  // NewStringUTF(env, const char*)
      // The guest's own bytes becoming a jstring, and the only place that text
      // ever enters the Java side. Without this the slot fell to the default,
      // which hands back a fresh block -- so every string the engine built for
      // Java arrived empty. It reads as working, because an empty string is a
      // legal string: the preference lookup that decides where the game's
      // server lives was being asked for the key "", and answered nothing.
      const uint64_t made =
          AllocateText(reinterpret_cast<const char*>(c->x[1]));
      if (c->x[1]) {
        std::lock_guard<std::mutex> held(g_asset_text_lock);
        g_last_utf8 = reinterpret_cast<const char*>(c->x[1]);
      }
      if (TraceJni())
        fprintf(stderr, "[jni] NewStringUTF -> %#llx text=\"%.60s\"\n",
                static_cast<unsigned long long>(made),
                reinterpret_cast<const char*>(made));
      c->x[0] = made;
      return;
    }
    case 164:    // GetStringLength
    case 168: {  // GetStringUTFLength
      const char* s = reinterpret_cast<const char*>(c->x[1]);
      c->x[0] = s ? strlen(s) : 0;
      return;
    }
    case 165: {  // GetStringChars -- UTF-16, not the UTF-8 the handle holds
      // Handing the handle straight back gives the caller our bytes read as
      // sixteen-bit units, which is right only for a single ASCII character
      // and garbage for anything longer. That is the sort of accident that
      // reads as working: a one-character path survives it intact.
      // ponytail: ASCII widened one byte per unit. Decode properly if a title
      // ever hands us a string that is not.
      const char* s = reinterpret_cast<const char*>(c->x[1]);
      const uint64_t wide = Allocate();
      if (wide && s) {
        auto* out = reinterpret_cast<uint16_t*>(wide);
        size_t i = 0;
        for (; s[i] && (i + 1) * sizeof(uint16_t) < kBlock; ++i)
          out[i] = static_cast<unsigned char>(s[i]);
        out[i] = 0;
      }
      c->x[0] = wide;
      if (c->x[2]) *reinterpret_cast<uint8_t*>(c->x[2]) = 1;  // isCopy = true
      return;
    }
    case 169: {  // GetStringUTFChars -- a jstring already holds its own text
      // An empty answer here is the dangerous one: a caller that takes the
      // length and subtracts one gets -1, hands that to memcpy, and runs off
      // the end of the arena a long way from the call that caused it. Worth
      // being able to see which string was empty and which was not.
      static const bool trace = TraceJni();
      if (trace) {
        const char* s = reinterpret_cast<const char*>(c->x[1]);
        fprintf(stderr, "[jni] GetStringUTFChars %#llx %s \"%.48s\" lr=%#llx\n",
                static_cast<unsigned long long>(c->x[1]),
                arc_jni_owns(c->x[1]) ? "ours" : "not ours", s ? s : "",
                static_cast<unsigned long long>(ARC_X_R(c, 30)));
      }
      const char* s = reinterpret_cast<const char*>(c->x[1]);
      // An opaque handle has no bytes of its own. The boot.config Scanner
      // reads through one, and the file's text is the answer it needs. A
      // handle made by AllocateText is different even when the string is
      // empty: that empty string is the answer, and substituting the last
      // asset read for it is how the whole boot config once became the
      // culture name.
      bool has_text = false;
      if (arc_jni_owns(c->x[1])) {
        has_text = *reinterpret_cast<const uint64_t*>(
                       c->x[1] + kBlock - sizeof(uint64_t)) ==
                   kTextHandleMagic;
      }
      if (!has_text && (!s || !*s)) {
        std::lock_guard<std::mutex> held(g_asset_text_lock);
        if (!g_last_asset_text.empty()) {
          c->x[0] = reinterpret_cast<uint64_t>(g_last_asset_text.c_str());
          if (c->x[2]) *reinterpret_cast<uint8_t*>(c->x[2]) = 0;
          return;
        }
      }
      c->x[0] = c->x[1];
      if (c->x[2]) *reinterpret_cast<uint8_t*>(c->x[2]) = 0;  // isCopy = false
      return;
    }
    case 6: {  // FindClass(env, const char* name)
      // The class handle used to be an anonymous block. That made the bridge
      // unable to tell one class from another, so every method was resolved by
      // name alone and a call through it had no owner to report. Carrying the
      // name is what lets a Java method be answered per class -- which is the
      // only way a binary protocol like NPF's can be implemented from this
      // side at all.
      const char* class_name = reinterpret_cast<const char*>(c->x[1]);
      // A Choreographer here is a name with no vsync behind it. The engine
      // enables VSYNC timing when it finds the class and then measures every
      // frame from callbacks that never arrive, which leaves its clock stopped
      // while the render loop keeps drawing. Answering "not there" is the truth
      // for a host with no Java vsync thread, and the engine's own timer then
      // supplies the frame step.
      static const bool refuse_choreographer =
          getenv("ARC_NO_CHOREOGRAPHER") != nullptr;
      if (refuse_choreographer && class_name &&
          strncmp(class_name, "android/view/Choreographer", 26) == 0) {
        c->x[0] = 0;
        return;
      }
      c->x[0] = AllocateText(class_name);
      return;
    }

    case 7:    // FromReflectedMethod(env, jobject method)
    case 8:    // FromReflectedField(env, jobject field) -- the object is the id
    case 9:    // ToReflectedMethod(env, jclass, jmethodID, jboolean)
    case 12: {  // ToReflectedField
      // Unity keeps a Java `Method` object where a jmethodID goes and converts
      // between the two views with these calls. There is no Java here, so the
      // two views *are* the same value: the id `getMethodID` answered with.
      // Treating the reflection object as opaque was what made every resolved
      // method anonymous -- the conversion turned a real name into a blank.
      // The object-to-id direction carries the object in x1; the id-to-object
      // direction carries the id in x2, behind the class.
      c->x[0] = (index == 7 || index == 8) ? c->x[1] : c->x[2];
      return;
    }

    case 33:     // GetMethodID(env, class, name, signature)
    case 113: {  // GetStaticMethodID
      const char* name = reinterpret_cast<const char*>(c->x[2]);
      const char* signature = reinterpret_cast<const char*>(c->x[3]);
      const char* owner = arc_jni_owns(c->x[1])
                              ? reinterpret_cast<const char*>(c->x[1])
                              : nullptr;
      // "This method is not there." Every method id this bridge hands out is
      // a stand-in, so a title that looks one up always finds it -- and a
      // title that asks Java for something optional is written to cope with
      // not finding it, which is a cleaner answer than a stand-in it will
      // then call.
      //
      // FetchMTXItems is the case that needs it: it looks up
      // retrieveSkuInformation, and if the id is null it returns before it
      // touches anything. Given an id, it goes on to build the store request
      // out of a catalogue this host has no Java billing service to fill,
      // and reads a name out of an item that has none.
      if (name && MethodIsAbsent(name)) {
        c->x[0] = 0;
        return;
      }
      c->x[0] = MethodIdFor(owner, name, signature);
      return;
    }

    // Constructing a Java object. The one that matters is the SDK's callback
    // wrapper, whose whole content is the id it was constructed with -- so
    // the handle is given that id as its text and can be recognised later.
    //
    // A request object is the other case: it is constructed with the id of the
    // conversation it belongs to and would read it back in Java, which is not
    // here. That first argument is remembered beside the handle, so the answer
    // to its `execute` can name the same conversation.
    case 28:     // NewObject
    case 29:     // NewObjectV
    case 30: {   // NewObjectA
      uint64_t made = Allocate();
      uint64_t first = 0;
      bool has_first = false;
      if (index == 30) {
        if (c->x[3]) {
          memcpy(&first, reinterpret_cast<const void*>(c->x[3]), sizeof first);
          has_first = true;
        }
      } else if (index == 29) {
        if (c->x[3]) {
          std::vector<unsigned char> cursor(arc::ShimVaListSize());
          memcpy(cursor.data(), reinterpret_cast<const void*>(c->x[3]),
                 cursor.size());
          first = arc::ShimVaNextInt(cursor.data());
          has_first = true;
        }
      } else {
        first = c->x[3];
        has_first = true;
      }
      if (index == 29 && c->x[3]) {
        std::vector<unsigned char> cursor(arc::ShimVaListSize());
        memcpy(cursor.data(), reinterpret_cast<const void*>(c->x[3]),
               cursor.size());
        for (int i = 0; i < 4; ++i) {
          const uint64_t v = arc::ShimVaNextInt(cursor.data());
          // A small number that is not one of our handles is the id. Ids
          // start at zero and wrap at a million, so the range is the test.
          if (v < 1000000 && !arc_jni_owns(v)) {
            char text[24];
            snprintf(text, sizeof text, "%llu",
                     static_cast<unsigned long long>(v));
            made = AllocateText(text);
            break;
          }
        }
      }
      NoteCtorText(made, c->x[1], first, has_first);
      if (getenv("ARC_TRACE_CTOR")) {
        const char* cls = HandleText(c->x[1]);
        const char* arg = HandleText(first);
        fprintf(stderr,
                "[jni] NewObject idx=%zu cls=%s first=%#llx \"%s\" -> %#llx\n",
                index, cls ? cls : "(none)",
                static_cast<unsigned long long>(first), arg ? arg : "(none)",
                static_cast<unsigned long long>(made));
      }
      c->x[0] = made;
      return;
    }

    // A call returning an object. Where the method's name says that object is
    // a string, answer with one: the engine joins these into paths, and an
    // empty answer is not a harmless unknown -- it becomes the filesystem root
    // and everything looked up beneath it fails.
    case 34:     // CallObjectMethod(env, obj, methodID, ...)
    case 35:     // CallObjectMethodV
    case 36:     // CallObjectMethodA
    case 114:    // CallStaticObjectMethod
    case 115:    // CallStaticObjectMethodV
    case 116: {  // CallStaticObjectMethodA
      // `String.getBytes()` and `toString()` are how a Java string becomes
      // something the engine can read, and both are called *on* the string --
      // so the answer is that object's own text, not an entry in a table. This
      // is why nothing ever called GetStringUTFChars: the conversion goes this
      // way instead, and answering it with a blank handle made every path the
      // engine asked for come back empty.
      const char* name = MethodName(c->x[2]);
      // Unity takes ownership of a touch with `MotionEvent.obtain(event)` and
      // then reads the *copy*: the source first, to decide whether it is a
      // touch at all, and the rest of the fields afterwards. Answering the
      // copy with a blank object makes every event look like it came from
      // nowhere, and the tap is dropped before any coordinate is asked for.
      if (name && strcmp(name, "obtain") == 0) {
        MotionEventState state;
        const std::vector<uint64_t> args = CallArgs(c, index, 1);
        if (!args.empty() && MotionEventStateFor(args[0], &state)) {
          c->x[0] = arc_jni_motion_event(state.action, state.x, state.y,
                                         state.down_time, state.event_time);
          return;
        }
      }
      // The platform trust store, asked for in the vocabulary Java uses.
      // See EnsureTrustStore above: this is the answer a TLS handshake needs,
      // and without it the verifier trusts nothing.
      if (name && strcmp(name, "getTrustManagers") == 0) {
        NoteCalled(name);
        EnsureTrustStore();
        const uint64_t managers[1] = {g_trust_manager};
        c->x[0] = AllocateObjectArray(managers, 1);
        return;
      }
      if (name && strcmp(name, "getAcceptedIssuers") == 0) {
        NoteCalled(name);
        EnsureTrustStore();
        c->x[0] = g_trust_issuers;
        return;
      }
      if (name && strcmp(name, "getEncoded") == 0) {
        // Only for the certificates this store handed out. getEncoded() is a
        // general question, and answering it with a CA for some other object
        // would be a different kind of wrong.
        EnsureTrustStore();
        for (size_t i = 0; i < g_trust_cert_handles.size(); ++i) {
          if (g_trust_cert_handles[i] != c->x[1]) continue;
          NoteCalled(name);
          const std::vector<unsigned char>& der = g_trust_certs[i];
          const uint64_t array = AllocateByteArray(der.size());
          if (array && !der.empty())
            memcpy(reinterpret_cast<void*>(array), der.data(), der.size());
          c->x[0] = array;
          return;
        }
      }
      if (name && strcmp(name, "getDefaultAlgorithm") == 0) {
        NoteCalled(name);
        c->x[0] = AllocateText("X509");
        return;
      }
      // Unity does not resolve Java method ids itself. It hands the class,
      // the name and the signature to `ReflectionHelper.getMethodID` in Java
      // and keeps the `java.lang.reflect.Method` that comes back, turning it
      // into an id later with `FromReflectedMethod`. Java is not here, so this
      // one call decides the identity of every later Java call -- and
      // answering it with a blank object made all of them anonymous. A title
      // handing Java a binary command then looked exactly like one calling a
      // method with no name at all, which is indistinguishable from a title
      // that does nothing.
      if (name && (strcmp(name, "getMethodID") == 0 ||
                   strcmp(name, "getConstructorID") == 0)) {
        const bool ctor = strcmp(name, "getConstructorID") == 0;
        const std::vector<uint64_t> args = CallArgs(c, index, ctor ? 2 : 4);
        const char* cls = args.size() > 0 ? HandleText(args[0]) : nullptr;
        const char* member = ctor ? "<init>"
                                  : args.size() > 1 ? HandleText(args[1]) : nullptr;
        const char* signature =
            args.size() > (ctor ? 1u : 2u) ? HandleText(args[ctor ? 1 : 2])
                                           : nullptr;
        if (TraceJni()) {
          uint64_t jv[4] = {};
          if (c->x[3]) memcpy(jv, reinterpret_cast<const void*>(c->x[3]), sizeof jv);
          fprintf(stderr,
                  "[jni] %s raw slot=%zu x1=%#llx x2=%#llx x3=%#llx "
                  "jv0=%#llx jv1=%#llx jv2=%#llx jv3=%#llx -> cls=%.48s "
                  "member=%.48s sig=%.48s\n",
                  name, index, static_cast<unsigned long long>(c->x[1]),
                  static_cast<unsigned long long>(c->x[2]),
                  static_cast<unsigned long long>(c->x[3]),
                  static_cast<unsigned long long>(jv[0]),
                  static_cast<unsigned long long>(jv[1]),
                  static_cast<unsigned long long>(jv[2]),
                  static_cast<unsigned long long>(jv[3]), cls ? cls : "(none)",
                  member ? member : "(none)", signature ? signature : "(none)");
        }
        c->x[0] = MethodIdFor(cls, member, signature);
        return;
      }
      // The boot config is the one file Java reads before Unity starts native
      // code. Return its text from open, keep it for the Scanner that follows,
      // and make the Scanner calls operate on that text rather than on an
      // opaque handle with no bytes behind it.
      if (name && strcmp(name, "open") == 0) {
        uint64_t arg = 0;
        std::string text;
        if (TraceJni()) {
          struct VaProbe {
            uint64_t stack, gr_top, vr_top;
            int32_t gr_offs, vr_offs;
          } va = {};
          if (c->x[3]) memcpy(&va, reinterpret_cast<const void*>(c->x[3]), sizeof va);
          fprintf(stderr,
                  "[jni] open raw x1=%#llx x3=%#llx x4=%#llx x5=%#llx "
                  "va={%#llx,%#llx,%#llx,%d,%d}\n",
                  static_cast<unsigned long long>(c->x[1]),
                  static_cast<unsigned long long>(c->x[3]),
                  static_cast<unsigned long long>(c->x[4]),
                  static_cast<unsigned long long>(c->x[5]),
                  static_cast<unsigned long long>(va.stack),
                  static_cast<unsigned long long>(va.gr_top),
                  static_cast<unsigned long long>(va.vr_top), va.gr_offs,
                  va.vr_offs);
          const uint64_t* regs =
              reinterpret_cast<const uint64_t*>(va.gr_top + va.gr_offs);
          if (regs)
            fprintf(stderr,
                    "[jni] open regs %#llx %#llx %#llx %#llx %#llx %#llx\n",
                    static_cast<unsigned long long>(regs[0]),
                    static_cast<unsigned long long>(regs[1]),
                    static_cast<unsigned long long>(regs[2]),
                    static_cast<unsigned long long>(regs[3]),
                    static_cast<unsigned long long>(regs[4]),
                    static_cast<unsigned long long>(regs[5]));
          const uint64_t* around =
              reinterpret_cast<const uint64_t*>(va.gr_top - 80);
          if (around)
            fprintf(stderr,
                    "[jni] open around %#llx %#llx %#llx %#llx %#llx "
                    "%#llx %#llx %#llx %#llx %#llx\n",
                    static_cast<unsigned long long>(around[0]),
                    static_cast<unsigned long long>(around[1]),
                    static_cast<unsigned long long>(around[2]),
                    static_cast<unsigned long long>(around[3]),
                    static_cast<unsigned long long>(around[4]),
                    static_cast<unsigned long long>(around[5]),
                    static_cast<unsigned long long>(around[6]),
                    static_cast<unsigned long long>(around[7]),
                    static_cast<unsigned long long>(around[8]),
                    static_cast<unsigned long long>(around[9]));
          const uint64_t* stack = reinterpret_cast<const uint64_t*>(va.stack);
          if (stack)
            fprintf(stderr,
                    "[jni] open stack %#llx %#llx %#llx %#llx %#llx %#llx\n",
                    static_cast<unsigned long long>(stack[0]),
                    static_cast<unsigned long long>(stack[1]),
                    static_cast<unsigned long long>(stack[2]),
                    static_cast<unsigned long long>(stack[3]),
                    static_cast<unsigned long long>(stack[4]),
                    static_cast<unsigned long long>(stack[5]));
        }
        std::vector<uint64_t> candidates = OwnedArgs(c);
        if (arc_jni_owns(c->x[3])) candidates.push_back(c->x[3]);
        for (uint64_t candidate : candidates) {
          const std::string found =
              ReadAssetFile(reinterpret_cast<const char*>(candidate));
          if (TraceJni())
            fprintf(stderr, "[jni] open arg %#llx \"%.60s\" %s\n",
                    static_cast<unsigned long long>(candidate),
                    reinterpret_cast<const char*>(candidate),
                    found.empty() ? "no file" : "found");
          if (!found.empty()) {
            arg = candidate;
            text = found;
            break;
          }
        }
        // The argument slot the engine uses for open() is not always the
        // string it just built: the JNI bridge can pass a class handle in the
        // first vararg position. The last NewStringUTF is the asset name in
        // that path, and boot.config is the only file read this early.
        if (text.empty()) {
          std::lock_guard<std::mutex> held(g_asset_text_lock);
          if (!g_last_utf8.empty()) {
            text = ReadAssetFile(g_last_utf8.c_str());
            if (!text.empty()) arg = AllocateText(g_last_utf8.c_str());
          }
        }
        if (text.empty()) {
          text = ReadAssetFile("bin/Data/boot.config");
          if (!text.empty()) arg = AllocateText("bin/Data/boot.config");
        }
        if (!arg) {
          arg = FirstOwnedArg(c);
          if (!arg && arc_jni_owns(c->x[3])) arg = c->x[3];
        }
        if (!text.empty()) {
          NoteCalled(name);
          {
            std::lock_guard<std::mutex> held(g_asset_text_lock);
            g_last_asset_text = text;
          }
          if (TraceJni())
            fprintf(stderr, "[jni] open %s -> %zu bytes\n",
                    reinterpret_cast<const char*>(arg), text.size());
          c->x[0] = AllocateText(text.c_str());
          return;
        }
        if (TraceJni())
          fprintf(stderr, "[jni] open MISS arg=%#llx text=\"%.60s\"\n",
                  static_cast<unsigned long long>(arg),
                  arg ? reinterpret_cast<const char*>(arg) : "");
      }
      if (name && strcmp(name, "useDelimiter") == 0) {
        NoteCalled(name);
        c->x[0] = c->x[1];
        return;
      }
      if (name && (strcmp(name, "next") == 0 ||
                   strcmp(name, "nextLine") == 0)) {
        NoteCalled(name);
        std::lock_guard<std::mutex> held(g_asset_text_lock);
        const uint64_t made = AllocateText(g_last_asset_text.c_str());
        if (TraceJni())
          fprintf(stderr, "[jni] next -> %#llx (%zu bytes) text=\"%.60s\"\n",
                  static_cast<unsigned long long>(made), g_last_asset_text.size(),
                  reinterpret_cast<const char*>(made));
        c->x[0] = made;
        return;
      }
      // ClassLoader.findLibrary is how Unity turns a backend name into the
      // shared object it will dlopen. The answer has to be non-empty and has
      // to name a file that exists; see LibraryPathFor.
      if (name && strcmp(name, "findLibrary") == 0) {
        NoteCalled(name);
        const uint64_t which = FirstOwnedArg(c);
        std::string requested =
            which ? reinterpret_cast<const char*>(which) : std::string();
        // The va_list walk is not reliable for this bridge: some calls put the
        // name in a register that the generic vararg reader cannot see. The
        // name was just made with NewStringUTF, so that is the fallback.
        if (requested.empty()) {
          std::lock_guard<std::mutex> held(g_asset_text_lock);
          requested = g_last_utf8;
        }
        static const bool trace = TraceJni();
        if (trace)
          fprintf(stderr, "[jni] findLibrary \"%s\"\n", requested.c_str());
        const std::string path = LibraryPathFor(requested.c_str());
        if (trace) {
          fprintf(stderr, "[jni] findLibrary path \"%s\" (%zu) lr=%#llx\n",
                  path.c_str(), path.size(),
                  static_cast<unsigned long long>(ARC_X_R(c, 30)));
          fprintf(stderr, "[jni] findLibrary frames:");
          const size_t n = arc_frame_count();
          for (size_t i = 0; i < n && i < 12; ++i) {
            const uint64_t frame = arc_frame_at(i);
            fprintf(stderr, " %u:%llx", static_cast<unsigned>(frame >> 56),
                    static_cast<unsigned long long>(
                        frame & 0x00FFFFFFFFFFFFFFull));
          }
          fprintf(stderr, "\n");
        }
        if (!path.empty()) {
          const uint64_t made = AllocateText(path.c_str());
          if (trace)
            fprintf(stderr, "[jni] findLibrary made %#llx text \"%.60s\"\n",
                    static_cast<unsigned long long>(made),
                    reinterpret_cast<const char*>(made));
          g_trace_next_alloc = true;
          c->x[0] = made;
          return;
        }
      }
      // getComponentId is the same question in a different vocabulary: the
      // Nimble bridge hands Java an object per registered component and asks
      // it back which component that was. A handle we made carries its own
      // text, and here that text *is* the id.
      if (name && (strcmp(name, "getBytes") == 0 ||
                   strcmp(name, "toString") == 0 ||
                   strcmp(name, "getComponentId") == 0)) {
        NoteCalled(name);
        // getBytes and toString are called *on* the object; getComponentId
        // is static and takes it as an argument, where x1 is the class.
        const uint64_t which = strcmp(name, "getComponentId") == 0
                                   ? FirstOwnedArg(c)
                                   : c->x[1];
        const char* self = reinterpret_cast<const char*>(which);
        // Whether the object carries text at all is the question these turn
        // on: a handle we made for a string does, and one we handed back for
        // something we did not understand is a zeroed block that does not.
        static const bool trace = TraceJni();
        if (trace)
          fprintf(stderr, "[jni] %s on %#llx %s text=\"%.48s\"\n", name,
                  static_cast<unsigned long long>(which),
                  arc_jni_owns(which) ? "ours" : "not ours",
                  self ? self : "");
        c->x[0] = AllocateText(self);
        return;
      }
      // Where the Nimble SDK expects its backend to be. Its C++ half is a
      // proxy over a Java component, so every question about the environment
      // -- which server, which client id -- is a Java call, and a host that
      // answers nothing leaves the SDK convinced it has no environment at
      // all. One loopback server answers every service, so the key is not
      // read: the answer is the same address whatever was asked for.
      if (name && strcmp(name, "getServerUrlWithKey") == 0) {
        NoteCalled(name);
        static const char* v = getenv("ARC_SERVER");
        TraceVaArgs(name, c);
        if (v) {
          c->x[0] = AllocateText(v);
          return;
        }
      }
      // Read by key: the method name says only that a preference was wanted.
      // Nimble's persistence and its environment both ask this shape of
      // question, in their own vocabulary, so they share the answer.
      if (name && (strcmp(name, "getSharedPreference") == 0 ||
                   strcmp(name, "getStringPreference") == 0 ||
                   strcmp(name, "getParameter") == 0 ||
                   strcmp(name, "getStringValue") == 0)) {
        NoteCalled(name);
        TraceVaArgs(name, c);
        const uint64_t key = FirstOwnedArg(c);
        if (key) {
          if (const char* v =
                  PreferenceValue(reinterpret_cast<const char*>(key))) {
            c->x[0] = AllocateText(v);
            return;
          }
        }
      }
      // A *static* helper that turns something into a String -- the engine
      // fills a byte array and hands it to one of these. The text is in an
      // argument rather than in the object, and passing the handle straight
      // back is what a jstring means here anyway.
      if (name && (strcmp(name, "CreateNewStringUTFSafe") == 0 ||
                   strcmp(name, "createNewStringUTFSafe") == 0 ||
                   strcmp(name, "newStringUTF") == 0)) {
        NoteCalled(name);
        if (const uint64_t arg = FirstOwnedArg(c)) {
          c->x[0] = arg;
          return;
        }
      }
      // A File in this environment *is* its path text: every directory the
      // engine asks Java for is a handle carrying a Unix path, and a File built
      // from one is the same text again. Asking such an object to repeat itself
      // -- its absolute or canonical path -- has one right answer, and the
      // table's blank is not it. That blank is what made Unity's
      // persistentDataPath the empty string, so every cache root Nabe built
      // from it collapsed to "/" and the SDK looked for its own packs at the
      // root of the drive.
      if (name && (strcmp(name, "getAbsolutePath") == 0 ||
                   strcmp(name, "getPath") == 0 ||
                   strcmp(name, "getCanonicalPath") == 0)) {
        if (HandleText(c->x[1])) {
          NoteCalled(name);
          c->x[0] = c->x[1];
          return;
        }
      }
      const char* text = TextForMethod(c->x[2]);
      c->x[0] = text ? AllocateText(text) : Allocate();
      return;
    }

    // A method returning an integer, in every spelling JNI has for it: byte,
    // char, short, int and long, called on an object or on a class, through
    // varargs, a va_list or an array. The width does not change the answer.
    case 40: case 41: case 42:     // CallByteMethod{,V,A}
    case 43: case 44: case 45:     // CallCharMethod
    case 46: case 47: case 48:     // CallShortMethod
    case 49: case 50: case 51:     // CallIntMethod
    case 52: case 53: case 54:     // CallLongMethod
    case 120: case 121: case 122:  // CallStaticByteMethod
    case 123: case 124: case 125:  // CallStaticCharMethod
    case 126: case 127: case 128:  // CallStaticShortMethod
    case 129: case 130: case 131:  // CallStaticIntMethod
    case 132: case 133: case 134: {  // CallStaticLongMethod
      // A Java enum reaches C++ as an object whose ordinal() is asked for
      // separately, so the number is a property of the handle rather than of
      // the method name. Ours carry their text; an enum constant's text is
      // its ordinal, and reading it back is what keeps "which constant" and
      // "what number" the same answer.
      const char* name = MethodName(c->x[2]);
      if (name && (strcmp(name, "hasNext") == 0 ||
                   strcmp(name, "hasNextLine") == 0)) {
        NoteCalled(name);
        c->x[0] = 1;
        return;
      }
      if (name && strcmp(name, "ordinal") == 0) {
        NoteCalled(name);
        const char* self = reinterpret_cast<const char*>(c->x[1]);
        c->x[0] = arc_jni_owns(c->x[1]) && self ? strtoull(self, nullptr, 10)
                                                : 0;
        return;
      }
      bool answered = false;
      c->x[0] = static_cast<uint64_t>(NumberForMethod(c->x[2], &answered));
      return;
    }

    // A method returning a float or a double. These have to write the
    // floating-point register, not just x0: a JNI float return lives in s0,
    // and the generic fallback only filled x0, so the guest read whatever the
    // previous guest computation left in s0. getRefreshRate is the one that
    // matters -- the engine divides by it to turn a frame interval into ticks
    // -- and a stale value there (the host's uptime, ~1.8e12) saturated the
    // frame clock and left the game spinning in its first heavy frame. 60 Hz
    // is the Android baseline and what the engine expects when the host has
    // no display to ask.
    case 55: case 56: case 57:       // CallFloatMethod{,V,A}
    case 135: case 136: case 137: {  // CallStaticFloatMethod{,V,A}
      const char* name = MethodName(c->x[2]);
      if (name) NoteCalled(name);
      const bool refresh = name && strcmp(name, "getRefreshRate") == 0;
      if (refresh && getenv("ARC_TRACE_REFRESH"))
        fprintf(stderr, "[refresh] getRefreshRate -> 60\n");
      c->q[0].f32[0] = refresh ? 60.0f : 0.0f;
      c->x[0] = 0;
      return;
    }
    case 58: case 59: case 60:       // CallDoubleMethod{,V,A}
    case 138: case 139: case 140: {  // CallStaticDoubleMethod{,V,A}
      const char* name = MethodName(c->x[2]);
      if (name) NoteCalled(name);
      c->q[0].f64[0] = 0.0;
      c->x[0] = 0;
      return;
    }

    // A method returning nothing. Almost all of these are genuinely ignorable
    // -- telemetry, analytics, a crash-reporter key -- but one is not: the
    // engine writes its *own* diagnostic log through Java rather than through
    // __android_log, so dropping these silently throws away the running
    // commentary a port most needs. Anything log-shaped is printed.
    case 61:     // CallVoidMethod
    case 62:     // CallVoidMethodV
    case 63:     // CallVoidMethodA
    case 141:    // CallStaticVoidMethod
    case 142:    // CallStaticVoidMethodV
    case 143: {  // CallStaticVoidMethodA
      const char* name = MethodName(c->x[2]);
      if (name) NoteCalled(name);
      // What a title hands Java is the whole question when Java is not here:
      // the call names a method, and the arguments say what it was asked to
      // do. Printing the text ones costs little and turns "it called execute"
      // into the actual command.
      if (TraceJni()) {
        const std::vector<uint64_t> shown = CallArgs(c, index, 4);
        for (size_t i = 0; i < shown.size(); ++i) {
          const char* text = HandleText(shown[i]);
          if (!text) continue;
          fprintf(stderr, "[jni]   %s arg%zu (%zu) \"%.600s\"\n",
                  name ? name : "?", i, strlen(text), text);
        }
      }
      // Only the va_list forms carry arguments we can walk; the varargs forms
      // spread theirs across registers we were not handed.
      const bool va = index == 62 || index == 142;
      // The write half of the preference store. Both arguments are handles we
      // made, so "the first two of ours, in order" is the key and the value.
      if (name && va &&
          (strcmp(name, "setValue") == 0 ||
           strcmp(name, "setSharedPreference") == 0 ||
           strcmp(name, "setStringPreference") == 0)) {
        const std::vector<uint64_t> args = OwnedArgs(c);
        if (!args.empty()) {
          SetPreference(reinterpret_cast<const char*>(args[0]),
                        args.size() > 1
                            ? reinterpret_cast<const char*>(args[1])
                            : "");
          TracePreference(reinterpret_cast<const char*>(args[0]),
                          args.size() > 1
                              ? reinterpret_cast<const char*>(args[1])
                              : "(set empty)");
        }
        c->x[0] = 0;
        return;
      }
      // Asked Java to go and do something and then wait to be called back.
      // Nothing here will do the work, so the honest answer is the one Java
      // gives a device with no Play Services: the callback, with nothing in
      // it. What matters is that it arrives -- the SDK handles an empty
      // attestation, and handles never being answered by waiting forever.
      // ARC_NIMBLE_CALLBACK=off to go back to not answering, which is what
      // this did before and which stops the anonymous login dead.
      static const char* mode = getenv("ARC_NIMBLE_CALLBACK");
      const bool deliver = !(mode && strcmp(mode, "off") == 0);
      if (deliver && name && va &&
          strcmp(name, "requestSafetyNetAttestation") == 0) {
        for (uint64_t arg : OwnedArgs(c)) {
          const long id = CallbackIdOf(arg);
          if (id >= 0) {
            DeliverNimbleCallback(id, getenv("ARC_SAFETYNET_TOKEN"));
            break;
          }
        }
        c->x[0] = 0;
        return;
      }
      if (name && strcmp(name, "synchronize") == 0) {
        SyncPreferences();
        c->x[0] = 0;
        return;
      }
      if (name && va && IsLogMethod(name)) {
        // Every string argument, not just the first. A logger that takes a
        // title and a message puts the title first, so printing one argument
        // prints the category and throws the line away -- which is how the
        // Nimble SDK's entire diagnostic stream arrived as the word
        // "NexusService", repeated.
        bool any = false;
        for (uint64_t arg : OwnedArgs(c)) {
          fprintf(stderr, any ? "  %.400s" : "[game] %.400s",
                  reinterpret_cast<const char*>(arg));
          any = true;
        }
        if (any) fputc('\n', stderr);
      }
      c->x[0] = 0;
      return;
    }

    // A method returning a boolean, called on an object or on a class. Six
    // slots each because JNI spells every call three ways -- varargs, va_list
    // and array -- and the answer does not depend on which.
    case 37:     // CallBooleanMethod
    case 38:     // CallBooleanMethodV
    case 39:     // CallBooleanMethodA
    case 117:    // CallStaticBooleanMethod
    case 118:    // CallStaticBooleanMethodV
    case 119: {  // CallStaticBooleanMethodA
      // mkdir is the one boolean that has to do something rather than say
      // something. Answering true without making the directory is the worst of
      // the three possible answers: false would at least be reported, and
      // making it is what the caller asked for. The engine creates its content
      // directory this way and then writes downloads into it -- so a bare
      // "true" left every one of those opens failing, and a download loop
      // repeating forever with the server returning 200 each time.
      const char* name = MethodName(c->x[2]);
      if (name && strcmp(name, "mkdir") == 0) {
        NoteCalled(name);
        if (const uint64_t path = FirstOwnedArg(c)) {
          std::error_code ec;
          const auto* dir = reinterpret_cast<const char*>(path);
          if (!std::filesystem::is_directory(dir, ec))
            std::filesystem::create_directories(dir, ec);
          static const bool trace = getenv("ARC_TRACE_FILES") != nullptr;
          if (trace)
            fprintf(stderr, "[file] mkdir    %-5s %s\n", ec ? "FAIL" : "ok",
                    dir);
          c->x[0] = ec ? 0 : 1;
          return;
        }
      }
      bool answered = false;
      c->x[0] = FlagForMethod(c, c->x[2], &answered) ? 1 : 0;
      return;
    }

    case 32: {  // IsInstanceOf(env, jobject obj, jclass clazz)
      // The only hierarchy this bridge can answer is the one it built: a
      // handle that carries its class name is an instance of the class of the
      // same name. A block we handed back for a question we did not
      // understand carries nothing, and calling it an instance of anything
      // would be a guess -- which is what a verifier cannot afford.
      const char* object = HandleText(c->x[1]);
      const char* klass = HandleText(c->x[2]);
      c->x[0] = (object && klass && strcmp(object, klass) == 0) ? 1 : 0;
      return;
    }

    case 171: {  // GetArrayLength
      // An array of objects knows its own count. A byte array knows its own
      // length. Anything else is text, which is the same NUL-terminated bytes
      // a string handle holds -- and an array we did not fill is a zeroed
      // block, which measures zero, which is the truth about it.
      if (IsObjectArray(c->x[1])) {
        c->x[0] = ArrayCount(c->x[1]);
        return;
      }
      if (IsByteArray(c->x[1])) {
        c->x[0] = ByteArrayLength(c->x[1]);
        return;
      }
      const char* s = reinterpret_cast<const char*>(c->x[1]);
      c->x[0] = s ? strlen(s) : 0;
      return;
    }
    case 173: {  // GetObjectArrayElement
      if (IsObjectArray(c->x[1])) {
        c->x[0] = ArrayElement(c->x[1], c->x[2]);
        return;
      }
      c->x[0] = Allocate();
      return;
    }
    case 184:    // GetByteArrayElements
    case 222: {  // GetPrimitiveArrayCritical -- the same question, asked in
                 // the form that promises not to hold the collector up
      // The elements *are* the handle: that is where the bytes were put. The
      // default for a slot that returns something is a fresh block, and a
      // fresh block is empty -- so a caller reading an array through this got
      // nothing back and had no way to tell that from an empty array.
      c->x[0] = c->x[1];
      if (c->x[2]) *reinterpret_cast<uint8_t*>(c->x[2]) = 0;  // isCopy = false
      return;
    }
    case 176: {  // NewByteArray(env, jsize len)
      c->x[0] = AllocateByteArray(c->x[1]);
      return;
    }
    case 208: {  // SetByteArrayRegion(env, array, start, len, buf)
      // The engine builds a Java string by filling a byte array and handing it
      // to a helper. Dropping this leaves the array empty, the string empty,
      // and the engine's own log a column of blank lines. The same call is how
      // a title hands Java a binary payload, so the copy has to be exact --
      // including the zero bytes -- and the length has to grow with it.
      const auto* src = reinterpret_cast<const char*>(c->x[4]);
      const uint64_t start = c->x[2], len = c->x[3];
      static const char* dump = getenv("ARC_TRACE_JNI_BYTES");
      static unsigned long long dumped;
      const unsigned long long cap = dump ? strtoull(dump, nullptr, 0) : 0;
      if (dump && src && len && dumped < cap) {
        const unsigned long long room = cap - dumped;
        const unsigned long long show = len < room ? len : room;
        fprintf(stderr, "[jni] bytes array=%#llx start=%llu len=%llu:",
                static_cast<unsigned long long>(c->x[1]),
                static_cast<unsigned long long>(start),
                static_cast<unsigned long long>(len));
        for (uint64_t i = 0; i < show && i < 256; ++i)
          fprintf(stderr, " %02x", static_cast<unsigned char>(src[i]));
        fputc('\n', stderr);
        dumped += show;
      }
      if (!arc_jni_owns(c->x[1])) return;
      const size_t room = IsByteArray(c->x[1])
                              ? kBytesCapacity + 1
                              : kBlock - 2 * sizeof(uint64_t);
      const uint64_t copy = start < room
                                ? (len < room - start ? len : room - start)
                                : 0;
      auto* dst = reinterpret_cast<char*>(c->x[1]) + start;
      if (src && copy) memcpy(dst, src, static_cast<size_t>(copy));
      // NUL-terminated when there is room, because a byte array of text is
      // also read as one by the engine's own helpers.
      if (start + copy < room) dst[copy] = '\0';
      if (IsByteArray(c->x[1]) &&
          start + copy > ByteArrayLength(c->x[1]))
        SetByteArrayLength(c->x[1], start + copy);
      return;
    }

    case 191:    // ReleaseBooleanArrayElements
    case 192:    // ReleaseByteArrayElements
    case 223:    // ReleasePrimitiveArrayCritical
      return;    // nothing was copied, so there is nothing to write back

    case 215: {  // RegisterNatives(env, clazz, methods, count)
      // `methods` points at the guest's own array, in the mapped image, so it
      // is readable here. Copy the names rather than keeping pointers into an
      // image that a later unload could take away.
      struct GuestNativeMethod {
        const char* name;
        const char* signature;
        void* fn;
      };
      const auto* methods =
          reinterpret_cast<const GuestNativeMethod*>(c->x[2]);
      const int count = static_cast<int>(c->x[3]);
      if (methods && count > 0) {
        std::lock_guard<std::mutex> g(g_natives_lock);
        for (int i = 0; i < count; ++i) {
          if (TraceJni())
            fprintf(stderr, "[jni] RegisterNatives %s %s -> %#llx\n",
                    methods[i].name ? methods[i].name : "?",
                    methods[i].signature ? methods[i].signature : "?",
                    static_cast<unsigned long long>(
                        reinterpret_cast<uintptr_t>(methods[i].fn)));
          g_natives.push_back({methods[i].name ? methods[i].name : "",
                               methods[i].signature ? methods[i].signature : "",
                               reinterpret_cast<uint64_t>(methods[i].fn)});
        }
      }
      c->x[0] = 0;  // JNI_OK
      return;
    }

    case 219: {  // GetJavaVM(env, JavaVM** out)
      // Writing the VM out is the whole point of the call. Left unwritten, the
      // caller reads whatever that variable happened to hold and branches
      // through it as though it were a table of functions.
      if (c->x[1])
        *reinterpret_cast<uint64_t*>(c->x[1]) =
            reinterpret_cast<uint64_t>(&g_vm_table);
      c->x[0] = 0;  // JNI_OK
      return;
    }

    // The invocation interface. Attaching a thread and asking for its
    // environment both hand back the one environment there is: nothing in it
    // is per-thread, so every thread can share it.
    case kVmBase + 4:    // AttachCurrentThread(vm, JNIEnv**, void*)
    case kVmBase + 6:    // GetEnv(vm, void**, version)
    case kVmBase + 7: {  // AttachCurrentThreadAsDaemon
      if (c->x[1])
        *reinterpret_cast<uint64_t*>(c->x[1]) =
            reinterpret_cast<uint64_t>(&g_table);
      c->x[0] = 0;
      return;
    }
    case kVmBase + 3:  // DestroyJavaVM
    case kVmBase + 5:  // DetachCurrentThread
      c->x[0] = 0;
      return;

    default:
      break;
  }

  const SlotInfo* info = Lookup(index);
  if (info) {
    c->x[0] = info->returns_handle ? Allocate() : 0;
    return;
  }
  // Not one we recognise. Erring towards a handle is safer: one the caller
  // ignores costs a block of arena, whereas a zero it treats as a pointer or a
  // count becomes arithmetic on nothing.
  c->x[0] = Allocate();
}

template <size_t N>
void Slot(Arm64Ctx* c) {
  Handle(N, c);
}

template <size_t... I>
void FillSlots(std::index_sequence<I...>) {
  ((g_slots[I] = reinterpret_cast<uint64_t>(&Slot<I>)), ...);
}

struct Init {
  Init() {
    FillSlots(std::make_index_sequence<kSlots>{});
    // The VM's table is the tail of the same pool, so the two interfaces are
    // dispatched and named by one mechanism.
    for (size_t i = 0; i < kVmSlots; ++i) g_vm_slots[i] = g_slots[kVmBase + i];
  }
} g_init;

}  // namespace

extern "C" {

uint64_t arc_jni_env(void) { return reinterpret_cast<uint64_t>(&g_table); }

uint64_t arc_jni_vm(void) { return reinterpret_cast<uint64_t>(&g_vm_table); }

void arc_jni_register(void) {
  uint64_t low = g_slots[0], high = g_slots[0];
  for (size_t i = 0; i < kSlots; ++i) {
    arc_register_ctx_native(g_slots[i], NameOf(i),
                            reinterpret_cast<ArcCtxFn>(g_slots[i]));
    if (g_slots[i] < low) low = g_slots[i];
    if (g_slots[i] > high) high = g_slots[i];
  }
  // Printed because a branch through this table that the dispatcher does not
  // recognise is otherwise indistinguishable from one that never came from
  // here at all: with the extent in hand, an address either is a slot or is
  // not, and there is nothing to reason about.
  printf("jni        %zu slots at %p, stubs %#llx..%#llx\n", kSlots,
         static_cast<void*>(&g_table), static_cast<unsigned long long>(low),
         static_cast<unsigned long long>(high));
}

// The object handed to an entry point as its `this`. Nothing reads it
// directly -- every access goes through a field accessor -- so a block of
// arena is enough.
uint64_t arc_jni_object(void) { return Allocate(); }

// A jstring, which here is just its own text: every path that reads one --
// GetStringUTFChars, getBytes, toString -- hands the handle straight back.
uint64_t arc_jni_string(const char* text) { return AllocateText(text); }

uint64_t arc_jni_motion_event(int action, float x, float y, int64_t down_time,
                              int64_t event_time) {
  const uint64_t handle = AllocateText("android/view/MotionEvent");
  if (!handle) return 0;
  std::lock_guard<std::mutex> held(g_motion_lock);
  MotionEventState& state = g_motion_events[handle];
  state.action = action;
  state.x = x;
  state.y = y;
  state.down_time = down_time;
  state.event_time = event_time;
  return handle;
}

const char* arc_jni_text(uint64_t handle) { return HandleText(handle); }

const char* arc_jni_ctor_text(uint64_t handle) { return CtorTextFor(handle); }

const char* arc_jni_ctor_class(uint64_t handle) { return CtorClassFor(handle); }

// The bytes behind one of the environment's handles, or null if the handle is
// not a byte array. A title that speaks a binary protocol to its Java half
// hands the payload over as `byte[]`, and there is nothing textual to read.
const uint8_t* arc_jni_bytes(uint64_t handle, size_t* length) {
  if (!IsByteArray(handle)) return nullptr;
  if (length) *length = static_cast<size_t>(ByteArrayLength(handle));
  return reinterpret_cast<const uint8_t*>(handle);
}

void arc_jni_register_method(const char* owner, const char* name,
                             const char* signature, ArcJavaMethodFn fn) {
  if (!name || !fn) return;
  std::lock_guard<std::mutex> held(g_java_method_lock);
  g_java_methods.push_back(
      {owner ? owner : "", name, signature ? signature : "", fn});
}

uint64_t arc_guest_string(const char* utf8) { return GuestStringFor(utf8); }

int arc_call_managed(const char* assembly, const char* name_space,
                     const char* klass, const char* method, size_t argc,
                     const char* const* utf8_args) {
  if (!assembly || !klass || !method) return -1;
  if (argc > 8) return -1;
  const uint64_t class_from_name = GuestExport("il2cpp_class_from_name");
  const uint64_t method_from_name =
      GuestExport("il2cpp_class_get_method_from_name");
  const uint64_t invoke = GuestExport("il2cpp_runtime_invoke");
  if (!class_from_name || !method_from_name || !invoke) return -1;
  const uint64_t image = ManagedImage(assembly);
  if (!image) return -2;
  const uint64_t class_args[3] = {image,
                                  reinterpret_cast<uint64_t>(name_space),
                                  reinterpret_cast<uint64_t>(klass)};
  const uint64_t found_class = arc_call_guest(class_from_name, class_args, 3);
  if (!found_class) return -3;
  const uint64_t method_args[3] = {found_class,
                                   reinterpret_cast<uint64_t>(method),
                                   static_cast<uint64_t>(argc)};
  const uint64_t found_method =
      arc_call_guest(method_from_name, method_args, 3);
  if (!found_method) return -4;
  uint64_t strings[8] = {};
  for (size_t i = 0; i < argc; ++i) {
    strings[i] = GuestStringFor(utf8_args && utf8_args[i] ? utf8_args[i] : "");
    if (!strings[i]) return -5;
  }
  // The invoke reports a managed exception through the last argument rather
  // than by unwinding into host code, which is the only shape a managed
  // fault may take across this boundary.
  uint64_t exception = 0;
  const uint64_t invoke_args[4] = {found_method, 0,
                                   reinterpret_cast<uint64_t>(strings),
                                   reinterpret_cast<uint64_t>(&exception)};
  arc_call_guest(invoke, invoke_args, 4);
  return exception ? 1 : 0;
}

uint64_t arc_jni_lookup_native(const char* name, const char* signature) {
  if (!name) return 0;
  std::lock_guard<std::mutex> g(g_natives_lock);
  for (const RegisteredNative& n : g_natives)
    if (n.name == name && (!signature || n.signature == signature))
      return n.fn;
  return 0;
}

size_t arc_jni_native_count(void) {
  std::lock_guard<std::mutex> g(g_natives_lock);
  return g_natives.size();
}

void arc_jni_dump_natives(void) {
  std::lock_guard<std::mutex> g(g_natives_lock);
  for (const RegisteredNative& n : g_natives)
    fprintf(stderr, "[jni] native %-32s %s @ %#llx\n", n.name.c_str(),
            n.signature.c_str(),
            static_cast<unsigned long long>(n.fn));
}

size_t arc_jni_call_count(void) { return g_total; }

int arc_jni_owns(uint64_t address) {
  return g_arena && address >= reinterpret_cast<uint64_t>(g_arena) &&
         address < reinterpret_cast<uint64_t>(g_arena) + kArenaSize;
}

void arc_jni_report(void) {
  if (!g_total) {
    printf("  no JNI slots were called\n");
    return;
  }
  // Loaded explicitly: an atomic has no business being passed to a variadic
  // function, and MSVC is right to refuse it.
  printf("  %zu JNI calls across these slots:\n", g_total.load());
  for (size_t i = 0; i < kSlots; ++i)
    if (const size_t n = g_hits[i].load())
      printf("    %6zu x  slot %3zu  %s\n", n, i, NameOf(i));

  // The Java methods the engine called expecting an object back, that nothing
  // here had a value for. Each got a blank handle, which reads as an empty
  // string -- so any of these that was meant to be a path is a lookup failing
  // somewhere later for a reason that points nowhere near here.
  {
    std::lock_guard<std::mutex> held(g_field_lock);
    if (!g_unknown_fields.empty()) {
      printf("  object fields asked for that nothing here answers:\n   ");
      for (const std::string& s : g_unknown_fields) printf(" %s", s.c_str());
      printf("\n");
    }
  }
  std::lock_guard<std::mutex> held(g_method_lock);
  if (!g_method_called.empty()) {
    printf("  Java methods invoked:\n   ");
    for (const std::string& s : g_method_called) printf(" %s", s.c_str());
    printf("\n");
  }
  if (!g_method_unanswered.empty()) {
    printf("  ...of which these had no value to return:\n   ");
    for (const std::string& s : g_method_unanswered) printf(" %s", s.c_str());
    printf("\n");
  }
}

}  // extern "C"
