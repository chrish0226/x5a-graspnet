// ABI compatibility for ONE hash-checked vendor library, not a generic SDK.
// Original feedback parsing, CAN transport, gains and fault handling are kept.
// Only freshness predicates and the aggregate online latch are replaced.
#include "link_policy.hpp"
#include <atomic>
#include <chrono>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <openssl/evp.h>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <thread>

namespace {
std::array<std::atomic<int64_t>, 7> arrivals{};
std::atomic<int64_t> timeout4{300000000}, timeout2{800000000};
std::atomic<uint64_t> epoch{0};
bool had_healthy = false; // Accessed only by the SDK control thread.
bool previous_healthy = false;
template<class T> T read(const void* obj, size_t offset) {
  T value; std::memcpy(&value, static_cast<const char*>(obj) + offset, sizeof(value)); return value;
}
template<class T> T resolve(const char* symbol) {
  auto ptr = dlsym(RTLD_NEXT, symbol);
  if (!ptr) throw std::runtime_error(std::string("SDK symbol missing: ") + symbol);
  return reinterpret_cast<T>(ptr);
}
using Parse = void(*)(void*, void*);
using Load = void(*)(void*);
constexpr auto parse4_name = "_ZN3arx12hw_interface10MotorType410CanAnalyzeEPNS0_8CanFrameE";
constexpr auto parse2_name = "_ZN3arx12hw_interface10MotorType210CanAnalyzeEPNS0_8CanFrameE";
constexpr auto load_name = "_ZN3arx2x514ControllerBase15loadJointStatusEv";
void parsed(void* obj, void* frame, int type) {
  const int id = read<int>(obj, type == 4 ? 0x70 : 0x78);
  if (wrist::accepts_frame(type, id, read<uint32_t>(frame, 0),
      read<uint32_t>(frame, 4), read<unsigned char>(frame, 8)))
    arrivals[wrist::motor_index(id)].store(wrist::monotonic_ns(), std::memory_order_release);
}
bool is_online(void* obj, int type) {
  const int id = read<int>(obj, type == 4 ? 0x70 : 0x78);
  const int index = wrist::motor_index(id);
  if (index < 0) return false;
  // Function-argument evaluation order is unspecified. Read the receipt BEFORE
  // taking 'now', so a concurrent callback cannot make it appear in the future.
  const auto received = arrivals[index].load(std::memory_order_acquire);
  const auto now = wrist::monotonic_ns();
  return wrist::fresh(received, now, type == 4 ? timeout4.load() : timeout2.load());
}
}
namespace wrist {
int64_t monotonic_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}
void configure(double ms4, double ms2) {
  if (!(ms4 >= 100 && ms4 <= 500 && ms2 >= 200 && ms2 <= 800))
    throw std::invalid_argument("Motor timeouts must be bounded: type4 100..500ms, type2 200..800ms");
  timeout4.store(static_cast<int64_t>(ms4 * 1000000));
  timeout2.store(static_cast<int64_t>(ms2 * 1000000));
}
void validate_sdk() {
  auto original = resolve<Parse>(parse4_name);
  Dl_info info{};
  if (!dladdr(reinterpret_cast<void*>(original), &info) || !info.dli_fname)
    throw std::runtime_error("Cannot identify the loaded SDK");
  std::ifstream file(info.dli_fname, std::ios::binary);
  if (!file) throw std::runtime_error("Cannot read the loaded SDK");
  std::vector<char> bytes((std::istreambuf_iterator<char>(file)), {});
  unsigned char digest[EVP_MAX_MD_SIZE]; unsigned int length = 0;
  if (!EVP_Digest(bytes.data(), bytes.size(), digest, &length, EVP_sha256(), nullptr))
    throw std::runtime_error("Cannot verify SDK hash");
  std::ostringstream hex;
  for (unsigned i = 0; i < length; ++i) hex << std::hex << std::setw(2) << std::setfill('0') << int(digest[i]);
  if (hex.str() != "d07d53d14da17f0a1db2753e3339f9e4c055dad72df613c811b9692c4bede85c")
    throw std::runtime_error("Unverified SDK ABI; refuse hardware initialization");
  resolve<Parse>(parse2_name); resolve<Load>(load_name);
  resolve<Load>("_ZN3arx2x516ControllerThread9ArmThreadEv");
  resolve<Load>("_ZN3arx2x514ControllerBase6UpdateEv");
  resolve<Load>("_ZN3arx2x516ControllerThread12exchangeDataEv");
  resolve<Load>("_ZN3arx2x514ControllerBase11destoryFuncEv");
}
Snapshot snapshot() {
  Snapshot result{}; result.all_fresh = true; result.protection_epoch = epoch.load();
  std::array<int64_t, 7> received;
  for (int i=0; i<7; ++i) received[i]=arrivals[i].load(std::memory_order_acquire);
  const auto now = monotonic_ns();
  for (int i = 0; i < 7; ++i) {
    result.ages_ms[i] = received[i] > 0 ? (now - received[i]) / 1000000.0 : -1;
    result.all_fresh &= fresh(received[i], now, i < 3 ? timeout4.load() : timeout2.load());
  }
  return result;
}
}
extern "C" void parse4(void*, void*) asm("_ZN3arx12hw_interface10MotorType410CanAnalyzeEPNS0_8CanFrameE");
extern "C" void parse2(void*, void*) asm("_ZN3arx12hw_interface10MotorType210CanAnalyzeEPNS0_8CanFrameE");
extern "C" bool online4(void*) asm("_ZN3arx12hw_interface10MotorType46onlineEv");
extern "C" bool online2(void*) asm("_ZN3arx12hw_interface10MotorType26onlineEv");
extern "C" void load_status(void*) asm("_ZN3arx2x514ControllerBase15loadJointStatusEv");
extern "C" void parse4(void* obj, void* frame) {
  static auto original = resolve<Parse>(parse4_name); original(obj, frame); parsed(obj, frame, 4);
}
extern "C" void parse2(void* obj, void* frame) {
  static auto original = resolve<Parse>(parse2_name); original(obj, frame); parsed(obj, frame, 2);
}
extern "C" bool online4(void* obj) { return is_online(obj, 4); }
extern "C" bool online2(void* obj) { return is_online(obj, 2); }
extern "C" void load_status(void* obj) {
  static auto original = resolve<Load>(load_name);
  // The vendor ANDs this flag with each motor's online() result and otherwise
  // never restores it. Recompute it every control cycle; preserve error codes.
  *(static_cast<unsigned char*>(obj) + 0x180) = 1;
  original(obj);
  const bool healthy = *(static_cast<unsigned char*>(obj) + 0x180) != 0;
  if (!healthy) {
    // Verified offset/type in the supported binary: atomic<int> mode at +4.
    // A restored link must not resume the previous POSITION_CONTROL target.
    reinterpret_cast<std::atomic<int>*>(static_cast<char*>(obj) + 4)->store(2);
    if (had_healthy && previous_healthy) epoch.fetch_add(1);
  }
  had_healthy |= healthy;
  previous_healthy = healthy;
}

extern "C" void arm_loop(void*) asm("_ZN3arx2x516ControllerThread9ArmThreadEv");
extern "C" void arm_loop(void* obj) {
  static auto update = resolve<Load>("_ZN3arx2x514ControllerBase6UpdateEv");
  static auto exchange = resolve<Load>("_ZN3arx2x516ControllerThread12exchangeDataEv");
  static auto destroy = resolve<Load>("_ZN3arx2x514ControllerBase11destoryFuncEv");
  // Verified in the hash-checked ArmThread: stop at +0x700, completion +0x701,
  // 200Hz Update/exchange, then destoryFunc on shutdown. Only pacing changes.
  auto stop = reinterpret_cast<std::atomic<bool>*>(static_cast<char*>(obj) + 0x700);
  auto done = reinterpret_cast<std::atomic<bool>*>(static_cast<char*>(obj) + 0x701);
  while (!stop->load()) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(5);
    update(obj);
    exchange(obj);
    // The SDK busy-waits on system_clock. A backward NTP step stalls motor
    // command/feedback processing until wall time catches up. Never use UTC
    // or resend missed cycles in a burst; preserve the original 200Hz cadence.
    std::this_thread::sleep_until(deadline);
  }
  destroy(obj);
  done->store(true);
}
