// Calls ONLY inspected SDK leaf parsers/cache exchange; no hardware objects.
#include "link_policy.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>
extern "C" void parse4(void*, void*) asm("_ZN3arx12hw_interface10MotorType410CanAnalyzeEPNS0_8CanFrameE");
extern "C" void parse2(void*, void*) asm("_ZN3arx12hw_interface10MotorType210CanAnalyzeEPNS0_8CanFrameE");
extern "C" bool online4(void*) asm("_ZN3arx12hw_interface10MotorType46onlineEv");
extern "C" bool online2(void*) asm("_ZN3arx12hw_interface10MotorType26onlineEv");
extern "C" void exchange4(void*) asm("_ZN3arx12hw_interface10MotorType416ExchangeMotorMsgEv");
extern "C" void construct4(void*, int) asm("_ZN3arx12hw_interface10MotorType4C1Ei");
void check(bool ok) { if (!ok) std::abort(); }
template<class T> void put(void* obj, size_t offset, T value) {
  std::memcpy(static_cast<char*>(obj)+offset, &value, sizeof(value));
}
struct Frame { uint32_t id; uint32_t length; unsigned char data[8]; };
int main() {
  wrist::validate_sdk(); wrist::configure(300, 800);
  alignas(16) std::array<char, 256> obj{};
  // Inspected motor-record constructor sets only scalar fields and its vtable;
  // it does not open a CAN socket, start a thread, or command hardware.
  construct4(obj.data(), 1);
  auto vtable=*reinterpret_cast<void***>(obj.data());
  auto virtual_online=reinterpret_cast<bool(*)(void*)>(vtable[10]);
  check(reinterpret_cast<void*>(virtual_online) == reinterpret_cast<void*>(&online4));
  Frame frame{1, 8, {0x20,0x7f,0x15,0x7f,0xf7,0xf9,0x73,0x84}};
  check(!online4(obj.data()));
  parse4(obj.data(), &frame); exchange4(obj.data());
  check(online4(obj.data()));
  double q; std::memcpy(&q, obj.data()+0x10, sizeof(q));
  check(q != 0); // Original SDK parsing still updates the position cache.
  std::this_thread::sleep_for(std::chrono::milliseconds(120));
  check(online4(obj.data())); // 100ms legacy timeout no longer latches offline.
  check(virtual_online(obj.data())); // SDK vtable dispatch uses the same fix.
  std::this_thread::sleep_for(std::chrono::milliseconds(220));
  check(!online4(obj.data()));
  parse4(obj.data(), &frame); exchange4(obj.data());
  check(online4(obj.data())); // Actual SDK + shim recovery, not a policy mock.
  // A different motor's frames must not postpone this motor's timeout.
  std::this_thread::sleep_for(std::chrono::milliseconds(310));
  frame.id=2; parse4(obj.data(), &frame); check(!online4(obj.data()));
  alignas(16) std::array<char, 256> obj2{};
  put(obj2.data(), 0x78, int(5));
  Frame dm{0,8,{0x15,0x7e,0xf6,0x80,0x07,0xfd,0x20,0x1d}};
  parse2(obj2.data(), &dm); check(online2(obj2.data()));
  auto health=wrist::snapshot(); check(!health.all_fresh);
  check(health.ages_ms[3]>=0 && health.ages_ms[4]<0);
  frame.id=1;
  parse4(obj.data(), &frame);
  std::atomic<bool> running{true};
  std::thread receive([&] {
    while(running.load()) {
      parse4(obj.data(), &frame);
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
  });
  for(int i=0;i<200000;++i) {
    check(online4(obj.data()));
    check(wrist::snapshot().ages_ms[0]>=0);
  }
  running.store(false); receive.join();
  std::cout << "200000 concurrent receipt/clock checks passed without false offline or negative age\n";
  std::cout << "Exact SDK: original parsing retained, short gap tolerated, true timeout and recovery passed; no hardware initialized\n";
}
