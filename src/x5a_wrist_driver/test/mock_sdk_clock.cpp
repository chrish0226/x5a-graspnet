// No controller constructors, sockets or motors: only the SDK loop's callees.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <time.h>
namespace {
std::atomic<int64_t> wall_offset{0};
int cycles=0, cleanups=0;
int64_t first=0, last=0, maximum_gap=0;
int64_t step=0;
int64_t mono() {
  timespec t{}; clock_gettime(CLOCK_MONOTONIC,&t);
  return int64_t(t.tv_sec)*1000000000+t.tv_nsec;
}
}
extern "C" void mock_reset(int64_t step_ns) {
  wall_offset=0; cycles=cleanups=0; first=last=maximum_gap=0; step=step_ns;
}
extern "C" int mock_cycles() { return cycles; }
extern "C" int mock_cleanups() { return cleanups; }
extern "C" int64_t mock_max_gap() { return maximum_gap; }
extern "C" void mock_update(void*) asm("_ZN3arx2x514ControllerBase6UpdateEv");
extern "C" void mock_exchange(void*) asm("_ZN3arx2x516ControllerThread12exchangeDataEv");
extern "C" void mock_destroy(void*) asm("_ZN3arx2x514ControllerBase11destoryFuncEv");
extern "C" std::chrono::system_clock::time_point mock_wall_now() asm("_ZNSt6chrono3_V212system_clock3nowEv");
extern "C" std::chrono::system_clock::time_point mock_wall_now() {
  return std::chrono::system_clock::time_point(std::chrono::nanoseconds(mono()+wall_offset.load()));
}
extern "C" void mock_update(void*) {
  const auto now=mono();
  if(last && now-last>maximum_gap) maximum_gap=now-last;
  last=now; if(!first) first=now;
  if(++cycles==1) wall_offset=step;
}
extern "C" void mock_exchange(void* obj) {
  if(cycles>=3) reinterpret_cast<std::atomic<bool>*>(static_cast<char*>(obj)+0x700)->store(true);
}
extern "C" void mock_destroy(void*) { ++cleanups; }
