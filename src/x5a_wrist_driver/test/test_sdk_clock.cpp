#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <new>
extern "C" void fixed_loop(void*) asm("_ZN3arx2x516ControllerThread9ArmThreadEv");
extern "C" void mock_reset(int64_t);
extern "C" int mock_cycles();
extern "C" int mock_cleanups();
extern "C" int64_t mock_max_gap();
extern "C" void mock_update(void*) asm("_ZN3arx2x514ControllerBase6UpdateEv");
extern "C" void mock_exchange(void*) asm("_ZN3arx2x516ControllerThread12exchangeDataEv");
extern "C" void mock_destroy(void*) asm("_ZN3arx2x514ControllerBase11destoryFuncEv");
void check(bool value) { if(!value) std::abort(); }
int64_t run(void(*loop)(void*), int64_t step) {
  alignas(16) std::array<char,2048> obj{};
  new(obj.data()+0x700) std::atomic<bool>(false);
  auto done=new(obj.data()+0x701) std::atomic<bool>(false);
  mock_reset(step);
  loop(obj.data());
  check(mock_cycles()==3 && mock_cleanups()==1 && done->load());
  return mock_max_gap();
}
int main() {
  // Confirm all hardware-capable callbacks resolve to the inert substitutes
  // before executing even the legacy SDK loop with a fake object.
  check(dlsym(RTLD_DEFAULT,"_ZN3arx2x514ControllerBase6UpdateEv")==reinterpret_cast<void*>(&mock_update));
  check(dlsym(RTLD_DEFAULT,"_ZN3arx2x516ControllerThread12exchangeDataEv")==reinterpret_cast<void*>(&mock_exchange));
  check(dlsym(RTLD_DEFAULT,"_ZN3arx2x514ControllerBase11destoryFuncEv")==reinterpret_cast<void*>(&mock_destroy));
  auto sdk=dlopen(SDK_PATH,RTLD_NOW|RTLD_LOCAL); check(sdk);
  auto legacy=reinterpret_cast<void(*)(void*)>(dlsym(sdk,"_ZN3arx2x516ControllerThread9ArmThreadEv"));
  check(legacy && legacy!=fixed_loop);
  // Simulate a 350ms backward wall-clock step; the host's clock is untouched.
  auto old_gap=run(legacy,-350000000);
  check(old_gap>=350000000);
  for(auto step:{-350000000LL,-2000000000LL,2000000000LL}) {
    auto gap=run(fixed_loop,step);
    check(gap>=4000000 && gap<100000000);
    std::cout << "wall step ns=" << step << " fixed max cycle gap ns=" << gap << '\n';
  }
  std::cout << "Original SDK stalls " << old_gap/1000000.0
            << "ms; monotonic hook preserves updates and stop/cleanup. No hardware initialized.\n";
}
