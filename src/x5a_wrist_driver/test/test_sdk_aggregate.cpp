#include "link_policy.hpp"
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <new>
extern "C" int mock_sdk_retained();
extern "C" void load(void*) asm("_ZN3arx2x514ControllerBase15loadJointStatusEv");
void check(bool ok) { if (!ok) std::abort(); }
int main() {
  check(mock_sdk_retained()==1);
  alignas(16) std::array<unsigned char, 1024> obj{};
  auto mode=new(obj.data()+4) std::atomic<int>(5);
  // Preserve a sentinel where the SDK error vector lives.
  constexpr uint64_t sentinel=0x123456789abcdef0;
  std::memcpy(obj.data()+0x258, &sentinel, sizeof(sentinel));
  obj[0x180]=0; obj[0x188]=1; load(obj.data());
  check(obj[0x180]==1 && mode->load()==5);
  obj[0x188]=0; load(obj.data());
  check(obj[0x180]==0 && mode->load()==2);
  check(wrist::snapshot().protection_epoch==1);
  obj[0x188]=1; load(obj.data());
  check(obj[0x180]==1 && mode->load()==2);
  check(wrist::snapshot().protection_epoch==1);
  uint64_t saved; std::memcpy(&saved, obj.data()+0x258, sizeof(saved));
  check(saved==sentinel);
  std::cout << "Actual compatibility hook: aggregate recovers, mode stays PROTECT, fault memory untouched\n";
}
