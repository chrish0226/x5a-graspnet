#include "link_policy.hpp"
#include <cstdlib>
#include <iostream>
void check(bool ok) { if (!ok) std::abort(); }
int main() {
  using namespace wrist;
  check(!fresh(0, 1000, 300));
  check(fresh(1000, 1101, 300)); // An old 100ms-equivalent gap is tolerated.
  check(fresh(1000, 1300, 300));
  check(!fresh(1000, 1301, 300));
  check(fresh(1400, 1401, 300)); // Fresh data restores online, no stale latch.
  check(!fresh(1000, 999, 300));
  check(accepts_frame(4, 4, 4, 8, 0x20));
  check(!accepts_frame(4, 4, 2, 8, 0x20));
  check(!accepts_frame(4, 4, 4, 6, 0x20));
  check(accepts_frame(2, 5, 0, 8, 0x15));
  check(!accepts_frame(2, 5, 0, 8, 0x16));
  check(!accepts_frame(2, 5, 5, 8, 0x15));
  RecoveryGate gate;
  gate.observe(false, 0, 1000); check(!gate.protected_state);
  gate.observe(true, 0, 1100); check(!gate.stable(1299, 200));
  check(gate.stable(1300, 200));
  gate.observe(false, 1, 1500); check(gate.protected_state);
  gate.observe(true, 1, 1600); check(gate.protected_state);
  check(!gate.recover(1700, 0, 200, 500));
  check(!gate.recover(1800, 1700, 200, 500));
  check(gate.recover(2300, 1700, 200, 500));
  check(!gate.protected_state);
  // An outage and recovery between ROS timer ticks still latches protection.
  gate.observe(true, 2, 2400); check(gate.protected_state);
  std::cout << "Freshness boundaries, frame identity and explicit recovery passed\n";
}
