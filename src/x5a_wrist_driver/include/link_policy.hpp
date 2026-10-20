#pragma once
#include <array>
#include <cstdint>

namespace wrist {
constexpr std::array<int, 7> motor_ids{1, 2, 4, 5, 6, 7, 8};
inline int motor_index(int id) {
  for (int i = 0; i < 7; ++i) if (motor_ids[i] == id) return i;
  return -1;
}
inline bool fresh(int64_t received, int64_t now, int64_t timeout) {
  return received > 0 && now >= received && now - received <= timeout;
}
inline bool accepts_frame(int type, int motor_id, uint32_t can_id, uint32_t length, unsigned char first) {
  // This exact SDK's CanFrame uses uint32 id, uint32 length, uint8 data[8].
  if (length != 8 || motor_index(motor_id) < 0) return false;
  return type == 4 ? can_id == static_cast<uint32_t>(motor_id)
                   : type == 2 && can_id == 0 && (first & 15) == motor_id;
}
struct Snapshot {
  std::array<double, 7> ages_ms;
  bool all_fresh;
  uint64_t protection_epoch;
};
// Communication recovery and permission to send a new command are separate.
struct RecoveryGate {
  bool seen_healthy = false;
  bool protected_state = false;
  int64_t healthy_since = 0;
  uint64_t epoch = 0;
  void observe(bool healthy, uint64_t new_epoch, int64_t now) {
    if (new_epoch != epoch || (seen_healthy && !healthy)) {
      protected_state = true;
      healthy_since = 0;
    }
    epoch = new_epoch;
    if (!healthy) healthy_since = 0;
    else {
      seen_healthy = true;
      if (!healthy_since) healthy_since = now;
    }
  }
  bool stable(int64_t now, int64_t settle) const {
    return healthy_since > 0 && now >= healthy_since && now - healthy_since >= settle;
  }
  bool recover(int64_t now, int64_t last_command, int64_t settle, int64_t quiet) {
    if (!stable(now, settle) || (last_command > 0 && now - last_command < quiet)) return false;
    protected_state = false;
    return true;
  }
};
int64_t monotonic_ns();
void configure(double type4_timeout_ms, double type2_timeout_ms);
void validate_sdk();
Snapshot snapshot();
}
