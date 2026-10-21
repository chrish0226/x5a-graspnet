// Substitute only the aggregate-AND behavior verified in SDK disassembly.
// No real controller constructor, solver, transport, thread or motor command.
extern "C" int mock_sdk_retained() { return 1; }
extern "C" void original_load(void*) asm("_ZN3arx2x514ControllerBase15loadJointStatusEv");
extern "C" void original_load(void* obj) {
  auto bytes=static_cast<unsigned char*>(obj);
  bytes[0x180]=bytes[0x180] && bytes[0x188];
}
