# Site provider integration tests

The CPU-only tests cover preassembled MJCF declarations, input projection, tool RPC,
shared physics state, native provider output, reset, failures and resource cleanup.
The fixtures contain no runtime binaries or model-assembly API.

Run with the `orca` interpreter:

```bash
<conda-base>/envs/orca/bin/python -m pytest tests/sensor_provider
```

Without `ORCA_SENSOR_TEST_SDK`, native integration tests explicitly skip. To run
those tests, set it to an independently released OrcaSensorSDK matching the host
platform. The fixture builds its contact-grid, rangefinder, touch-grid and seven-pad example plugins
in pytest's temporary directory. Tests use OrcaGym's bundled contract tool by
default; `ORCA_SENSOR_TOOL` remains an explicit override. The public environment
acceptance test supplies only provider manifests and verifies the bundled Host,
stepping, seeded reset and cleanup. Other integration tests retain explicit Host
paths to cover that override. No local sibling path is assumed, and no vendor
plugin binary is committed to this suite.
