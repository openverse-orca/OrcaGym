# Orca Sensor Runtime

This directory contains the versioned Runtime built by OrcaSensorHost. On
Linux x86_64 with glibc 2.35 or later, the normal `orca-gym` installation
provides these files. Users do not need a separate Runtime or vendor SDK.

| Path | Contents |
|---|---|
| `VERSION`, `compatibility/` | Supported SDK release and contract metadata |
| `lib/<platform>/` | Precompiled Sensor Host |
| `bin/<platform>/` | Complete contract-tool bundle, including dependencies and third-party notices |
| `licenses/` | Company agreement, SDK addendum, agreement source record and public SDK MIT license |
| `native-manifest.json` | File identities, sizes and SHA-256 checksums for native components and their license files |
| `runtime-manifest.json` | File inventory and checksums for the complete Runtime artifact |

The Python integration lives in `orca_gym.sensor.providers`. Use
`orca_gym.sensor.providers.native.default_host_path()` to locate the bundled
Host. The contract tool runs during registration and preparation, not during
physics steps. Simulations without provider sensors do not load these native
components.

Vendor headers, C++ samples and manuals are delivered separately in
OrcaSensorSDK. This integration consumes preassembled XML and vendor packages;
it does not assemble or visualize models.
Other platforms do not yet have a validated plugin Runtime; ordinary
OrcaGym features retain their own platform requirements.

## Licenses

OrcaGym Python source remains covered by its [MIT license](../../../LICENSE).
The Orca-owned Host and contract-tool binaries are covered by the
[company agreement](licenses/ORCA_BINARY_LICENSE.html) together with the
[SDK Binary License Addendum](licenses/ORCA_BINARY_LICENSE_ADDENDUM.md).
The addendum permits commercial development, automated builds and the
redistribution described there, and takes precedence over conflicting
provisions of the agreement.

Public SDK material is separately covered by
[MIT](licenses/PUBLIC_SDK_LICENSE.txt). The tool's third-party components keep
their own licenses in `bin/<platform>/licenses/`; the company terms do not
restrict rights granted by those licenses or MIT. Redistribute the tool with
its dependencies and all applicable notices intact.

Keep the two release manifests, artifact bytes and licenses together when
updating this directory. `setup.py` verifies them during packaging.
`ORCA_SENSOR_BUNDLE_NATIVE=0` builds a developer-only portable wheel without the
Host or tool; normal supported-platform builds include the complete Runtime.
