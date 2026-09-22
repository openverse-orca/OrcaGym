THIRD-PARTY RUNTIME NOTICES
===========================

Keep this licenses directory when redistributing orca-sensor-tool.
The original distribution-package copyright files are retained verbatim.
The upstream/ directory adds notices from the exact upstream versions,
with source URLs and SHA-256 digests in the accompanying JSON files.

Some distribution notices describe the entire source package, including
build scripts, documentation and tools that are not bundled here. Their
references to GPL/LGPL do not classify every runtime library as GPL/LGPL.
The component summaries below identify the bundled library's license;
the complete upstream and distribution notices remain authoritative.

PyInstaller 6.22.0: GPL-2.0-or-later with the bootloader exception;
runtime hooks use Apache-2.0. Both texts are in PyInstaller-COPYING.txt.
CPython standard-library bytecode in base_library.zip and the executable
is covered by the Python and incorporated-software notices below.

Bundled components:
- libbz2-1.0:amd64 1.0.8-5build1: BSD-style bzip2 license
  licenses/libbz2-1.0_amd64/copyright
  licenses/common/GPL-2
  licenses/upstream/bzip2/LICENSE.txt
  licenses/upstream/libbz2-1.0_amd64.json
- libexpat1:amd64 2.4.7-1ubuntu0.7: MIT
  licenses/libexpat1_amd64/copyright
  licenses/upstream/expat/COPYING.txt
  licenses/upstream/libexpat1_amd64.json
- libffi8:amd64 3.4.2-4: MIT
  licenses/libffi8_amd64/copyright
  licenses/common/GPL
  licenses/upstream/libffi/LICENSE.txt
  licenses/upstream/libffi8_amd64.json
- liblzma5:amd64 5.2.5-2ubuntu1: Public domain (liblzma); see XZ component-specific notice
  licenses/liblzma5_amd64/copyright
  licenses/common/GPL-2
  licenses/common/GPL-3
  licenses/common/LGPL-2
  licenses/common/LGPL-2.1
  licenses/upstream/xz/COPYING.txt
  licenses/upstream/liblzma5_amd64.json
- libmpdec3:amd64 2.5.1-2build2: BSD-2-Clause
  licenses/libmpdec3_amd64/copyright
  licenses/common/GPL-2
  licenses/upstream/mpdecimal/LICENSE.txt
  licenses/upstream/libmpdec3_amd64.json
- libpython3.10-minimal:amd64 3.10.12-1~22.04.15: PSF-2.0 and incorporated-software notices
  licenses/libpython3.10-minimal_amd64/copyright
  licenses/common/GPL-2
  licenses/upstream/cpython/LICENSE.txt
  licenses/upstream/libpython3.10-minimal_amd64.json
- libpython3.10-stdlib:amd64 3.10.12-1~22.04.15: PSF-2.0 and incorporated-software notices
  licenses/libpython3.10-stdlib_amd64/copyright
  licenses/common/GPL-2
  licenses/upstream/cpython/LICENSE.txt
  licenses/upstream/libpython3.10-stdlib_amd64.json
- libpython3.10:amd64 3.10.12-1~22.04.15: PSF-2.0 and incorporated-software notices
  licenses/libpython3.10_amd64/copyright
  licenses/common/GPL-2
  licenses/upstream/cpython/LICENSE.txt
  licenses/upstream/libpython3.10_amd64.json
- libssl3:amd64 3.0.2-0ubuntu1.23: Apache-2.0
  licenses/libssl3_amd64/copyright
  licenses/common/Apache-2.0
  licenses/common/Artistic
  licenses/common/GPL-1
  licenses/upstream/openssl/LICENSE.txt
  licenses/upstream/libssl3_amd64.json
- zlib1g:amd64 1:1.2.11.dfsg-2ubuntu9.2: Zlib
  licenses/zlib1g_amd64/copyright
  licenses/upstream/zlib/README.txt
  licenses/upstream/zlib1g_amd64.json

Changes made for this SDK distribution:
The following copies of third-party ELF binaries were modified by Orca
to use relative loader search paths (ELF RUNPATH). Upstream source code
was not changed by this packaging step. The original distribution's
package versions, including its patch revisions, are listed above.
- _internal/libbz2.so.1.0: RUNPATH set to $ORIGIN
- _internal/libcrypto.so.3: RUNPATH set to $ORIGIN
- _internal/libexpat.so.1: RUNPATH set to $ORIGIN
- _internal/libffi.so.8: RUNPATH set to $ORIGIN
- _internal/liblzma.so.5: RUNPATH set to $ORIGIN
- _internal/libmpdec.so.3: RUNPATH set to $ORIGIN
- _internal/libpython3.10.so.1.0: RUNPATH set to $ORIGIN
- _internal/libssl.so.3: RUNPATH set to $ORIGIN
- _internal/libz.so.1: RUNPATH set to $ORIGIN
- _internal/python3.10/lib-dynload/_bz2.cpython-310-x86_64-linux-gnu.so: RUNPATH set to $ORIGIN:$ORIGIN/../..
- _internal/python3.10/lib-dynload/_codecs_cn.cpython-310-x86_64-linux-gnu.so: RUNPATH set to $ORIGIN:$ORIGIN/../..
- _internal/python3.10/lib-dynload/_codecs_hk.cpython-310-x86_64-linux-gnu.so: RUNPATH set to $ORIGIN:$ORIGIN/../..
- _internal/python3.10/lib-dynload/_codecs_iso2022.cpython-310-x86_64-linux-gnu.so: RUNPATH set to $ORIGIN:$ORIGIN/../..
- _internal/python3.10/lib-dynload/_codecs_jp.cpython-310-x86_64-linux-gnu.so: RUNPATH set to $ORIGIN:$ORIGIN/../..
- _internal/python3.10/lib-dynload/_codecs_kr.cpython-310-x86_64-linux-gnu.so: RUNPATH set to $ORIGIN:$ORIGIN/../..
- _internal/python3.10/lib-dynload/_codecs_tw.cpython-310-x86_64-linux-gnu.so: RUNPATH set to $ORIGIN:$ORIGIN/../..
- _internal/python3.10/lib-dynload/_contextvars.cpython-310-x86_64-linux-gnu.so: RUNPATH set to $ORIGIN:$ORIGIN/../..
- _internal/python3.10/lib-dynload/_ctypes.cpython-310-x86_64-linux-gnu.so: RUNPATH set to $ORIGIN:$ORIGIN/../..
- _internal/python3.10/lib-dynload/_decimal.cpython-310-x86_64-linux-gnu.so: RUNPATH set to $ORIGIN:$ORIGIN/../..
- _internal/python3.10/lib-dynload/_hashlib.cpython-310-x86_64-linux-gnu.so: RUNPATH set to $ORIGIN:$ORIGIN/../..
- _internal/python3.10/lib-dynload/_json.cpython-310-x86_64-linux-gnu.so: RUNPATH set to $ORIGIN:$ORIGIN/../..
- _internal/python3.10/lib-dynload/_lzma.cpython-310-x86_64-linux-gnu.so: RUNPATH set to $ORIGIN:$ORIGIN/../..
- _internal/python3.10/lib-dynload/_multibytecodec.cpython-310-x86_64-linux-gnu.so: RUNPATH set to $ORIGIN:$ORIGIN/../..
- _internal/python3.10/lib-dynload/_opcode.cpython-310-x86_64-linux-gnu.so: RUNPATH set to $ORIGIN:$ORIGIN/../..
- _internal/python3.10/lib-dynload/_ssl.cpython-310-x86_64-linux-gnu.so: RUNPATH set to $ORIGIN:$ORIGIN/../..
- _internal/python3.10/lib-dynload/resource.cpython-310-x86_64-linux-gnu.so: RUNPATH set to $ORIGIN:$ORIGIN/../..
- _internal/python3.10/lib-dynload/termios.cpython-310-x86_64-linux-gnu.so: RUNPATH set to $ORIGIN:$ORIGIN/../..

System dependencies:
Libraries required from the operating system are not redistributed
merely because a program dynamically links to them.
Not bundled in this tool: libc.so.6, libgcc_s.so.1, libstdc++.so.6.
