# Cyclone GPU

Cyclone GPU is a compact OpenCL version of Cyclone for compressed secp256k1 public key prefix search.

This build was made especially for AMD Radeon cards and uses OpenCL instead of CUDA. It may also work on other OpenCL-capable GPUs, but AMD is the primary target.

It uses a GPU batch kernel named `thread-global-chain`, generates random private keys inside a puzzle or custom range, and verifies full matches on the CPU before writing a result file.

Use it only on ranges and keys you are authorized to test.

Important: Cyclone GPU searches by compressed public key only. The `-k` parameter must be a 66-character compressed public key beginning with `02` or `03`. Bitcoin addresses, HASH160 values, WIF/private keys, and uncompressed public keys are not accepted as search targets.

## Highlights

- GPU-only OpenCL search path
- Made especially for AMD Radeon OpenCL cards
- Compressed public key prefix matching only
- Puzzle ranges: `[2^(p-1), 2^p - 1]`
- Custom mask-aligned ranges
- Partial match reporting
- Full match verification on CPU
- `KEYFOUND.txt` output with checked count, elapsed time, and speed
- Selftest, benchmark, and profile modes
- Minimal bundled OpenCL headers/import library for Windows MinGW builds

## Project Layout

```text
.
|-- main_gpu.cpp          # Host OpenCL code, CLI, verification
|-- cyclone_kernel.cl     # Cyclone GPU kernels
|-- secp256k1.cl          # OpenCL secp256k1 math
|-- Int.* Point.*         # CPU bigint/point code used for setup and verification
|-- SECP256K1.*           # CPU GTable generation and match verification
|-- OpenCL/               # Minimal OpenCL headers and libOpenCL.a
|-- Makefile              # GPU-only build
`-- cyclone_gpu.exe       # Built executable, if already compiled
```

## Requirements

- Windows or Linux
- AMD Radeon GPU with a working OpenCL runtime
- Other OpenCL-capable GPUs may work, but are not the primary target

Windows build requirements:

- MinGW-w64/MSYS2 with `g++` and `mingw32-make`

Example MSYS2 packages:

```powershell
pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-make
```

Make sure `g++` and `mingw32-make` are available in `PATH`.

Linux build requirements:

- `g++`
- `make`
- OpenCL headers and loader library
- AMD OpenCL runtime, for example ROCm OpenCL or AMDGPU-PRO OpenCL

Ubuntu/Debian package names may look like this:

```bash
sudo apt install build-essential ocl-icd-opencl-dev
```

You still need a working GPU OpenCL runtime/driver from AMD.

## Build

From the project directory:

Windows:

```powershell
mingw32-make clean
mingw32-make
```

Linux:

```bash
make clean
make
```

The output file is:

```text
cyclone_gpu.exe on Windows
cyclone_gpu on Linux
```

## Usage

```powershell
.\cyclone_gpu.exe -k <compressed_public_key_hex> [-p <puzzle> | -r <startHex:endHex>] -b <prefix_bytes> [-w <work_size>]
```

Parameters:

```text
-k <hex>       Target compressed public key, 66 hex chars
-b <bytes>     Number of prefix bytes to compare, 1..33
-p <bits>      Puzzle range: 2^(bits-1) through 2^bits - 1
-r <a:b>       Custom range, startHex:endHex
-w <size>      OpenCL work size, default 262144
--bench <n>    Run n benchmark loops
--profile <n>  Time RNG, base multiply, and batch stages separately
--selftest     Compare known CPU/GPU points and a full match test
```

`-k` must be a compressed public key, for example:

```text
02f6a8148a62320e149cb15c544fe8a25ab483a0095d2280d03b8a00a7feada13d
```

Do not pass a Bitcoin address, HASH160, WIF key, or uncompressed public key.

Custom `-r` ranges currently must be mask-aligned: `max - min` must be `2^n - 1`.

## Recommended Settings

For real search, start with `-w 65536`:

```powershell
.\cyclone_gpu.exe -k 02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16 -p 135 -b 6 -w 65536
```

On an RX 6600 XT, `-w 65536` has reached about `1029 Mkeys/s` in real runs. Good values to compare:

```text
-w 65536
-w 131072
-w 262144
```

Use the value that gives the best stable speed on your GPU. Larger is not always faster; `-w 524288` was slower on the test RX 6600 XT.

Prefix length controls how much of the compressed public key is checked on the GPU:

```text
-b 4   very fast, more partial matches
-b 5   good balance
-b 6   recommended for long runs
-b 33  full compressed public key check, slower
```

The program still verifies full matches on the CPU, so a smaller `-b` can still find the real key. It may simply print more partial matches while searching.

## Example Output

Command:

```powershell
.\cyclone_gpu.exe -k 02f6a8148a62320e149cb15c544fe8a25ab483a0095d2280d03b8a00a7feada13d -p 35 -b 4 -w 65536
```

Example console output:

```text
=== Cyclone OpenCL GPU v1.0 ===
Preparing CPU : GTable
Platform      : AMD Accelerated Parallel Processing
Device        : AMD Radeon RX 6600 XT
Device type   : GPU
Compute units : 16
Max clock     : 2428 MHz
Global memory : 8176 MB
Target        : 02f6a8148a62320e...feada13d
Prefix bytes  : 4
Mode          : Random GPU
Batch kernel  : thread-global-chain
Puzzle        : 35
Range         : 400000000:7FFFFFFFF
Batches/loop  : 65536
Checked/loop  : 33554432
Search started.
Time          : 00:03 | Speed: 1667.73 Mkeys/s | Total: 43285217280
================== PARTIAL MATCH FOUND! ============
Prefix bytes  : 4
Private Key   : 00000000000000000000000000000000000000000000000000000004ddb7e1bc
Found PubKey  : 02f6a8149c07e9a1a7f3c7e979916a52f2c57228403a50ad150cf68afd8daea077
Target PubKey : 02f6a8148a62320e149cb15c544fe8a25ab483a0095d2280d03b8a00a7feada13d

================== FOUND MATCH! ====================
Match type    : FULL
Private Key   : 00000000000000000000000000000000000000000000000000000004aed21170
Public Key    : 02f6a8148a62320e149cb15c544fe8a25ab483a0095d2280d03b8a00a7feada13d
Target PubKey : 02f6a8148a62320e149cb15c544fe8a25ab483a0095d2280d03b8a00a7feada13d
```

`Mkeys/s` means millions of checked keys per second.

## Selftest

This checks GPU scalar multiplication, batch `base +/- 1`, and a known full match for private key `3`.

```powershell
.\cyclone_gpu.exe -k 02f9308a019258c31049344f85f89d5229b531c845836f99b08601f113bce036f9 -p 2 -b 33 --selftest
```

Expected final line includes:

```text
thread-global-chain batch found key ...0003 flag=2
```

## Benchmark

Use benchmark mode to tune `-w`:

```powershell
.\cyclone_gpu.exe -k 02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16 -p 40 -b 4 -w 65536 --bench 5
.\cyclone_gpu.exe -k 02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16 -p 40 -b 4 -w 131072 --bench 5
.\cyclone_gpu.exe -k 02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16 -p 40 -b 4 -w 262144 --bench 5
```

## Profile

Profile mode shows where time is spent:

```powershell
.\cyclone_gpu.exe -k 02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16 -p 40 -b 4 -w 65536 --profile 3
```

Typical output sections:

```text
RNG
Base multiply
Batch +/-
Read result
```

## Result File

During search, partial prefix matches are printed to the console. A full match writes `KEYFOUND.txt` with:

- private key
- public key
- target public key
- total checked count
- elapsed time
- speed

## Notes

- Prefix byte `1` checks only compressed key parity byte: `02` or `03`.
- Prefix byte `33` checks the full compressed public key.
- The GPU generates random keys inside the selected range.
- CPU code is still used for GTable setup and final candidate verification.

## Thanks

If this project helped you and you want to say thanks:

```text
BTC: bc1qa3c5xdc6a3n2l3w0sq3vysustczpmlvhdwr8vc
```
