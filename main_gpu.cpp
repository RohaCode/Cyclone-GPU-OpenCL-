#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <CL/cl.h>
#ifdef _WIN32
#include <windows.h>
#endif

#include "Int.h"
#include "Point.h"
#include "SECP256K1.h"
#include <memory>

typedef struct { unsigned int v[8]; } uint256_gpu_t;
typedef struct { uint256_gpu_t x, y; } point_affine_gpu_t;
typedef struct { uint64_t s0, s1, s2, s3; } xoshiro_state_gpu_t;
typedef struct {
    unsigned int flag;
    uint256_gpu_t privKey;
    uint256_gpu_t x;
    uint256_gpu_t y;
} cyclone_result_gpu_t;

static uint64_t splitmix64_seed(uint64_t& x) {
    uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static void checkCl(cl_int err, const char* what) {
    if(err != CL_SUCCESS) {
        std::ostringstream oss;
        oss << what << " failed with OpenCL error " << err;
        throw std::runtime_error(oss.str());
    }
}

static std::string readKernelSource(const std::string& filename) {
    std::ifstream file(filename);
    if(!file) {
        throw std::runtime_error("Cannot open kernel file: " + filename);
    }
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

static std::string getDeviceInfoString(cl_device_id device, cl_device_info param) {
    size_t size = 0;
    if(clGetDeviceInfo(device, param, 0, nullptr, &size) != CL_SUCCESS || size == 0) {
        return "";
    }

    std::vector<char> value(size + 1, 0);
    if(clGetDeviceInfo(device, param, size, value.data(), nullptr) != CL_SUCCESS) {
        return "";
    }
    return std::string(value.data());
}

static std::string getPlatformInfoString(cl_platform_id platform, cl_platform_info param) {
    size_t size = 0;
    if(clGetPlatformInfo(platform, param, 0, nullptr, &size) != CL_SUCCESS || size == 0) {
        return "";
    }

    std::vector<char> value(size + 1, 0);
    if(clGetPlatformInfo(platform, param, size, value.data(), nullptr) != CL_SUCCESS) {
        return "";
    }
    return std::string(value.data());
}

static std::string getDisplayDeviceName(cl_device_id device) {
    const cl_device_info CL_DEVICE_BOARD_NAME_AMD_LOCAL = 0x4038;
    std::string openclName = getDeviceInfoString(device, CL_DEVICE_NAME);
    std::string boardName = getDeviceInfoString(device, CL_DEVICE_BOARD_NAME_AMD_LOCAL);

    if(!boardName.empty()) {
        return boardName;
    }
    return openclName.empty() ? "Unknown OpenCL GPU" : openclName;
}

static std::string deviceTypeToString(cl_device_type type) {
    if(type & CL_DEVICE_TYPE_GPU) return "GPU";
    if(type & CL_DEVICE_TYPE_CPU) return "CPU";
    if(type & CL_DEVICE_TYPE_ACCELERATOR) return "ACCELERATOR";
    return "UNKNOWN";
}

static cl_ulong getDeviceInfoUlong(cl_device_id device, cl_device_info param) {
    cl_ulong value = 0;
    clGetDeviceInfo(device, param, sizeof(value), &value, nullptr);
    return value;
}

static cl_uint getDeviceInfoUint(cl_device_id device, cl_device_info param) {
    cl_uint value = 0;
    clGetDeviceInfo(device, param, sizeof(value), &value, nullptr);
    return value;
}

static void printDeviceSummary(cl_platform_id platform, cl_device_id device) {
    cl_device_type type = 0;
    clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(type), &type, nullptr);

    std::cout << "Platform      : " << getPlatformInfoString(platform, CL_PLATFORM_NAME) << "\n";
    std::cout << "Device        : " << getDisplayDeviceName(device) << "\n";
    std::cout << "Device type   : " << deviceTypeToString(type) << "\n";
    std::cout << "Compute units : " << getDeviceInfoUint(device, CL_DEVICE_MAX_COMPUTE_UNITS) << "\n";
    std::cout << "Max clock     : " << getDeviceInfoUint(device, CL_DEVICE_MAX_CLOCK_FREQUENCY) << " MHz\n";
    std::cout << "Global memory : " << (getDeviceInfoUlong(device, CL_DEVICE_GLOBAL_MEM_SIZE) / (1024 * 1024)) << " MB\n" << std::flush;
}

static Int hexToInt(const std::string& hex) {
    Int number;
    char buf[65] = {0};
    std::string padded = hex.size() >= 64 ? hex.substr(hex.size() - 64) : std::string(64 - hex.size(), '0') + hex;
    std::strncpy(buf, padded.c_str(), 64);
    number.SetBase16(buf);
    return number;
}

static std::string intToHex(const Int& value) {
    Int temp;
    temp.Set((Int*)&value);
    return temp.GetBase16();
}

static std::string compactHex(const Int& value) {
    std::string hex = intToHex(value);
    size_t first = hex.find_first_not_of('0');
    if(first == std::string::npos) {
        return "0";
    }
    return hex.substr(first);
}

static bool isPowerOfTwoMinusOneHex(std::string hex) {
    size_t first = hex.find_first_not_of('0');
    if(first == std::string::npos) {
        return false;
    }

    hex = hex.substr(first);
    const char lead = (char)std::tolower((unsigned char)hex[0]);
    if(lead != '1' && lead != '3' && lead != '7' && lead != 'f') {
        return false;
    }

    for(size_t i = 1; i < hex.size(); i++) {
        if(std::tolower((unsigned char)hex[i]) != 'f') {
            return false;
        }
    }
    return true;
}

static bool intLess(const Int& a, const Int& b) {
    Int aa;
    Int bb;
    aa.Set((Int*)&a);
    bb.Set((Int*)&b);
    return aa.IsLower(&bb);
}

static void intToGpu(const Int& src, uint256_gpu_t& dst) {
    for(int i = 0; i < 8; i++) {
        dst.v[i] = ((Int&)src).bits[7 - i];
    }
}

static std::string gpuToHex(const uint256_gpu_t& value) {
    std::ostringstream oss;
    for(int i = 0; i < 8; i++) {
        oss << std::hex << std::setw(8) << std::setfill('0') << value.v[i];
    }
    return oss.str();
}

static std::string pointToCompressedHex(const uint256_gpu_t& x, const uint256_gpu_t& y) {
    return std::string((y.v[7] & 1U) ? "03" : "02") + gpuToHex(x);
}

static std::string pointToCompressedHex(const Point& p) {
    uint256_gpu_t x;
    uint256_gpu_t y;
    intToGpu(p.x, x);
    intToGpu(p.y, y);
    return pointToCompressedHex(x, y);
}

static void printUsage(const char* programName) {
    std::cerr << "Usage: " << programName << " -k <public_key_hex> [-p <puzzle> | -r <startHex:endHex>] -b <prefix_length> [-w <work_size>]\n";
    std::cerr << "  -k : Target compressed public key (66 hex chars)\n";
    std::cerr << "  -b : Number of prefix bytes to compare (1-33)\n";
    std::cerr << "  -p : Puzzle size, searches [2^(p-1), 2^p - 1]\n";
    std::cerr << "  -r : Custom range startHex:endHex\n";
    std::cerr << "  -w : OpenCL global work size (default: 262144)\n";
    std::cerr << "  --bench <loops> : Run fixed kernel loops and print speed without stopping on match\n";
    std::cerr << "  --profile <loops> : Time rng, base multiplication, and batch stages separately\n";
    std::cerr << "  --selftest : Compare CPU/GPU points for puzzle 2 diagnostics\n";
}

static std::string formatElapsed(double elapsedSeconds) {
    uint64_t totalSeconds = (uint64_t)elapsedSeconds;
    uint64_t hours = totalSeconds / 3600;
    uint64_t minutes = (totalSeconds / 60) % 60;
    uint64_t seconds = totalSeconds % 60;

    std::ostringstream oss;
    oss << std::setfill('0');
    if(hours > 0) {
        oss << hours << ':';
    }
    oss << std::setw(2) << minutes << ':' << std::setw(2) << seconds;
    return oss.str();
}

static void printProgressLine(double elapsedSeconds, double speedMkeys, uint64_t totalChecked) {
    std::ostringstream oss;
    oss << "Time          : " << formatElapsed(elapsedSeconds)
        << " | Speed: " << std::fixed << std::setprecision(2)
        << speedMkeys << " Mkeys/s | Total: " << totalChecked;
    std::cout << '\r' << oss.str() << std::string(24, ' ') << std::flush;
}

static int getConsoleCursorY() {
#ifdef _WIN32
    CONSOLE_SCREEN_BUFFER_INFO info;
    if(GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &info)) {
        return (int)info.dwCursorPosition.Y;
    }
#endif
    return -1;
}

static void setConsoleCursorY(int y) {
#ifdef _WIN32
    if(y >= 0) {
        HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
        CONSOLE_SCREEN_BUFFER_INFO info;
        if(GetConsoleScreenBufferInfo(out, &info)) {
            COORD pos = info.dwCursorPosition;
            pos.X = 0;
            pos.Y = (SHORT)y;
            SetConsoleCursorPosition(out, pos);
        }
    }
#else
    (void)y;
#endif
}

static void clearConsoleLine() {
    std::cout << '\r' << std::string(120, ' ') << '\r';
}

static void printPartialBlock(int prefixLen, const std::string& privHex,
                              const std::string& foundPubHex,
                              const std::string& targetPubHex) {
    std::cout << "================== PARTIAL MATCH FOUND! ============\n";
    std::cout << "Prefix bytes  : " << prefixLen << "\n";
    std::cout << "Private Key   : " << privHex << "\n";
    std::cout << "Found PubKey  : " << foundPubHex << "\n";
    std::cout << "Target PubKey : " << targetPubHex << "\n";
}

int main(int argc, char* argv[]) {
    try {
        std::cout << "=== Cyclone OpenCL GPU v1.0 ===\n" << std::flush;

        std::string targetPubHex;
        std::string rangeArg;
        int prefixLen = 4;
        int puzzleBits = 0;
        bool puzzleProvided = false;
        bool rangeProvided = false;
        bool selfTest = false;
        int benchLoops = 0;
        int profileLoops = 0;
        size_t globalWorkSize = 16384;

        for(int i = 1; i < argc; i++) {
            if(!std::strcmp(argv[i], "-k") && i + 1 < argc) {
                targetPubHex = argv[++i];
            } else if(!std::strcmp(argv[i], "-b") && i + 1 < argc) {
                prefixLen = std::stoi(argv[++i]);
            } else if(!std::strcmp(argv[i], "-p") && i + 1 < argc) {
                puzzleBits = std::stoi(argv[++i]);
                puzzleProvided = true;
            } else if(!std::strcmp(argv[i], "-r") && i + 1 < argc) {
                rangeArg = argv[++i];
                rangeProvided = true;
            } else if(!std::strcmp(argv[i], "-w") && i + 1 < argc) {
                globalWorkSize = (size_t)std::stoull(argv[++i]);
            } else if(!std::strcmp(argv[i], "-t") && i + 1 < argc) {
                ++i;
            } else if(!std::strcmp(argv[i], "--selftest")) {
                selfTest = true;
            } else if(!std::strcmp(argv[i], "--thread-batch")) {
                // Kept as a no-op so old launch commands still work.
            } else if(!std::strcmp(argv[i], "--bench") && i + 1 < argc) {
                benchLoops = std::stoi(argv[++i]);
            } else if(!std::strcmp(argv[i], "--profile") && i + 1 < argc) {
                profileLoops = std::stoi(argv[++i]);
            } else {
                std::cerr << "Unknown parameter: " << argv[i] << "\n";
                printUsage(argv[0]);
                return 1;
            }
        }

        if(targetPubHex.size() != 66 || prefixLen <= 0 || prefixLen > 33 || (!puzzleProvided && !rangeProvided)) {
            printUsage(argv[0]);
            return 1;
        }

        Int minKey;
        Int maxKey;
        if(puzzleProvided) {
            if(puzzleBits <= 0 || puzzleBits > 256) {
                throw std::runtime_error("Invalid puzzle value. Must be between 1 and 256.");
            }
            Int one((uint64_t)1);
            minKey.Set(&one);
            minKey.ShiftL(puzzleBits - 1);
            maxKey.Set(&one);
            maxKey.ShiftL(puzzleBits);
            maxKey.Sub(&one);
        } else {
            const size_t colon = rangeArg.find(':');
            if(colon == std::string::npos) {
                throw std::runtime_error("Invalid range format. Expected startHex:endHex.");
            }
            minKey = hexToInt(rangeArg.substr(0, colon));
            maxKey = hexToInt(rangeArg.substr(colon + 1));
        }

        if(!intLess(minKey, maxKey)) {
            throw std::runtime_error("Range start must be less than range end.");
        }

        Int rangeSize;
        rangeSize.Sub(&maxKey, &minKey);
        if(!isPowerOfTwoMinusOneHex(intToHex(rangeSize))) {
            throw std::runtime_error("Current GPU random range requires max-min to be 2^n-1. Use -p for now, or provide a mask-aligned -r range.");
        }

        std::vector<unsigned char> targetPrefix(33, 0);
        for(int i = 0; i < 33; i++) {
            targetPrefix[i] = (unsigned char)std::stoul(targetPubHex.substr(i * 2, 2), nullptr, 16);
        }

        std::cout << "Preparing CPU : GTable\n" << std::flush;
        auto secp = std::make_unique<Secp256K1>();
        secp->Init();
        
        // 1. Full GTable for initial points (32 * 256 points)
        std::vector<point_affine_gpu_t> gpuGTable(32 * 256);
        for(int i = 0; i < 32 * 256; i++) {
            intToGpu(secp->GTable[i].x, gpuGTable[i].x);
            intToGpu(secp->GTable[i].y, gpuGTable[i].y);
        }

        // 2. Fast BatchTable for stepping (1024 points: 1G, 2G, ..., 1024G)
        const size_t batchWidth = 1024;
        std::vector<point_affine_gpu_t> gpuBatchTable(batchWidth);
        for(size_t i = 0; i < batchWidth; i++) {
            Int idx((uint64_t)(i + 1));
            Point nextP = secp->ComputePublicKey(&idx);
            intToGpu(nextP.x, gpuBatchTable[i].x);
            intToGpu(nextP.y, gpuBatchTable[i].y);
        }

        cl_int err = CL_SUCCESS;
        cl_platform_id platform = nullptr;
        cl_device_id device = nullptr;
        checkCl(clGetPlatformIDs(1, &platform, nullptr), "clGetPlatformIDs");
        checkCl(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr), "clGetDeviceIDs");

        printDeviceSummary(platform, device);

        cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        checkCl(err, "clCreateContext");
        cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
        checkCl(err, "clCreateCommandQueue");

        std::string source = readKernelSource("cyclone_kernel.cl");
        const char* srcPtr = source.c_str();
        size_t srcLen = source.size();
        cl_program program = clCreateProgramWithSource(context, 1, &srcPtr, &srcLen, &err);
        checkCl(err, "clCreateProgramWithSource");

        err = clBuildProgram(program, 1, &device, "-I . -cl-mad-enable -cl-fast-relaxed-math", nullptr, nullptr);
        if(err != CL_SUCCESS) {
            size_t logSize = 0;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
            std::vector<char> log(logSize + 1, 0);
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
            std::cerr << "Build Log:\n" << log.data() << "\n";
            return 1;
        }

        cl_kernel rngKernel = clCreateKernel(program, "cyclone_generate_base_keys_gpu", &err);
        checkCl(err, "clCreateKernel cyclone_generate_base_keys_gpu");
        cl_kernel startKernel = clCreateKernel(program, "cyclone_search_gpu", &err);
        checkCl(err, "clCreateKernel cyclone_search_gpu");
        cl_kernel threadBatchKernel = clCreateKernel(program, "cyclone_check_batch_thread_gpu", &err);
        checkCl(err, "clCreateKernel cyclone_check_batch_thread_gpu");
        cl_kernel debugScalarKernel = clCreateKernel(program, "cyclone_debug_scalar_gpu", &err);
        checkCl(err, "clCreateKernel cyclone_debug_scalar_gpu");
        cl_kernel debugBatchKernel = clCreateKernel(program, "cyclone_debug_batch_gpu", &err);
        checkCl(err, "clCreateKernel cyclone_debug_batch_gpu");

        if(globalWorkSize < 256) {
            globalWorkSize = 256;
        }
        if(globalWorkSize % 256 != 0) {
            globalWorkSize = ((globalWorkSize + 255) / 256) * 256;
        }

        const size_t groupsPerLaunch = globalWorkSize;
        const size_t checkedPerGroup = batchWidth * 2;
        std::vector<xoshiro_state_gpu_t> rngStates(groupsPerLaunch);
        uint64_t seedBase = ((uint64_t)std::random_device{}() << 32) ^ (uint64_t)std::random_device{}();
        for(size_t i = 0; i < groupsPerLaunch; i++) {
            uint64_t seed = seedBase + (uint64_t)i * 0x9e3779b97f4a7c15ULL;
            rngStates[i].s0 = splitmix64_seed(seed);
            rngStates[i].s1 = splitmix64_seed(seed);
            rngStates[i].s2 = splitmix64_seed(seed);
            rngStates[i].s3 = splitmix64_seed(seed);
        }
        uint256_gpu_t gpuMinKey;
        uint256_gpu_t gpuRangeMask;
        intToGpu(minKey, gpuMinKey);
        intToGpu(rangeSize, gpuRangeMask);
        cyclone_result_gpu_t zeroResult = {};
        cl_mem bufRngStates = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                             sizeof(xoshiro_state_gpu_t) * rngStates.size(), rngStates.data(), &err);
        checkCl(err, "clCreateBuffer rngStates");
        cl_mem bufKeys = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(uint256_gpu_t) * groupsPerLaunch, nullptr, &err);
        checkCl(err, "clCreateBuffer keys");
        cl_mem bufGTable = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          sizeof(point_affine_gpu_t) * gpuGTable.size(), gpuGTable.data(), &err);
        checkCl(err, "clCreateBuffer gTable");
        cl_mem bufBatchTable = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          sizeof(point_affine_gpu_t) * gpuBatchTable.size(), gpuBatchTable.data(), &err);
        checkCl(err, "clCreateBuffer batchTable");
        cl_mem bufPrefix = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          targetPrefix.size(), targetPrefix.data(), &err);
        checkCl(err, "clCreateBuffer targetPrefix");
        cl_mem bufStartPoints = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                               sizeof(point_affine_gpu_t) * groupsPerLaunch, nullptr, &err);
        checkCl(err, "clCreateBuffer startPoints");
        cl_mem bufResult = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                          sizeof(cyclone_result_gpu_t), &zeroResult, &err);
        checkCl(err, "clCreateBuffer result");
        const size_t chainEntries = groupsPerLaunch * (batchWidth - 1);
        const size_t chainBytes = sizeof(uint256_gpu_t) * chainEntries;
        const cl_ulong globalMem = getDeviceInfoUlong(device, CL_DEVICE_GLOBAL_MEM_SIZE);
        if(chainBytes > (size_t)(globalMem * 3 / 4)) {
            throw std::runtime_error("Batch chain buffer is too large for this GPU memory. Use a smaller -w value.");
        }
        cl_mem bufThreadChain = clCreateBuffer(context, CL_MEM_READ_WRITE, chainBytes, nullptr, &err);
        checkCl(err, "clCreateBuffer threadBatchChain");

        checkCl(clSetKernelArg(rngKernel, 0, sizeof(cl_mem), &bufRngStates), "clSetKernelArg rng 0");
        checkCl(clSetKernelArg(rngKernel, 1, sizeof(cl_mem), &bufKeys), "clSetKernelArg rng 1");
        checkCl(clSetKernelArg(rngKernel, 2, sizeof(uint256_gpu_t), &gpuMinKey), "clSetKernelArg rng 2");
        checkCl(clSetKernelArg(rngKernel, 3, sizeof(uint256_gpu_t), &gpuRangeMask), "clSetKernelArg rng 3");

        checkCl(clSetKernelArg(startKernel, 0, sizeof(cl_mem), &bufKeys), "clSetKernelArg start 0");
        checkCl(clSetKernelArg(startKernel, 1, sizeof(cl_mem), &bufGTable), "clSetKernelArg start 1");
        checkCl(clSetKernelArg(startKernel, 2, sizeof(cl_mem), &bufPrefix), "clSetKernelArg start 2");
        checkCl(clSetKernelArg(startKernel, 3, sizeof(int), &prefixLen), "clSetKernelArg start 3");
        checkCl(clSetKernelArg(startKernel, 4, sizeof(cl_mem), &bufStartPoints), "clSetKernelArg start 4");
        checkCl(clSetKernelArg(startKernel, 5, sizeof(cl_mem), &bufResult), "clSetKernelArg start 5");

        checkCl(clSetKernelArg(threadBatchKernel, 0, sizeof(cl_mem), &bufKeys), "clSetKernelArg thread batch 0");
        checkCl(clSetKernelArg(threadBatchKernel, 1, sizeof(cl_mem), &bufBatchTable), "clSetKernelArg thread batch 1");
        checkCl(clSetKernelArg(threadBatchKernel, 2, sizeof(cl_mem), &bufPrefix), "clSetKernelArg thread batch 2");
        checkCl(clSetKernelArg(threadBatchKernel, 3, sizeof(int), &prefixLen), "clSetKernelArg thread batch 3");
        checkCl(clSetKernelArg(threadBatchKernel, 4, sizeof(cl_mem), &bufStartPoints), "clSetKernelArg thread batch 4");
        checkCl(clSetKernelArg(threadBatchKernel, 5, sizeof(cl_mem), &bufResult), "clSetKernelArg thread batch 5");
        checkCl(clSetKernelArg(threadBatchKernel, 6, sizeof(cl_mem), &bufThreadChain), "clSetKernelArg thread batch 6");

        size_t localWorkSize = 256;

        if(selfTest) {
            std::cout << "Selftest      : CPU expected\n";
            for(int k = 1; k <= 3; k++) {
                Int key((uint64_t)k);
                Point p = secp->ComputePublicKey(&key);
                std::cout << "  CPU " << k << " : " << pointToCompressedHex(p) << "\n";
            }

            std::vector<uint256_gpu_t> debugKeys(3);
            std::vector<point_affine_gpu_t> debugOut(3);
            for(int k = 1; k <= 3; k++) {
                Int key((uint64_t)k);
                intToGpu(key, debugKeys[k - 1]);
            }

            cl_mem bufDebugKeys = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                 sizeof(uint256_gpu_t) * debugKeys.size(), debugKeys.data(), &err);
            checkCl(err, "clCreateBuffer debugKeys");
            cl_mem bufDebugOut = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                                sizeof(point_affine_gpu_t) * debugOut.size(), nullptr, &err);
            checkCl(err, "clCreateBuffer debugOut");

            checkCl(clSetKernelArg(debugScalarKernel, 0, sizeof(cl_mem), &bufDebugKeys), "clSetKernelArg debug scalar 0");
            checkCl(clSetKernelArg(debugScalarKernel, 1, sizeof(cl_mem), &bufGTable), "clSetKernelArg debug scalar 1");
            checkCl(clSetKernelArg(debugScalarKernel, 2, sizeof(cl_mem), &bufDebugOut), "clSetKernelArg debug scalar 2");
            size_t scalarGlobal = 3;
            checkCl(clEnqueueNDRangeKernel(queue, debugScalarKernel, 1, nullptr, &scalarGlobal, nullptr, 0, nullptr, nullptr),
                    "clEnqueueNDRangeKernel debug scalar");
            checkCl(clFinish(queue), "clFinish debug scalar");
            checkCl(clEnqueueReadBuffer(queue, bufDebugOut, CL_TRUE, 0,
                                        sizeof(point_affine_gpu_t) * debugOut.size(), debugOut.data(), 0, nullptr, nullptr),
                    "clEnqueueReadBuffer debug scalar");

            std::cout << "Selftest      : GPU scalar\n";
            for(int k = 1; k <= 3; k++) {
                std::cout << "  GPU " << k << " : " << pointToCompressedHex(debugOut[k - 1].x, debugOut[k - 1].y) << "\n";
            }

            std::vector<uint256_gpu_t> selftestBases(256);
            Int two((uint64_t)2);
            uint256_gpu_t base2;
            intToGpu(two, base2);
            for(size_t i = 0; i < selftestBases.size(); i++) {
                selftestBases[i] = base2;
            }
            checkCl(clEnqueueWriteBuffer(queue, bufKeys, CL_TRUE, 0, sizeof(uint256_gpu_t) * selftestBases.size(), selftestBases.data(), 0, nullptr, nullptr),
                    "clEnqueueWriteBuffer selftest base");
            size_t debugLocalWorkSize = 256;
            size_t oneGroup = 256;
            checkCl(clEnqueueNDRangeKernel(queue, startKernel, 1, nullptr, &oneGroup, &debugLocalWorkSize, 0, nullptr, nullptr),
                    "clEnqueueNDRangeKernel selftest start");

            checkCl(clSetKernelArg(debugBatchKernel, 0, sizeof(cl_mem), &bufKeys), "clSetKernelArg debug batch 0");
            checkCl(clSetKernelArg(debugBatchKernel, 1, sizeof(cl_mem), &bufGTable), "clSetKernelArg debug batch 1");
            checkCl(clSetKernelArg(debugBatchKernel, 2, sizeof(cl_mem), &bufStartPoints), "clSetKernelArg debug batch 2");
            checkCl(clSetKernelArg(debugBatchKernel, 3, sizeof(cl_mem), &bufDebugOut), "clSetKernelArg debug batch 3");
            size_t batchGlobal = 256;
            checkCl(clEnqueueNDRangeKernel(queue, debugBatchKernel, 1, nullptr, &batchGlobal, &debugLocalWorkSize, 0, nullptr, nullptr),
                    "clEnqueueNDRangeKernel debug batch");
            checkCl(clFinish(queue), "clFinish debug batch");
            checkCl(clEnqueueReadBuffer(queue, bufDebugOut, CL_TRUE, 0,
                                        sizeof(point_affine_gpu_t) * debugOut.size(), debugOut.data(), 0, nullptr, nullptr),
                    "clEnqueueReadBuffer debug batch");

            std::cout << "Selftest      : GPU batch base=2\n";
            std::cout << "  base   : " << pointToCompressedHex(debugOut[0].x, debugOut[0].y) << "\n";
            std::cout << "  base+1 : " << pointToCompressedHex(debugOut[1].x, debugOut[1].y) << "\n";
            std::cout << "  base-1 : " << pointToCompressedHex(debugOut[2].x, debugOut[2].y) << "\n";

            checkCl(clEnqueueWriteBuffer(queue, bufResult, CL_TRUE, 0, sizeof(zeroResult), &zeroResult, 0, nullptr, nullptr),
                    "clEnqueueWriteBuffer selftest active result");
            size_t threadBatchGlobal = batchWidth;
            checkCl(clEnqueueNDRangeKernel(queue, threadBatchKernel, 1, nullptr, &threadBatchGlobal, &localWorkSize, 0, nullptr, nullptr),
                    "clEnqueueNDRangeKernel selftest active batch");
            checkCl(clFinish(queue), "clFinish selftest active batch");
            cyclone_result_gpu_t activeResult = {};
            checkCl(clEnqueueReadBuffer(queue, bufResult, CL_TRUE, 0, sizeof(activeResult), &activeResult, 0, nullptr, nullptr),
                    "clEnqueueReadBuffer selftest active result");
            std::cout << "Selftest      : thread-global-chain batch found key " << gpuToHex(activeResult.privKey)
                      << " flag=" << activeResult.flag << "\n";

            clReleaseMemObject(bufDebugOut);
            clReleaseMemObject(bufDebugKeys);
            clReleaseKernel(debugBatchKernel);
            clReleaseKernel(debugScalarKernel);
            clReleaseMemObject(bufThreadChain);
            clReleaseMemObject(bufResult);
            clReleaseMemObject(bufStartPoints);
            clReleaseMemObject(bufPrefix);
            clReleaseMemObject(bufGTable);
            clReleaseMemObject(bufKeys);
            clReleaseMemObject(bufRngStates);
            clReleaseKernel(threadBatchKernel);
            clReleaseKernel(startKernel);
            clReleaseKernel(rngKernel);
            clReleaseProgram(program);
            clReleaseCommandQueue(queue);
            clReleaseContext(context);
            return 0;
        }

        if(profileLoops > 0) {
            double rngSeconds = 0.0;
            double startSeconds = 0.0;
            double batchSeconds = 0.0;
            double readSeconds = 0.0;
            uint64_t profileChecked = 0;

            for(int i = 0; i < profileLoops; i++) {
                auto t0 = std::chrono::high_resolution_clock::now();
                checkCl(clEnqueueWriteBuffer(queue, bufResult, CL_TRUE, 0, sizeof(zeroResult), &zeroResult, 0, nullptr, nullptr),
                        "clEnqueueWriteBuffer reset profile result");
                checkCl(clEnqueueNDRangeKernel(queue, rngKernel, 1, nullptr, &groupsPerLaunch, nullptr, 0, nullptr, nullptr),
                        "clEnqueueNDRangeKernel profile rng");
                checkCl(clFinish(queue), "clFinish profile rng");
                auto t1 = std::chrono::high_resolution_clock::now();

                checkCl(clEnqueueNDRangeKernel(queue, startKernel, 1, nullptr, &groupsPerLaunch, &localWorkSize, 0, nullptr, nullptr),
                        "clEnqueueNDRangeKernel profile start");
                checkCl(clFinish(queue), "clFinish profile start");
                auto t2 = std::chrono::high_resolution_clock::now();

                checkCl(clEnqueueNDRangeKernel(queue, threadBatchKernel, 1, nullptr, &groupsPerLaunch, &localWorkSize, 0, nullptr, nullptr),
                        "clEnqueueNDRangeKernel profile batch");
                checkCl(clFinish(queue), "clFinish profile batch");
                auto t3 = std::chrono::high_resolution_clock::now();

                cyclone_result_gpu_t result = {};
                checkCl(clEnqueueReadBuffer(queue, bufResult, CL_TRUE, 0, sizeof(result), &result, 0, nullptr, nullptr),
                        "clEnqueueReadBuffer profile result");
                auto t4 = std::chrono::high_resolution_clock::now();

                rngSeconds += std::chrono::duration<double>(t1 - t0).count();
                startSeconds += std::chrono::duration<double>(t2 - t1).count();
                batchSeconds += std::chrono::duration<double>(t3 - t2).count();
                readSeconds += std::chrono::duration<double>(t4 - t3).count();
                profileChecked += (uint64_t)groupsPerLaunch * checkedPerGroup;
            }

            const double totalSeconds = rngSeconds + startSeconds + batchSeconds + readSeconds;
            std::cout << "Profile loops : " << profileLoops << "\n";
            std::cout << "Checked       : " << profileChecked << "\n";
            std::cout << "Total speed   : " << std::fixed << std::setprecision(2)
                      << (double)profileChecked / totalSeconds / 1e6 << " Mkeys/s\n";
            std::cout << "RNG           : " << std::setprecision(4) << rngSeconds << " s ("
                      << std::setprecision(1) << (rngSeconds * 100.0 / totalSeconds) << "%)\n";
            std::cout << "Base multiply : " << std::setprecision(4) << startSeconds << " s ("
                      << std::setprecision(1) << (startSeconds * 100.0 / totalSeconds) << "%)\n";
            std::cout << "Batch +/-     : " << std::setprecision(4) << batchSeconds << " s ("
                      << std::setprecision(1) << (batchSeconds * 100.0 / totalSeconds) << "%)\n";
            std::cout << "Read result   : " << std::setprecision(4) << readSeconds << " s ("
                      << std::setprecision(1) << (readSeconds * 100.0 / totalSeconds) << "%)\n";

            clReleaseMemObject(bufResult);
            clReleaseMemObject(bufStartPoints);
            clReleaseMemObject(bufThreadChain);
            clReleaseMemObject(bufPrefix);
            clReleaseMemObject(bufGTable);
            clReleaseMemObject(bufKeys);
            clReleaseMemObject(bufRngStates);
            clReleaseKernel(debugBatchKernel);
            clReleaseKernel(debugScalarKernel);
            clReleaseKernel(threadBatchKernel);
            clReleaseKernel(startKernel);
            clReleaseKernel(rngKernel);
            clReleaseProgram(program);
            clReleaseCommandQueue(queue);
            clReleaseContext(context);
            return 0;
        }
        uint64_t totalChecked = 0;
        const auto start = std::chrono::high_resolution_clock::now();
        std::cout << "Target        : " << targetPubHex.substr(0, 16) << "..." << targetPubHex.substr(58) << "\n";
        std::cout << "Prefix bytes  : " << prefixLen << "\n";
        std::cout << "Mode          : Random GPU\n";
        std::cout << "Batch kernel  : thread-global-chain\n";
        if(puzzleProvided) {
            std::cout << "Puzzle        : " << puzzleBits << "\n";
        } else {
            std::cout << "Puzzle        : custom range\n";
        }
        std::cout << "Range         : " << compactHex(minKey) << ":" << compactHex(maxKey) << "\n";
        std::cout << "Batches/loop  : " << groupsPerLaunch << "\n";
        std::cout << "Checked/loop  : " << (groupsPerLaunch * checkedPerGroup) << "\n";
        if(benchLoops > 0) {
            std::cout << "Benchmark     : " << benchLoops << " loops\n";
        }
        std::cout << "Search started.\n";

        int completedLoops = 0;
        int partialBlockY = -1;
        while(true) {
            if(benchLoops > 0) {
                checkCl(clEnqueueWriteBuffer(queue, bufResult, CL_TRUE, 0, sizeof(zeroResult), &zeroResult, 0, nullptr, nullptr),
                        "clEnqueueWriteBuffer reset benchmark result");
            }

            checkCl(clEnqueueNDRangeKernel(queue, rngKernel, 1, nullptr, &groupsPerLaunch, nullptr, 0, nullptr, nullptr),
                    "clEnqueueNDRangeKernel rng");
            checkCl(clEnqueueNDRangeKernel(queue, startKernel, 1, nullptr, &groupsPerLaunch, &localWorkSize, 0, nullptr, nullptr),
                    "clEnqueueNDRangeKernel start");
            checkCl(clEnqueueNDRangeKernel(queue, threadBatchKernel, 1, nullptr, &groupsPerLaunch, &localWorkSize, 0, nullptr, nullptr),
                    "clEnqueueNDRangeKernel batch");
            checkCl(clFinish(queue), "clFinish");

            totalChecked += (uint64_t)groupsPerLaunch * checkedPerGroup;
            cyclone_result_gpu_t result = {};
            checkCl(clEnqueueReadBuffer(queue, bufResult, CL_TRUE, 0, sizeof(result), &result, 0, nullptr, nullptr),
                    "clEnqueueReadBuffer result");

            const auto now = std::chrono::high_resolution_clock::now();
            const double elapsed = std::chrono::duration<double>(now - start).count();
            const double speedMkeys = (double)totalChecked / elapsed / 1e6;
            printProgressLine(elapsed, speedMkeys, totalChecked);

            completedLoops++;
            if(benchLoops > 0 && completedLoops >= benchLoops) {
                std::cout << "\nBenchmark done.\n";
                break;
            }

            if(benchLoops == 0 && result.flag != 0) {
                const std::string privHex = gpuToHex(result.privKey);
                const std::string pubHex = pointToCompressedHex(result.x, result.y);
                Int verifyKey = hexToInt(privHex);
                Point verifyPoint = secp->ComputePublicKey(&verifyKey);
                const std::string cpuPubHex = pointToCompressedHex(verifyPoint);

                if(cpuPubHex != pubHex) {
                    std::cout << "\nWarning       : GPU candidate failed CPU verification, continuing search.\n";
                    checkCl(clEnqueueWriteBuffer(queue, bufResult, CL_TRUE, 0, sizeof(zeroResult), &zeroResult, 0, nullptr, nullptr),
                            "clEnqueueWriteBuffer reset invalid result");
                    continue;
                }

                const bool fullMatch = (cpuPubHex == targetPubHex);
                if(!fullMatch) {
                    clearConsoleLine();
                    if(partialBlockY < 0) {
                        partialBlockY = getConsoleCursorY();
                        printPartialBlock(prefixLen, privHex, cpuPubHex, targetPubHex);
                    } else {
                        const int progressY = getConsoleCursorY();
                        setConsoleCursorY(partialBlockY);
                        for(int i = 0; i < 5; i++) {
                            clearConsoleLine();
                            if(i < 4) {
                                std::cout << "\n";
                            }
                        }
                        setConsoleCursorY(partialBlockY);
                        printPartialBlock(prefixLen, privHex, cpuPubHex, targetPubHex);
                        setConsoleCursorY(progressY);
                    }
                    printProgressLine(elapsed, speedMkeys, totalChecked);
                    checkCl(clEnqueueWriteBuffer(queue, bufResult, CL_TRUE, 0, sizeof(zeroResult), &zeroResult, 0, nullptr, nullptr),
                            "clEnqueueWriteBuffer reset partial result");
                    continue;
                }

                std::cout << "\r" << std::string(96, ' ') << "\r";
                std::cout << "================== FOUND MATCH! ====================\n";
                std::cout << "Match type    : FULL\n";
                std::cout << "Private Key   : " << privHex << "\n";
                std::cout << "Public Key    : " << cpuPubHex << "\n";
                std::cout << "Target PubKey : " << targetPubHex << "\n";
                std::ofstream file("KEYFOUND.txt");
                if(file) {
                    file << "================== FOUND MATCH! ====================\n";
                    file << "Match type    : FULL\n";
                    file << "Mode          : Random GPU\n";
                    file << "Range         : " << compactHex(minKey) << ":" << compactHex(maxKey) << "\n";
                    file << "Prefix bytes  : " << prefixLen << "\n";
                    file << "Target PubKey : " << targetPubHex << "\n";
                    file << "Private Key   : " << privHex << "\n";
                    file << "Public Key    : " << cpuPubHex << "\n";
                    file << "Total Checked : " << totalChecked << "\n";
                    file << "Elapsed Time  : " << formatElapsed(elapsed) << "\n";
                    file << "Speed         : " << (double)totalChecked / elapsed / 1e6 << " Mkeys/s\n";
                }
                break;
            }
        }

        clReleaseMemObject(bufResult);
        clReleaseMemObject(bufStartPoints);
        clReleaseMemObject(bufThreadChain);
        clReleaseMemObject(bufPrefix);
        clReleaseMemObject(bufGTable);
        clReleaseMemObject(bufKeys);
        clReleaseMemObject(bufRngStates);
        clReleaseKernel(debugBatchKernel);
        clReleaseKernel(debugScalarKernel);
        clReleaseKernel(threadBatchKernel);
        clReleaseKernel(startKernel);
        clReleaseKernel(rngKernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 0;
    } catch(const std::exception& ex) {
        std::cerr << "\nError: " << ex.what() << "\n";
        return 1;
    }
}
