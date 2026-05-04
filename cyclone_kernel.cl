#ifndef _CYCLONE_KERNEL_CL
#define _CYCLONE_KERNEL_CL

#include "secp256k1.cl"

typedef struct {
    uint256_t x;
    uint256_t y;
} point_affine_t;

typedef struct {
    uint256_t x;
    uint256_t y;
    uint256_t z;
} point_projective_t;

typedef struct {
    unsigned int flag;
    uint256_t privKey;
    uint256_t x;
    uint256_t y;
} cyclone_result_t;

typedef struct {
    ulong s0;
    ulong s1;
    ulong s2;
    ulong s3;
} xoshiro_state_t;

static inline ulong rotl64(ulong x, int k)
{
    return (x << k) | (x >> (64 - k));
}

static inline ulong xoshiro_next(__global xoshiro_state_t* state)
{
    ulong result = state->s0 + state->s3;
    ulong t = state->s1 << 17;

    state->s2 ^= state->s0;
    state->s3 ^= state->s1;
    state->s1 ^= state->s2;
    state->s0 ^= state->s3;
    state->s2 ^= t;
    state->s3 = rotl64(state->s3, 45);

    return result;
}

static inline unsigned int getKeyByte(uint256_t key, int byteIndex)
{
    // Scalar logic: v[7] is MSB word, v[0] is LSB word.
    // Each word is Little-Endian bytes: byte 0 is (v >> 0)
    return (key.v[7 - (byteIndex / 4)] >> ((byteIndex & 3) * 8)) & 0xff;
}

static inline point_projective_t addMixedCyclone(point_projective_t p1, point_affine_t p2)
{
    uint256_t u1 = mulModP256k(p2.y, p1.z);
    uint256_t v1 = mulModP256k(p2.x, p1.z);
    uint256_t u = subModP256k(u1, p1.y);
    uint256_t v = subModP256k(v1, p1.x);
    uint256_t us2 = squareModP256k(u);
    uint256_t vs2 = squareModP256k(v);
    uint256_t vs3 = mulModP256k(vs2, v);
    uint256_t us2w = mulModP256k(us2, p1.z);
    uint256_t vs2v2 = mulModP256k(vs2, p1.x);
    uint256_t twoVs2v2 = addModP256k(vs2v2, vs2v2);
    uint256_t a = subModP256k(us2w, vs3);
    a = subModP256k(a, twoVs2v2);

    point_projective_t r;
    r.x = mulModP256k(v, a);
    uint256_t vs3u2 = mulModP256k(vs3, p1.y);
    r.y = subModP256k(vs2v2, a);
    r.y = mulModP256k(r.y, u);
    r.y = subModP256k(r.y, vs3u2);
    r.z = mulModP256k(vs3, p1.z);
    return r;
}

static inline bool computePublicKeyProjectiveCyclone(uint256_t privKey, __global const point_affine_t* gTable, point_projective_t* outQ)
{
    int i = 0;
    unsigned int b = 0;

    for(i = 0; i < 32; i++) {
        b = getKeyByte(privKey, i);
        if(b != 0) break;
    }

    if(i >= 32) return false;

    point_projective_t q;
    q.x = gTable[i * 256 + (int)b - 1].x;
    q.y = gTable[i * 256 + (int)b - 1].y;
    q.z = (uint256_t){{0, 0, 0, 0, 0, 0, 0, 1}};

    for(i = i + 1; i < 32; i++) {
        b = getKeyByte(privKey, i);
        if(b != 0) {
            q = addMixedCyclone(q, gTable[i * 256 + (int)b - 1]);
        }
    }

    *outQ = q;
    return true;
}

static inline uint256_t one256()
{
    return (uint256_t){{0, 0, 0, 0, 0, 0, 0, 1}};
}

static inline uint256_t zero256()
{
    return (uint256_t){{0, 0, 0, 0, 0, 0, 0, 0}};
}

static inline uint256_t negModP256k(uint256_t a)
{
    if(equal256k(a, zero256())) return a;
    return subModP256k(zero256(), a);
}

static inline bool toAffineCyclone(point_projective_t* q, point_affine_t* aff) {
    if(equal256k(q->z, zero256())) return false;
    uint256_t invZ = invModP256k(q->z);
    aff->x = mulModP256k(q->x, invZ);
    aff->y = mulModP256k(q->y, invZ);
    return true;
}

static inline uint256_t add256Raw(uint256_t a, uint256_t b)
{
    unsigned int carry = 0;
    for(int i = 7; i >= 0; i--) {
        a.v[i] = addc(a.v[i], b.v[i], &carry);
    }
    return a;
}

static inline uint256_t randomMasked256(__global xoshiro_state_t* state, uint256_t mask)
{
    uint256_t value;
    for(int i = 0; i < 4; i++) {
        ulong r = xoshiro_next(state);
        value.v[i * 2] = (unsigned int)(r >> 32);
        value.v[i * 2 + 1] = (unsigned int)r;
    }
    for(int i = 0; i < 8; i++) value.v[i] &= mask.v[i];
    return value;
}

static inline bool compressedPrefixMatches(uint256_t x, uint256_t y, __constant const uchar* target, int prefixLen)
{
    // 1. Check parity byte (target[0])
    uchar parity = (y.v[7] & 1) ? (uchar)0x03 : (uchar)0x02;
    if(parity != target[0]) return false;

    // 2. Check X bytes if prefixLen > 1
    for(int i = 0; i < prefixLen - 1; i++) {
        int wordIdx = i / 4;
        int byteInWord = i % 4; // 0=MSB, 3=LSB within word
        // Coordinate logic: v[0] is MSB word, each word is Big-Endian
        uchar xb = (uchar)((x.v[wordIdx] >> ((3 - byteInWord) * 8)) & 0xff);
        if(xb != target[i + 1]) return false;
    }
    return true;
}

static inline bool xPrefixMatches(uint256_t x, __constant const uchar* target, int prefixLen)
{
    int xBytes = prefixLen - 1;
    int fullWords = xBytes / 4;
    int tailBytes = xBytes & 3;

    for(int i = 0; i < fullWords; i++) {
        // Construct targetWord in Big-endian order to match x.v[i]
        uint targetWord = ((uint)target[1 + i * 4] << 24) | ((uint)target[2 + i * 4] << 16) | ((uint)target[3 + i * 4] << 8) | (uint)target[4 + i * 4];
        if(x.v[i] != targetWord) return false;
    }

    if(tailBytes != 0) {
        uint mask = tailBytes == 1 ? 0xff000000U : (tailBytes == 2 ? 0xffff0000U : 0xffffff00U);
        uint targetWord = 0;
        int base = 1 + fullWords * 4;
        if(tailBytes >= 1) targetWord |= (uint)target[base] << 24;
        if(tailBytes >= 2) targetWord |= (uint)target[base + 1] << 16;
        if(tailBytes >= 3) targetWord |= (uint)target[base + 2] << 8;
        if((x.v[fullWords] & mask) != targetWord) return false;
    }
    return true;
}

static inline bool compressedFullMatches(uint256_t x, uint256_t y, __constant const uchar* target)
{
    return compressedPrefixMatches(x, y, target, 33);
}

static inline void storeMatchResult(uint256_t privKey, uint256_t x, uint256_t y, __constant const uchar* targetPrefix, __global cyclone_result_t* result)
{
    if(compressedFullMatches(x, y, targetPrefix)) {
        atomic_xchg((volatile __global unsigned int*)&result->flag, 2u);
        result->privKey = privKey;
        result->x = x;
        result->y = y;
    } else {
        if(atomic_cmpxchg((volatile __global unsigned int*)&result->flag, 0u, 1u) == 0u) {
            result->privKey = privKey;
            result->x = x;
            result->y = y;
        }
    }
}

static inline uint256_t addSmall256(uint256_t a, unsigned int value)
{
    unsigned int carry = value;
    for(int i = 7; i >= 0; i--) {
        unsigned int old = a.v[i];
        a.v[i] += carry;
        carry = (carry != 0 && a.v[i] < old) ? 1 : 0;
    }
    return a;
}

static inline uint256_t subSmall256(uint256_t a, unsigned int value)
{
    unsigned int borrow = value;
    for(int i = 7; i >= 0; i--) {
        unsigned int old = a.v[i];
        a.v[i] -= borrow;
        borrow = (borrow != 0 && old < borrow) ? 1 : 0;
    }
    return a;
}

__kernel void cyclone_search_gpu(
    __global xoshiro_state_t* rngStates,
    __global const point_affine_t* gTable,
    __constant const uchar* targetPrefix,
    const int prefixLen,
    __global point_affine_t* startPoints,
    __global cyclone_result_t* result,
    __global uint256_t* baseKeys,
    const uint256_t minKey,
    const uint256_t rangeMask)
{
    const int gid = get_global_id(0);

    uint256_t baseKey = randomMasked256(&rngStates[gid], rangeMask);
    baseKey = add256Raw(minKey, baseKey);
    baseKeys[gid] = baseKey;

    point_projective_t q;
    if(!computePublicKeyProjectiveCyclone(baseKey, gTable, &q)) return;

    point_affine_t aff;
    if(!toAffineCyclone(&q, &aff)) return;

    startPoints[gid] = aff;

    if(compressedPrefixMatches(aff.x, aff.y, targetPrefix, prefixLen)) {
        storeMatchResult(baseKey, aff.x, aff.y, targetPrefix, result);
    }
}

#define BATCH_SIZE 1024

__kernel void cyclone_check_batch_thread_gpu(
    __global const uint256_t* baseKeys,
    __global const point_affine_t* batchTable,
    __constant const uchar* targetPrefix,
    const int prefixLen,
    __global const point_affine_t* startPoints,
    __global cyclone_result_t* result,
    __global uint256_t* chain)
{
    const int gid = get_global_id(0);
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    const int dim = get_global_size(0);

    __local uint256_t loc_factors[256];
    __local uint256_t loc_prefixes[256];
    __local uint256_t loc_inverted[256];

    uint256_t startX = startPoints[gid].x;
    uint256_t startY = startPoints[gid].y;
    uint256_t negStartY = negModP256k(startY);

    uint256_t inverse = one256();
    for(int i = 1; i < BATCH_SIZE; i++) {
        point_affine_t offsetPoint = batchTable[i - 1];
        uint256_t denom = subModP256k(offsetPoint.x, startX);
        uint256_t factor = equal256k(denom, zero256()) ? one256() : denom;
        inverse = mulModP256k(inverse, factor);
        chain[(i - 1) * dim + gid] = inverse;
    }

    loc_factors[lid] = inverse;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(lid == 0) {
        uint256_t acc = one256();
        for(int i = 0; i < lsize; i++) {
            loc_prefixes[i] = acc;
            acc = mulModP256k(acc, loc_factors[i]);
        }
        uint256_t invAcc = invModP256k(acc);
        for(int i = lsize - 1; i >= 0; i--) {
            loc_inverted[i] = mulModP256k(invAcc, loc_prefixes[i]);
            invAcc = mulModP256k(invAcc, loc_factors[i]);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    inverse = loc_inverted[lid];

    if(result->flag == 2) return;

    uint256_t baseKey = baseKeys[gid];
    for(int i = BATCH_SIZE - 1; i >= 1; i--) {
        point_affine_t offsetPoint = batchTable[i - 1];
        uint256_t invDenom;
        uint256_t denom = subModP256k(offsetPoint.x, startX);
        uint256_t factor = equal256k(denom, zero256()) ? one256() : denom;

        if(i > 1) {
            invDenom = mulModP256k(inverse, chain[(i - 2) * dim + gid]);
            inverse = mulModP256k(inverse, factor);
        } else {
            invDenom = inverse;
        }

        uint256_t rise = subModP256k(offsetPoint.y, startY);
        uint256_t slope = mulModP256k(rise, invDenom);
        uint256_t slopeSq = mulModP256k(slope, slope);
        uint256_t plusX = subModP256k(subModP256k(slopeSq, startX), offsetPoint.x);

        if(prefixLen == 1 || xPrefixMatches(plusX, targetPrefix, prefixLen)) {
            uint256_t plusY = addModP256k(negStartY, mulModP256k(slope, subModP256k(startX, plusX)));
            if(compressedPrefixMatches(plusX, plusY, targetPrefix, prefixLen)) {
                storeMatchResult(addSmall256(baseKey, (unsigned int)i), plusX, plusY, targetPrefix, result);
            }
        }

        rise = negModP256k(addModP256k(offsetPoint.y, startY));
        slope = mulModP256k(rise, invDenom);
        slopeSq = mulModP256k(slope, slope);
        uint256_t minusX = subModP256k(subModP256k(slopeSq, startX), offsetPoint.x);

        if(prefixLen == 1 || xPrefixMatches(minusX, targetPrefix, prefixLen)) {
            uint256_t minusY = addModP256k(negStartY, mulModP256k(slope, subModP256k(startX, minusX)));
            if(compressedPrefixMatches(minusX, minusY, targetPrefix, prefixLen)) {
                storeMatchResult(subSmall256(baseKey, (unsigned int)i), minusX, minusY, targetPrefix, result);
            }
        }
    }
}

#endif
