#ifndef _SECP256K1_CL
#define _SECP256K1_CL

typedef struct {
    unsigned int v[8];
} uint256_t;

__constant const unsigned int _P[8] = {
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFC2F
};

__constant const unsigned int _P_MINUS1[8] = {
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFC2E
};

static inline unsigned int addc(unsigned int a, unsigned int b, unsigned int* carry)
{
    unsigned int r = a + b + *carry;
    *carry = (r < a) ? 1 : ((r == a && *carry) ? 1 : 0);
    return r;
}

static inline unsigned int subc(unsigned int a, unsigned int b, unsigned int* borrow)
{
    unsigned int r = a - b - *borrow;
    *borrow = (a < b) ? 1 : ((a == b && *borrow) ? 1 : 0);
    return r;
}

static inline void madd977(unsigned int* h, unsigned int* l, unsigned int a, unsigned int b)
{
    unsigned long r = (unsigned long)a * 977 + b;
    *l = (unsigned int)r;
    *h = (unsigned int)(r >> 32);
}

bool equal256k(uint256_t a, uint256_t b)
{
    for(int i = 0; i < 8; i++) {
        if(a.v[i] != b.v[i]) return false;
    }
    return true;
}

void subP(unsigned int a[8], unsigned int b[8])
{
    unsigned int borrow = 0;
    b[7] = subc(a[7], 0xFFFFFC2F, &borrow);
    b[6] = subc(a[6], 0xFFFFFFFE, &borrow);
    b[5] = subc(a[5], 0xFFFFFFFF, &borrow);
    b[4] = subc(a[4], 0xFFFFFFFF, &borrow);
    b[3] = subc(a[3], 0xFFFFFFFF, &borrow);
    b[2] = subc(a[2], 0xFFFFFFFF, &borrow);
    b[1] = subc(a[1], 0xFFFFFFFF, &borrow);
    b[0] = subc(a[0], 0xFFFFFFFF, &borrow);
}

void addP(unsigned int a[8], unsigned int b[8])
{
    unsigned int carry = 0;
    b[7] = addc(a[7], 0xFFFFFC2F, &carry);
    b[6] = addc(a[6], 0xFFFFFFFE, &carry);
    b[5] = addc(a[5], 0xFFFFFFFF, &carry);
    b[4] = addc(a[4], 0xFFFFFFFF, &carry);
    b[3] = addc(a[3], 0xFFFFFFFF, &carry);
    b[2] = addc(a[2], 0xFFFFFFFF, &carry);
    b[1] = addc(a[1], 0xFFFFFFFF, &carry);
    b[0] = addc(a[0], 0xFFFFFFFF, &carry);
}

bool greaterThanEqualToP(const unsigned int a[8])
{
    for(int i = 0; i < 8; i++) {
        if(a[i] > _P_MINUS1[i]) return true;
        if(a[i] < _P_MINUS1[i]) return false;
    }
    return true;
}

void multiply256(const unsigned int x[8], const unsigned int y[8], unsigned int out_high[8], unsigned int out_low[8])
{
    unsigned int z[16] = {0};
    for(int i = 7; i >= 0; i--) {
        unsigned int high = 0;
        for(int j = 7; j >= 0; j--) {
            unsigned long prod = (unsigned long)x[i] * y[j] + z[i + j + 1] + high;
            z[i + j + 1] = (unsigned int)prod;
            high = (unsigned int)(prod >> 32);
        }
        z[i] = high;
    }
    for(int i = 0; i < 8; i++) {
        out_high[i] = z[i];
        out_low[i] = z[8 + i];
    }
}

void mulModP(const unsigned int a[8], const unsigned int b[8], unsigned int product_low[8])
{
    unsigned int high[8];
    multiply256(a, b, high, product_low);

    unsigned int hWord = 0;
    unsigned int carry = 0;
    for(int i = 6; i >= 0; i--) {
        product_low[i] = addc(product_low[i], high[i + 1], &carry);
    }
    unsigned int p7 = addc(high[0], 0, &carry);
    unsigned int p6 = carry;

    carry = 0;
    for(int i = 7; i >= 0; i--) {
        unsigned int t = 0;
        madd977(&hWord, &t, high[i], hWord);
        product_low[i] = addc(product_low[i], t, &carry);
    }
    p7 = addc(p7, hWord, &carry);
    p6 = addc(p6, 0, &carry);

    carry = 0;
    product_low[6] = addc(product_low[6], p7, &carry);
    product_low[5] = addc(product_low[5], p6, &carry);
    for(int i = 4; i >= 0; i--) product_low[i] = addc(product_low[i], 0, &carry);
    unsigned int p7_2 = carry;

    carry = 0;
    hWord = 0;
    unsigned int t = 0;
    madd977(&hWord, &t, p7, 0);
    product_low[7] = addc(product_low[7], t, &carry);
    product_low[6] = addc(product_low[6], hWord, &carry);
    for(int i = 5; i >= 0; i--) product_low[i] = addc(product_low[i], 0, &carry);
    p7_2 = addc(p7_2, carry, &carry);

    if(p7_2 || greaterThanEqualToP(product_low)) {
        subP(product_low, product_low);
    }
}

uint256_t addModP256k(uint256_t a, uint256_t b)
{
    uint256_t c;
    unsigned int carry = 0;
    for(int i = 7; i >= 0; i--) c.v[i] = addc(a.v[i], b.v[i], &carry);
    if(carry || greaterThanEqualToP(c.v)) subP(c.v, c.v);
    return c;
}

uint256_t subModP256k(uint256_t a, uint256_t b)
{
    uint256_t c;
    unsigned int borrow = 0;
    for(int i = 7; i >= 0; i--) c.v[i] = subc(a.v[i], b.v[i], &borrow);
    if(borrow) addP(c.v, c.v);
    return c;
}

uint256_t mulModP256k(uint256_t a, uint256_t b)
{
    uint256_t c;
    mulModP(a.v, b.v, c.v);
    return c;
}

uint256_t squareModP256k(uint256_t a)
{
    return mulModP256k(a, a);
}

uint256_t invModP256k(uint256_t a)
{
    uint256_t res;
    res.v[0]=0; res.v[1]=0; res.v[2]=0; res.v[3]=0; res.v[4]=0; res.v[5]=0; res.v[6]=0; res.v[7]=1;
    uint256_t p = {{0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFC2F}};
    uint256_t exp = {{0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFC2D}};
    uint256_t base = a;
    for(int i = 0; i < 256; i++) {
        unsigned int bit = (exp.v[7 - (i/32)] >> (i%32)) & 1;
        if(bit) res = mulModP256k(res, base);
        base = mulModP256k(base, base);
    }
    return res;
}

#endif
