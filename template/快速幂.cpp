//快速幂
long long modPow(long long x, long long exp, long long mod) {
    long long res = 1;
    while (exp > 0) {
        if (exp & 1) res = res * x % mod;
        x = x * x % mod;
        exp >>= 1;
    }
    return res;
}