// AMX-BF16 block of lin_bf16.h, included with the tile operations defined as macros: KERN_TCFG(cfg), KERN_TZERO(t),
// KERN_TLOAD(t, ptr, stride), KERN_TDP(c, a, b) (tdpbf16ps), KERN_TSTORE(t, ptr, stride), KERN_TREL() (tile numbers
// literal: the hardware forms are inline assembly).
//
// Tiles: C 0..3 fp32 [16 n][16 m], A 4, 5 the weight panel (16 rows x 32 k), B 6, 7 the activations (16 k pairs x 16 m
// x 2). Per output: C[n][m] += sum over k pairs in order of w[n][2p] x[m][2p] + w[n][2p + 1] x[m][2p + 1].
static void amx_block(const uint16_t *panel, int64_t K, const uint16_t *xb, int64_t Mp, int64_t M, const QW &W, int64_t n0,
                      float *const *yr) {
    struct alignas(64) Cfg { uint8_t palette, start; uint8_t pad[14]; uint16_t colsb[16]; uint8_t rows[16]; } cfg = {};
    cfg.palette = 1;
    for (int t = 0; t < 8; t++) { cfg.colsb[t] = 64; cfg.rows[t] = 16; }
    KERN_TCFG(&cfg);
    alignas(64) float c[32][32];
    const int64_t ldx = Mp * 4;  // bytes per k pair row of the activations
    for (int64_t m0 = 0; m0 < M; m0 += 32) {
        KERN_TZERO(0); KERN_TZERO(1); KERN_TZERO(2); KERN_TZERO(3);
        for (int64_t k0 = 0; k0 < K; k0 += 32) {
            KERN_TLOAD(4, panel + k0, K * 2);
            KERN_TLOAD(5, panel + 16 * K + k0, K * 2);
            KERN_TLOAD(6, xb + ((k0 / 2) * Mp + m0) * 2, ldx);
            KERN_TLOAD(7, xb + ((k0 / 2) * Mp + m0 + 16) * 2, ldx);
            KERN_TDP(0, 4, 6);
            KERN_TDP(1, 4, 7);
            KERN_TDP(2, 5, 6);
            KERN_TDP(3, 5, 7);
        }
        KERN_TSTORE(0, &c[0][0], 128);
        KERN_TSTORE(1, &c[0][16], 128);
        KERN_TSTORE(2, &c[16][0], 128);
        KERN_TSTORE(3, &c[16][16], 128);
        const int64_t mn = std::min<int64_t>(32, M - m0), nn = std::min<int64_t>(32, W.N - n0);
        for (int64_t m = 0; m < mn; m++)
            for (int64_t n = 0; n < nn; n++) yr[m0 + m][n0 + n] = c[n][m] + (W.bias ? W.bias[n0 + n] : 0.f);
    }
    KERN_TREL();
}
