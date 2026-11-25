// The start of a request at its first sound, between the redraw (utterance samples, redraw.h)
// and the player (player samples): the audio is held until the first sample with |x| > 4 / 32768 (-78 dBFS) is found
// or 2 frames are decoded; the player gets it from 5 ms before that sample on, no fade; if no sample exceeds the
// threshold by then, nothing is cut. Only the start of a request; chunk boundaries and pauses are not touched.
// Player sample p = utterance sample p + cut(). The held audio is passed on split at the utterance's frame
// boundaries, so every frame start after the cut is the start of one write to the player (an upsampler behind it
// keeps its state per write). Include after redraw.h.
#pragma once

struct LeadCut : Playback {
    static constexpr float T = 4.f / 32768.f;
    static const int64_t PRE = 120, WAIT = 6400, FRAME = 3200;
    Playback &inner;
    std::vector<float> held;
    int64_t c = -1, s = -1, n = 0;   // cut point (-1: not known yet), first sound (-1: none found), utterance samples in

    explicit LeadCut(Playback &p) : inner(p) {}
    void reset() { held.clear(); c = s = -1; n = 0; }
    int64_t cut() const { return c; }

    void write(int64_t at, const float *x, int64_t k) override {   // at = n
        if (c >= 0) { inner.write(at - c, x, k); n = at + k; return; }
        held.insert(held.end(), x, x + k);
        for (int64_t i = n; i < n + k && s < 0; i++) if (std::fabs(held[(size_t)i]) > T) s = i;
        n += k;
        if (s >= 0) c = std::max<int64_t>(0, s - PRE);
        else if (n >= WAIT) c = 0;
        if (c >= 0) release();
    }
    // The end of the request: a request shorter than 2 frames without a sample above the threshold plays uncut.
    void finish() { if (c < 0) { c = 0; release(); } }
    // A swap (redraw.h): in player samples; refused if it would start inside the cut silence.
    bool replace(int64_t at, const float *x, int64_t k) override {
        if (c < 0) {   // still held: nothing played
            if (at > n) return false;
            held.resize((size_t)at);
            n = at;
            write(at, x, k);
            return true;
        }
        if (at < c) return false;
        const int64_t e = std::min(at + k, (at / FRAME + 1) * FRAME);   // the swap, then the rest frame by frame as usual
        if (!inner.replace(at - c, x, e - at)) return false;
        for (int64_t a = e; a < at + k;) {
            const int64_t b = std::min(at + k, (a / FRAME + 1) * FRAME);
            inner.write(a - c, x + (a - at), b - a);
            a = b;
        }
        n = at + k;
        return true;
    }
    double played(double dt) override { return inner.played(dt) + (double)(c >= 0 ? c : n); }   // in utterance samples
    void charge(int64_t frames, int64_t chars) override { inner.charge(frames, chars); }
    double frame_cost(double measured) override { return inner.frame_cost(measured); }

  private:
    void release() {   // the held audio from c on, in pieces that end at the utterance's frame boundaries
        for (int64_t a = c; a < n;) {
            const int64_t b = std::min(n, (a / FRAME + 1) * FRAME);
            inner.write(a - c, held.data() + a, b - a);
            a = b;
        }
        held.clear();
        held.shrink_to_fit();
    }
};
