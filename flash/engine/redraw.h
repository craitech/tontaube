// Streaming redraw of flagged spots: the frame loop
// of fm.cpp, the VibeVoice decoder and the player behind abstract hooks, the detector of pitch_detect.h on the decoded
// audio. When the detector flags a spot, the generator goes back one frame before the spot's frame (state restored
// exactly: key/value cache rows, hidden state, autoencoder context, the next chunk's prepared prefix, the decoder's
// streaming state) and draws the frames again; at most 3 redraws, then the best take is kept. Only audio the player
// has not played yet is replaced: budget by the playback position before an attempt, the position checked after every
// attempt frame, and the swap itself checked inside the player; if refused, the original take stays and the generator
// continues from it. Include after fm.cpp.
#pragma once
#include "pitch_detect.h"

// The VibeVoice decoder as the redraw needs it: one latent frame -> 3200 samples, and its streaming state as bytes.
// In the browser: vv_dec.h.
struct FrameDecoder {
    virtual ~FrameDecoder() {}
    virtual bool decode(const float *lat, float *audio) = 0;
    virtual size_t state_size() const = 0;
    virtual void get_state(uint8_t *dst) const = 0;
    virtual void set_state(const uint8_t *src) = 0;
    // Optional: decode on the decoder's own threads while the caller goes on, then run `after` there (if any);
    // wait() before the audio is read. false: not supported (decode() is used).
    virtual bool start(const float *lat, float *audio, const std::function<void()> &after) { (void)lat; (void)audio; (void)after; return false; }
    // asynchronous decoders (the GPU): submit() starts the frame (audio lands in `audio`), wait() until it is there
    virtual bool submit(const float *lat, float *audio) { (void)lat; (void)audio; return false; }
    virtual void wait() {}
};

// Where the audio goes, in 24 kHz samples of the utterance. write(): the next samples, appended as made (at = the end).
// replace(): the swap of a redraw, in one step inside the player: everything from sample `at` on becomes x; refused
// (false, nothing changed) if `at` is played or less than the player's guard (one output block) ahead of it.
// played(dt): samples played or handed to the device dt seconds from now, kept on the safe side. charge(): the
// simulated clock of offline runs (frames generated, characters read); frame_cost(): seconds per generated frame
// for the budget.
struct Playback {
    virtual ~Playback() {}
    virtual void write(int64_t at, const float *x, int64_t n) = 0;
    virtual bool replace(int64_t at, const float *x, int64_t n) = 0;
    virtual double played(double dt) = 0;
    virtual void charge(int64_t frames, int64_t chars) { (void)frames; (void)chars; }
    virtual double frame_cost(double measured) { return measured; }
};

// A player that plays in real time from the first audio on, on a simulated clock: every generated frame costs
// 1 / (7.5 speed) s, every character of a chunk's text per_char / speed s. Keeps the audio it got.
// For checks: refuse = every swap refused; late = played() reports the position this many seconds behind the truth (the
// swap itself checks the true position); cost_scale < 1 = the budget expects attempts to be that much cheaper.
struct SimPlayback : Playback {
    double speed = 2.0, per_char = 0.004, guard = 0.05, clock = 0, play = 0, stall = 0, first_audio = -1, late = 0, cost_scale = 1;
    bool refuse = false;
    int64_t have = 0;   // samples the player has been given (the most it ever had)
    std::vector<float> audio;
    void write(int64_t at, const float *x, int64_t n) override {
        if (at != (int64_t)audio.size()) { fprintf(stderr, "SimPlayback: write at %lld, end %zu\n", (long long)at, audio.size()); exit(3); }
        audio.insert(audio.end(), x, x + n);
        have = std::max<int64_t>(have, (int64_t)audio.size());
    }
    bool replace(int64_t at, const float *x, int64_t n) override {
        if (refuse || (play + guard) * 24000.0 >= (double)at) return false;
        audio.resize((size_t)at);
        audio.insert(audio.end(), x, x + n);
        have = std::max<int64_t>(have, (int64_t)audio.size());
        return true;
    }
    void charge(int64_t frames, int64_t chars) override {
        const double dt = frames / (7.5 * speed) + chars * per_char / speed;
        clock += dt;
        if (first_audio >= 0) {
            double nxt = play + dt;
            const double avail = have / 24000.0;
            if (nxt > avail) { stall += nxt - avail; nxt = avail; }
            play = nxt;
        } else if (have > 0) {
            first_audio = clock;
            play = 0;
        }
    }
    double played(double dt) override { return (play - late + dt) * 24000.0; }
    double frame_cost(double) override { return cost_scale / (7.5 * speed); }
};

// A player without a speaker that starts with the first audio and plays in real time (wall clock); played() ignores
// stalls (on the safe side). Keeps the audio it got.
struct ClockPlayback : Playback {
    double t0 = -1, guard = 0.05;
    std::vector<float> audio;
    void write(int64_t at, const float *x, int64_t n) override {
        (void)at;
        audio.insert(audio.end(), x, x + n);
        if (t0 < 0 && !audio.empty()) t0 = now_s();
    }
    bool replace(int64_t at, const float *x, int64_t n) override {
        if (played(guard) >= (double)at) return false;
        audio.resize((size_t)at);
        audio.insert(audio.end(), x, x + n);
        return true;
    }
    double played(double dt) override { return t0 < 0 ? 0.0 : (now_s() + dt - t0) * 24000.0; }
    double lead() const { return t0 < 0 ? 0.0 : audio.size() / 24000.0 - (now_s() - t0); }   // seconds queued
};

struct RedrawStats {
    int64_t spots = 0, attempts = 0, clean[3] = {0, 0, 0}, kept_best = 0, budget = 0, previous_chunk = 0, extra_frames = 0,
            back0 = 0, frames = 0, early_stop = 0, played_stop = 0, refused = 0, det_off = 0;
    double det_s = 0, dec_s = 0, gen_s = 0;
};

struct Redraw {
    static const int NS = FM::NSNAP;
    static constexpr double SPF = 3200.0 / 120.0;   // detector steps per frame
    FM &m;
    FrameDecoder &dec;
    Playback &out;
    PitchDetect *det;     // null: redraw off (the plain frame loop)
    PitchDetect *det_cfg; // the configured detector; det may be switched off for one request (min_speed)
    // Played in real time and slower than min_speed x real time over the first check_frames frames: the detector is
    // off for the rest of the request (a redraw would rarely finish before its audio plays). 0: never.
    double min_speed = 1.3;
    int check_frames = 8;
    double dt_sum = 0;
    int64_t dt_n = 0;
    float cut = 0.85f;
    int max_redraws = 3;
    double margin = 0.05;  // seconds: one output block
    RedrawStats st;
    std::string log;       // one JSON object per spot, comma separated
    double t_frame = 0.07;  // measured seconds per frame (generation + decoder + detector), moving average
    // Pipelining (ahead, decoder with start()): while the decoder works on frame k, the frame model makes frame k + 1
    // (it needs only its own codes). The next next() takes it; a spot found before that undoes it exactly (rewind to
    // before it, noise generator back), so the result is the same as without.
    bool ahead = false;
    // Pipelined, an attempt frame is not overlapped (frame model, then decoder): its cost for the budget is measured
    // on the attempts (t_att); before the first one, t_frame + t_fm (the step made ahead), which is not below it.
    double t_fm = 0, t_att = -1;
    double attempt_cost() const { return !ahead ? t_frame : t_att > 0 ? t_att : t_frame + t_fm; }
    std::function<void(bool)> reading;  // called with true before a chunk's text is read, false before its frames
    struct Ahead { int64_t k = -1; bool ok = false; float lat[64]; std::mt19937_64 rng; Gauss gauss; } pf;
    int64_t n_ahead = 0, n_undone = 0;

    struct Take { FM::Take fm; int64_t lo = 0, K = 0; std::vector<uint8_t> dst; std::vector<float> audio; };
    std::vector<std::vector<uint8_t>> dstate;   // decoder state before frame k, at k % NS
    std::vector<int64_t> dkey;
    std::vector<std::vector<float>> faudio;     // audio of frame k, at k % NS
    int64_t handled = 0, scanned = 0;   // steps decided; final steps scanned and found below the cutoff
    std::vector<float> mv;

    Redraw(FM &m_, FrameDecoder &d, Playback &o, PitchDetect *det_) : m(m_), dec(d), out(o), det(det_), det_cfg(det_) {}

    int64_t K() const { return m.frames() - m.R.n_first - (pf.k >= 0 && pf.ok); }   // frames of the utterance's audio so far
    int64_t local(int64_t k) const { return k + m.R.n_first - m.chunk_first(); }

    // One utterance. The decoder must be in its start state (prime latents decoded) when this is called.
    FM::Result speak(const std::vector<Text> &chunks, float temp, std::mt19937_64 &rng) {
        st = RedrawStats();
        log.clear();
        handled = scanned = 0;
        pf.k = -1;
        n_ahead = n_undone = 0;
        det = det_cfg;
        dt_sum = 0;
        dt_n = 0;
        m.keep_snaps = det != nullptr;
        if (det) {
            det->reset();
            dstate.assign(NS, std::vector<uint8_t>(dec.state_size()));
            dkey.assign(NS, -1);
            faudio.assign(NS, std::vector<float>(3200));
        }
        const double d0 = det ? det->t_det : 0;
        m.begin(chunks, nullptr, false, temp, rng, nullptr);
        if (reading) reading(true);
        while (m.begin_chunk()) {
            if (reading) reading(false);
            int64_t chars = 0;
            for (auto &sg : m.g_segs) chars += (int64_t)sg.size();
            out.charge(0, chars);
            for (;;) {
                if (next(true)) {
                    if (det) judge(false);
                    if (det && dt_n == check_frames && min_speed > 0 && out.played(0) > -1e17) {
                        const double speed = (3200.0 / 24000.0) / out.frame_cost(dt_sum / dt_n);
                        if (speed < min_speed) { det = nullptr; m.keep_snaps = false; st.det_off = 1; }
                        dt_n++;   // checked once
                    }
                } else {
                    if (det) judge(true);    // the chunk's tail, before the next chunk starts
                    if (m.chunk_ended()) break;
                }
            }
            if (reading) reading(true);
        }
        if (reading) reading(false);
        if (det) det->finish();
        if (det_cfg) st.det_s = det_cfg->t_det - d0;
        m.keep_snaps = false;
        return m.finish();
    }

    // The next frame: generate, decode, detector, player (to_player false: an attempt, the player waits for the choice).
    bool next(bool to_player) {
        const double t0 = now_s();
        const int64_t k = K();
        if (det) { dec.get_state(dstate[k % NS].data()); dkey[k % NS] = k; }
        float lat[64];
        if (pf.k == k) {   // made while the decoder worked on the frame before
            pf.k = -1;
            if (!pf.ok) return false;
            memcpy(lat, pf.lat, sizeof lat);
        } else if (!m.step(lat)) {
            return false;
        }
        const double t1 = now_s();
        float *a = det ? faudio[k % NS].data() : nullptr;
        std::vector<float> tmp;
        if (!a) { tmp.resize(3200); a = tmp.data(); }
        bool pushed = false;  // the detector's push: on the decoder's thread after the decode when pipelined
        if (ahead && dec.start(lat, a, det ? std::function<void()>([this, a]() { det->push(a, 3200); }) : nullptr)) {
            if (to_player) {  // (a redraw attempt only waits: its next frame may not be wanted)
                pf.rng = *m.g_rng;
                pf.gauss = m.g_gauss;
                const double a0 = now_s();
                pf.ok = m.step(pf.lat);
                t_fm = t_fm > 0 ? 0.9 * t_fm + 0.1 * (now_s() - a0) : now_s() - a0;
                pf.k = k + 1;
                n_ahead++;
            }
            dec.wait();
            pushed = true;
        } else if (dec.submit(lat, a)) {  // the GPU: the next backbone step on the CPU meanwhile
            m.flush();
            dec.wait();
        } else if (!dec.decode(lat, a)) {
            fprintf(stderr, "decoder failed\n");
            exit(4);
        }
        const double t2 = now_s();
        if (det && !pushed) det->push(a, 3200);
        const double dt = now_s() - t0;   // without the player (it may wait while enough is queued)
        if (!ahead || to_player) t_frame = 0.9 * t_frame + 0.1 * dt;
        else t_att = t_att > 0 ? 0.9 * t_att + 0.1 * dt : dt;
        if (to_player && dt_n < check_frames) { dt_sum += dt; dt_n++; }
        if (to_player) out.write(k * 3200, a, 3200);
        out.charge(1, 0);
        st.frames++;
        st.gen_s += t1 - t0;
        st.dec_s += t2 - t1;
        return true;
    }

    // Back to the state before the frame made ahead (rewind to just before it, its noise draws given back).
    void undo_ahead() {
        if (pf.k < 0) return;
        const int64_t b = local(pf.k);
        if (!m.rewind(b, b)) { fprintf(stderr, "redraw: frame made ahead not kept\n"); exit(3); }
        if (m.R.hn.size() > (size_t)(m.chunk_first() + b) * m.H) m.R.hn.resize((size_t)(m.chunk_first() + b) * m.H);
        *m.g_rng = pf.rng;
        m.g_gauss = pf.gauss;
        pf.k = -1;
        n_undone++;
    }

    bool unplayed(int64_t b0, int64_t frames) { return out.played(frames * out.frame_cost(attempt_cost()) + margin) < (double)b0 * 3200.0; }

    // The highest spot measure over steps a..b-1 of the audio as it is now (as if it ended here).
    double score(int64_t a, int64_t b) {
        det->tentative(a, b, mv);
        float s = -1.f;
        for (float v : mv) s = std::max(s, v);
        return s;
    }

    void save(int64_t lo, Take &t) {
        m.save_take(local(lo), t.fm);
        t.lo = lo;
        t.K = K();
        t.dst.resize(dec.state_size() * (t.K - lo + 1));
        for (int64_t k = lo; k < t.K; k++) memcpy(t.dst.data() + dec.state_size() * (k - lo), dstate[k % NS].data(), dec.state_size());
        dec.get_state(t.dst.data() + dec.state_size() * (t.K - lo));
        t.audio.clear();
        for (int64_t k = lo; k < t.K; k++) t.audio.insert(t.audio.end(), faudio[k % NS].begin(), faudio[k % NS].end());
    }
    void load(const Take &t) {
        m.load_take(t.fm);
        for (int64_t k = t.lo; k < t.K; k++) {
            memcpy(dstate[k % NS].data(), t.dst.data() + dec.state_size() * (k - t.lo), dec.state_size());
            dkey[k % NS] = k;
            std::copy(t.audio.begin() + (k - t.lo) * 3200, t.audio.begin() + (k - t.lo + 1) * 3200, faudio[k % NS].begin());
        }
        dec.set_state(t.dst.data() + dec.state_size() * (t.K - t.lo));
        det->cut(t.lo * 3200);
        det->push(t.audio.data(), (int64_t)t.audio.size());
    }
    void rewind(int64_t b0, int64_t lo) {
        if (dkey[b0 % NS] != b0 || !m.rewind(local(b0), local(lo))) { fprintf(stderr, "redraw: frame %lld no longer kept\n", (long long)b0); exit(3); }
        dec.set_state(dstate[b0 % NS].data());
        det->cut(b0 * 3200);
    }

    // After each frame (tail: when the chunk has stopped): the earliest hot step not handled yet, and its redraw.
    void judge(bool tail) {
        const int64_t nk = K();
        const int64_t fe = (int64_t)((double)nk * SPF) - (tail ? 10 : PitchDetect::GUARD);
        const int64_t f1 = (int64_t)std::ceil(SPF);   // the first step of frame 1: frame 0 is never judged
        const int64_t lo = std::max<int64_t>({handled, scanned, f1});
        if (fe <= lo) return;
        int64_t t = -1;
        if (tail) {
            det->tentative(lo, fe, mv);
            for (int64_t i = 0; i < (int64_t)mv.size(); i++) if (mv[i] >= cut) { t = lo + i; break; }
        } else {
            if (fe > det->final_end()) { fprintf(stderr, "redraw: detector behind (%lld > %lld)\n", (long long)fe, (long long)det->final_end()); exit(3); }
            for (int64_t i = lo; i < fe; i++) if (det->measure(i) >= cut) { t = i; break; }
            if (t < 0) scanned = fe;   // final scores do not change (and the detector keeps only 5 s)
        }
        if (t < 0) return;
        undo_ahead();
        const int64_t f = (int64_t)((double)t / SPF), end_frame = f + 2;
        st.spots++;
        const int64_t c0 = std::max<int64_t>(0, m.chunk_first() - m.R.n_first);   // the chunk's first frame
        // scored range from L: new audio from frame b* on changes the scores up to 48 steps before it
        const int64_t b_star = f - 1 >= std::max<int64_t>(1, c0) ? f - 1 : f;
        const int64_t L = std::max<int64_t>(f1, std::min<int64_t>(t - 30, (int64_t)((double)b_star * SPF) - PitchDetect::GUARD));
        auto reg_end = [&](int64_t frames) { return (int64_t)((double)std::min(frames, end_frame + 1) * SPF) - PitchDetect::GUARD; };
        auto take_score = [&]() { return score(L, std::max<int64_t>(t + 1, reg_end(K()))); };
        double best_score = take_score();
        char b[200];
        snprintf(b, sizeof b, "%s{\"step\":%lld,\"frame\":%lld,\"chunk\":%lld,\"takes\":[%.3f", log.empty() ? "" : ",", (long long)t,
                 (long long)f, (long long)m.chunk_now, best_score);
        log += b;
        Take orig, best_take;
        int best = 0, ntakes = 1, cur = 0;   // takes: 0 = the original, then the attempts that were scored; cur: the take
                                             // the generator holds (-1: an attempt cut short by the playback)
        bool clean = false, refused = false;
        int64_t swap = -1;                   // the first attempt's b0: every attempt is swapped in from there
        const char *stop = "";
        std::string early_takes;             // takes stopped early, by index
        for (int k = 0; k < max_redraws; k++) {
            int64_t b0 = -1;
            int back = 0;
            for (int bb = 1; bb >= 0; bb--)
                if (f - bb >= std::max<int64_t>(1, c0) && unplayed(f - bb, end_frame - (f - bb) + 1)) { b0 = f - bb; back = bb; break; }
            if (b0 < 0) {
                const bool prev_chunk = f < c0 || (f == c0 && c0 == 0);
                (prev_chunk ? st.previous_chunk : st.budget)++;
                stop = prev_chunk ? "previous_chunk" : "budget";
                break;
            }
            if (k == 0) { swap = b0; save(swap, orig); }
            else if (cur != best) load(best == 0 ? orig : best_take);
            st.back0 += back == 0;
            rewind(b0, swap);
            st.attempts++;
            cur = -1;
            double sc = -1;
            bool early = false, too_late = false;
            while (K() <= end_frame && next(false)) {
                st.extra_frames++;
                if (out.played(margin) >= (double)swap * 3200.0) { too_late = true; break; }   // the swap point plays
                const int64_t fe2 = (int64_t)((double)K() * SPF) - PitchDetect::GUARD;          // this take already fails?
                if (fe2 > L) {
                    float hi = -1.f;
                    for (int64_t i = L; i < fe2; i++) hi = std::max(hi, det->measure(i));
                    if (hi >= cut) { sc = hi; early = true; break; }
                }
            }
            if (too_late) { st.played_stop++; stop = "played"; break; }
            if (early) st.early_stop++;
            else sc = take_score();
            snprintf(b, sizeof b, ",%.3f", sc);
            log += b;
            if (early) early_takes += (early_takes.empty() ? "" : ",") + std::to_string(ntakes);
            cur = ntakes++;
            if (sc < best_score) { best_score = sc; best = cur; }
            if (sc < (double)cut) { st.clean[k]++; clean = true; break; }
            if (best == cur && k + 1 < max_redraws) save(swap, best_take);
        }
        if (ntakes > 1 && !clean) st.kept_best++;
        if (swap >= 0) {
            // keep the lowest-scoring take; an attempt only if the player takes it in one checked swap, else the
            // original (what the player holds), and the generator continues from exactly that
            if (best != 0 && out.played(margin) >= (double)swap * 3200.0) refused = true;
            if (best != 0 && !refused) {
                if (cur != best) load(best_take);
                std::vector<float> x;
                for (int64_t k = swap; k < K(); k++) x.insert(x.end(), faudio[k % NS].begin(), faudio[k % NS].end());
                if (!out.replace(swap * 3200, x.data(), (int64_t)x.size())) refused = true;
            }
            if (refused) { st.refused++; best = 0; }
            if (best == 0 && cur != 0) load(orig);
        }
        snprintf(b, sizeof b, "],\"kept\":%d%s%s%s%s", best, *stop ? ",\"stop\":\"" : "", stop, *stop ? "\"" : "",
                 refused ? ",\"refused\":true" : "");
        log += b;
        log += early_takes.empty() ? "}" : ",\"early\":[" + early_takes + "]}";
        handled = std::max<int64_t>({handled, reg_end(K()), t + 1});
    }
};
