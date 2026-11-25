// 사과게임 고속 엔진 (DLL). 규칙은 apple_ai/rules.py 와 같다: 합이 10인 tight 직사각형.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>
#define API extern "C" __declspec(dllexport)

typedef uint64_t u64;
typedef std::chrono::steady_clock Clock;
static const int MAXC = 170, NC = 41, NF = 7, NP = NC + NF;

struct Mask {
    u64 a[3];
    bool hit(const Mask& o) const { return (a[0] & o.a[0]) | (a[1] & o.a[1]) | (a[2] & o.a[2]); }
};
struct Rng {
    u64 s;
    explicit Rng(u64 seed) : s(seed) { next(); next(); }
    u64 next() {
        u64 z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    double uni() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
};
struct Move {
    uint8_t r1, c1, r2, c2, gain, combo;
    uint8_t cells[10];
    Mask m;
    int code() const { return ((r1 * 17 + c1) * 10 + r2) * 17 + c2; }
};
struct Board {
    int R, C;
    uint8_t v[MAXC];
};
struct Policy {
    double logw[NC];   // 조합 로그 가중치
    double phi[NF];    // 박스 특징 계수
    double eps;        // 균등 혼합 비율
};

static u64 PR[10];
static std::unordered_map<u64, int> comboId;

API int params() { return NP; }

API void init(const int* combos) {   // NC x 10
    Rng g(12345);
    for (int d = 0; d < 10; d++) PR[d] = g.next() | 1;
    comboId.clear();
    for (int i = 0; i < NC; i++) {
        u64 k = 0;
        for (int d = 1; d < 10; d++) k += PR[d] * (u64)combos[i * 10 + d];
        comboId[k] = i;
    }
}

static Board mk(const uint8_t* cells, int R, int C) {
    Board b; b.R = R; b.C = C; memcpy(b.v, cells, R * C); return b;
}
static void apply(Board& b, const Move& m) { for (int i = 0; i < m.gain; i++) b.v[m.cells[i]] = 0; }
static int remaining(const Board& b) { int n = 0; for (int i = 0; i < b.R * b.C; i++) n += b.v[i] != 0; return n; }

static void enumerate(const Board& b, std::vector<Move>& out) {
    out.clear();
    const int R = b.R, C = b.C;
    int S[11][18], N[11][18];
    for (int c = 0; c <= C; c++) S[0][c] = N[0][c] = 0;
    for (int r = 0; r < R; r++) {
        int rs = 0, rn = 0;
        S[r + 1][0] = N[r + 1][0] = 0;
        for (int c = 0; c < C; c++) {
            int x = b.v[r * C + c];
            rs += x; rn += x != 0;
            S[r + 1][c + 1] = S[r][c + 1] + rs;
            N[r + 1][c + 1] = N[r][c + 1] + rn;
        }
    }
    for (int r1 = 0; r1 < R; r1++) for (int c1 = 0; c1 < C; c1++) {
        for (int r2 = r1; r2 < R; r2++) {
            const int *s1 = S[r1], *s2 = S[r2 + 1];
            const int base = s2[c1] - s1[c1];
            if (s2[c1 + 1] - s1[c1 + 1] - base > 10) break;
            for (int c2 = c1; c2 < C; c2++) {
                int s = s2[c2 + 1] - s1[c2 + 1] - base;
                if (s < 10) continue;
                if (s > 10) break;
                if (N[r1 + 1][c2 + 1] - N[r1 + 1][c1] - N[r1][c2 + 1] + N[r1][c1] == 0) continue;
                if (N[r2 + 1][c2 + 1] - N[r2 + 1][c1] - N[r2][c2 + 1] + N[r2][c1] == 0) continue;
                if (N[r2 + 1][c1 + 1] - N[r1][c1 + 1] - N[r2 + 1][c1] + N[r1][c1] == 0) continue;
                if (N[r2 + 1][c2 + 1] - N[r1][c2 + 1] - N[r2 + 1][c2] + N[r1][c2] == 0) continue;
                Move m;
                m.r1 = (uint8_t)r1; m.c1 = (uint8_t)c1; m.r2 = (uint8_t)r2; m.c2 = (uint8_t)c2;
                m.gain = 0; m.m.a[0] = m.m.a[1] = m.m.a[2] = 0;
                u64 k = 0;
                for (int r = r1; r <= r2; r++) for (int c = c1; c <= c2; c++) {
                    int i = r * C + c;
                    if (b.v[i]) { m.cells[m.gain++] = (uint8_t)i; k += PR[b.v[i]]; m.m.a[i >> 6] |= 1ull << (i & 63); }
                }
                m.combo = (uint8_t)comboId.find(k)->second;
                out.push_back(m);
            }
        }
    }
}

// 박스 특징: 빈칸수, 가장자리수, 긴변, 사라지는 다른 수, 고아가 되는 사과, 자기 칸의 대안 수, 진행도*사라지는 수
static void features(const Board& b, const std::vector<Move>& mv, std::vector<double>& F) {
    const int n = (int)mv.size();
    F.assign(n * NF, 0.0);
    int deg[MAXC] = {0};
    for (auto& m : mv) for (int i = 0; i < m.gain; i++) deg[m.cells[i]]++;
    const double t = 1.0 - (double)remaining(b) / (b.R * b.C);
    int cnt[MAXC] = {0};
    int touched[MAXC];
    for (int i = 0; i < n; i++) {
        const Move& m = mv[i];
        double* f = &F[i * NF];
        int h = m.r2 - m.r1 + 1, w = m.c2 - m.c1 + 1, kill = 0, orphan = 0, own = 0, nt = 0;
        for (int j = 0; j < n; j++) if (j != i && m.m.hit(mv[j].m)) {
            kill++;
            for (int k = 0; k < mv[j].gain; k++) {
                int c = mv[j].cells[k];
                if (m.m.a[c >> 6] >> (c & 63) & 1) continue;
                if (cnt[c]++ == 0) touched[nt++] = c;
            }
        }
        for (int k = 0; k < nt; k++) { orphan += cnt[touched[k]] == deg[touched[k]]; cnt[touched[k]] = 0; }
        for (int k = 0; k < m.gain; k++) own += deg[m.cells[k]] - 1;
        f[0] = (h * w - m.gain) / 5.0;
        f[1] = (m.r1 == 0) + (m.c1 == 0) + (m.r2 == b.R - 1) + (m.c2 == b.C - 1);
        f[2] = std::max(h, w) / 5.0;
        f[3] = kill / 5.0;
        f[4] = orphan / 3.0;
        f[5] = own / 5.0;
        f[6] = t * kill / 5.0;
    }
}

struct Work {   // 스레드별 작업 공간
    std::vector<Move> mv;
    std::vector<double> F, sc, p;
};
static bool usesF(const Policy& P) { for (int i = 0; i < NF; i++) if (P.phi[i] != 0) return true; return false; }

// 로그 점수 = logw[combo] - log(n_combo) + phi.f  (phi=0 이면 "조합 먼저, 박스 균등" 과 같은 분포)
static void logits(const Board& b, const Policy& P, Work& w, bool needF) {
    int n = (int)w.mv.size(), nc[NC] = {0};
    for (auto& m : w.mv) nc[m.combo]++;
    w.sc.resize(n);
    if (needF) features(b, w.mv, w.F);
    for (int i = 0; i < n; i++) {
        double s = P.logw[w.mv[i].combo] - std::log((double)nc[w.mv[i].combo]);
        if (needF) for (int k = 0; k < NF; k++) s += P.phi[k] * w.F[i * NF + k];
        w.sc[i] = s;
    }
}
static void softmax(const std::vector<double>& sc, std::vector<double>& p) {
    double mx = -1e300, z = 0;
    for (double s : sc) mx = std::max(mx, s);
    p.resize(sc.size());
    for (size_t i = 0; i < sc.size(); i++) z += p[i] = std::exp(sc[i] - mx);
    for (double& x : p) x /= z;
}
static int pick(const std::vector<double>& p, Rng& g) {
    double u = g.uni();
    for (size_t i = 0; i < p.size(); i++) if ((u -= p[i]) < 0) return (int)i;
    return (int)p.size() - 1;
}
static void mixEps(const Policy& P, Work& w) {
    if (P.eps <= 0) return;
    int nc[NC] = {0}, m = 0;
    for (auto& x : w.mv) if (nc[x.combo]++ == 0) m++;
    for (size_t i = 0; i < w.p.size(); i++) w.p[i] = (1 - P.eps) * w.p[i] + P.eps / m / nc[w.mv[i].combo];
}
static int step(const Board& b, const Policy& P, Work& w, Rng& g, bool uf) {   // 없으면 -1
    enumerate(b, w.mv);
    if (w.mv.empty()) return -1;
    logits(b, P, w, uf);
    softmax(w.sc, w.p);
    mixEps(P, w);
    return pick(w.p, g);
}
static int playout(Board b, const Policy& P, Work& w, Rng& g, bool uf, std::vector<Move>* path = nullptr) {
    int s = 0, i;
    while ((i = step(b, P, w, g, uf)) >= 0) {
        s += w.mv[i].gain;
        if (path) path->push_back(w.mv[i]);
        apply(b, w.mv[i]);
    }
    return s;
}
template <class Fn> static void parallel(int n, int threads, Fn fn) {
    std::atomic<int> next(0);
    std::vector<std::thread> t;
    for (int k = 0; k < threads; k++) t.emplace_back([&, k] { int i; while ((i = next++) < n) fn(i, k); });
    for (auto& x : t) x.join();
}
static int writePath(const std::vector<Move>& p, int* out) {
    for (size_t i = 0; i < p.size(); i++) { out[4 * i] = p[i].r1; out[4 * i + 1] = p[i].c1; out[4 * i + 2] = p[i].r2; out[4 * i + 3] = p[i].c2; }
    return (int)p.size();
}

API int enum_moves(const uint8_t* cells, int R, int C, int* out, const Policy* P, double* prob, double* feat) {
    Board b = mk(cells, R, C);
    Work w;
    enumerate(b, w.mv);
    if (w.mv.empty()) return 0;
    logits(b, *P, w, true);
    softmax(w.sc, w.p);
    mixEps(*P, w);
    for (size_t i = 0; i < w.mv.size(); i++) {
        out[5 * i] = w.mv[i].r1; out[5 * i + 1] = w.mv[i].c1; out[5 * i + 2] = w.mv[i].r2; out[5 * i + 3] = w.mv[i].c2;
        out[5 * i + 4] = w.mv[i].combo;
        prob[i] = w.p[i];
        for (int k = 0; k < NF; k++) feat[i * NF + k] = w.F[i * NF + k];
    }
    return (int)w.mv.size();
}

// 판마다 k 번 진행: 평균과 최고
API void games(const uint8_t* boards, int n, int R, int C, const Policy* P, int k, u64 seed, int threads, double* mean, int* best) {
    bool uf = usesF(*P);
    std::vector<Work> W(threads);
    parallel(n, threads, [&](int i, int t) {
        Board b = mk(boards + (size_t)i * R * C, R, C);
        Rng g(seed * 1000003 + i);
        double s = 0; int mx = 0;
        for (int j = 0; j < k; j++) { int x = playout(b, *P, W[t], g, uf); s += x; mx = std::max(mx, x); }
        mean[i] = s / k; best[i] = mx;
    });
}

// 반사실 기여도 정책 경사 (모든 수). out: n x NP 기울기, fis: n x NP 피셔 대각
API void pg(const uint8_t* boards, int n, int R, int C, const Policy* P, int K, u64 seed, int threads, double* out, double* fis) {
    bool uf = usesF(*P);
    std::vector<Work> W(threads), V(threads);
    memset(out, 0, sizeof(double) * n * NP);
    memset(fis, 0, sizeof(double) * n * NP);
    parallel(n, threads, [&](int i, int t) {
        Work &w = W[t], &v = V[t];
        Rng g(seed * 1000003 + i);
        Board b = mk(boards + (size_t)i * R * C, R, C);
        std::vector<Board> st;
        int a;
        while ((a = step(b, *P, w, g, uf)) >= 0) { st.push_back(b); apply(b, w.mv[a]); }
        if (st.empty()) return;
        double T = (double)st.size();
        b = st[g.next() % st.size()];
        enumerate(b, w.mv);
        int m = (int)w.mv.size();
        if (m < 2) return;
        logits(b, *P, w, true);
        softmax(w.sc, w.p);
        u64 base = g.next();
        std::vector<double> q(m);
        double qb = 0;
        for (int j = 0; j < m; j++) {
            Board nb = b; apply(nb, w.mv[j]);
            double s = 0;
            for (int k = 0; k < K; k++) { Rng h(base + k * 7919); s += playout(nb, *P, v, h, uf); }
            q[j] = w.mv[j].gain + s / K;
            qb += w.p[j] * q[j];
        }
        double e[NP] = {0}, e2[NP] = {0};
        double *o = out + (size_t)i * NP, *fi = fis + (size_t)i * NP;
        for (int j = 0; j < m; j++) {
            double p = w.p[j], adv = T * (1 - P->eps) * p * (q[j] - qb);
            o[w.mv[j].combo] += adv;
            e[w.mv[j].combo] += p; e2[w.mv[j].combo] += p;
            for (int k = 0; k < NF; k++) { double f = w.F[j * NF + k]; o[NC + k] += adv * f; e[NC + k] += p * f; e2[NC + k] += p * f * f; }
        }
        for (int k = 0; k < NP; k++) fi[k] = T * (1 - P->eps) * (e2[k] - e[k] * e[k]);
    });
}

// ---------- NRPA (스레드마다 독립, 최고를 취함) ----------
struct Shared {
    std::mutex mu; int best = -1; std::vector<Move> path; std::atomic<long long> runs{0};
    volatile int* progress;
    void offer(int s, const std::vector<Move>& p) {
        std::lock_guard<std::mutex> l(mu);
        if (s > best) { best = s; path = p; if (progress) *progress = s; }
    }
};
struct Step { std::vector<int> codes; std::vector<float> bias; int chosen; Move mv; };
struct Seq { int score = -1; std::vector<Step> st; };
struct Nrpa {
    const Policy& P; bool uf; Board b0; Rng g; Shared& S; std::vector<int> N; double alpha;
    Clock::time_point t0; double seconds; volatile int* cancel; Work w; int localBest = -1;
    bool over() { return std::chrono::duration<double>(Clock::now() - t0).count() >= seconds || (cancel && *cancel); }
    void run(const std::vector<float>& pol, Seq& q) {
        Board b = b0; q.score = 0; size_t k = 0;
        while (true) {
            enumerate(b, w.mv);
            if (w.mv.empty()) break;
            logits(b, P, w, uf);
            if (k == q.st.size()) q.st.emplace_back();
            Step& s = q.st[k++];
            int m = (int)w.mv.size();
            s.codes.resize(m); s.bias.resize(m);
            for (int i = 0; i < m; i++) { s.codes[i] = w.mv[i].code(); s.bias[i] = (float)w.sc[i]; w.sc[i] += pol[s.codes[i]]; }
            softmax(w.sc, w.p);
            s.chosen = pick(w.p, g); s.mv = w.mv[s.chosen];
            q.score += s.mv.gain;
            apply(b, s.mv);
        }
        q.st.resize(k);
        S.runs++;
        if (q.score > localBest) {
            localBest = q.score;
            std::vector<Move> p; for (auto& s : q.st) p.push_back(s.mv);
            S.offer(q.score, p);
        }
    }
    void adapt(std::vector<float>& pol, const Seq& q) {
        static thread_local std::vector<std::pair<int, float>> d;
        std::vector<double> sc, p;
        d.clear();
        for (auto& s : q.st) {
            sc.resize(s.codes.size());
            for (size_t i = 0; i < sc.size(); i++) sc[i] = s.bias[i] + pol[s.codes[i]];
            softmax(sc, p);
            for (size_t i = 0; i < sc.size(); i++) d.push_back({s.codes[i], (float)(alpha * (((int)i == s.chosen) - p[i]))});
        }
        for (auto& x : d) pol[x.first] += x.second;
    }
    void nest(int level, std::vector<float> pol, Seq& best) {
        Seq q;
        best.score = -1;
        for (int i = 0; i < N[level] && !over(); i++) {
            if (level == 1) run(pol, q); else nest(level - 1, pol, q);
            if (q.score >= best.score && q.score >= 0) best = q;
            if (best.score >= 0) adapt(pol, best);
        }
    }
};
API int nrpa(const uint8_t* cells, int R, int C, const Policy* P, double seconds, int threads, int level, int n1, int n2, int n3, double alpha,
             u64 seed, volatile int* cancel, volatile int* progress, int* outPath, int* outLen, double* stats) {
    auto t0 = Clock::now();
    Board b = mk(cells, R, C);
    Shared S; S.progress = progress;
    std::atomic<int> restarts(0);
    parallel(threads, threads, [&](int i, int) {
        Nrpa x{*P, usesF(*P), b, Rng(seed * 1000003 + i), S, {0, n1, n2, n3}, alpha, t0, seconds, cancel};
        while (!x.over()) { Seq q; x.nest(level, std::vector<float>(28900, 0.f), q); restarts++; }
    });
    *outLen = writePath(S.path, outPath); stats[0] = (double)S.runs; stats[1] = restarts;
    return S.best;
}
