"""Cascade speech denoiser (classical DSP only, no ML).

Stage 0 : zero-phase high-pass (wind / rumble)
Stage 1 : sweeping-tone (siren/alarm) suppression, optional
Stage 2 : two-pass NON-CAUSAL noise PSD estimation over the whole clip
Stage 3 : decision-directed MMSE-LSA gain
Stage 4 : TSNR refinement (Plapous et al., 2006)
Stage 5 : musical-noise suppression (gain smoothing in low-gain regions)
Stage 6 : optional leak of the high-passed noisy signal

CLI:  python denoise.py --input_dir input --output_dir output
"""
import argparse
import json
import os

import numpy as np
import soundfile as sf
from scipy.ndimage import gaussian_filter1d, minimum_filter1d, uniform_filter, uniform_filter1d
from scipy.ndimage import maximum_filter1d, median_filter as _medfilt
from scipy.signal import butter, istft, sosfiltfilt, stft
from scipy.special import exp1

DEFAULTS = dict(
    hp_hz=60.0,
    n_fft=512, hop=128,
    min_win=120,
    min_bias=1.7,
    delta=3.0,
    noise_sigma=40.0,
    noise_scale=1.1,
    n_iter=2,
    alpha_dd=0.96,
    gain_floor_db=-18.0,
    tsnr=True,
    gain_smooth=True,
    leak=0.0,
    tonal_db=0.0,          # 0 = off
    tonal_max_cut_db=20.0,
    tonal_max_peaks=5,
    ceps_w=0.0,
    ceps_win=9,
    ceps_q_lo_ms=1.5,
    floor_pause_db=None,
    pause_thr=2.0,
    siren_w=0.0,
    siren_lo=500.0, siren_hi=2000.0, siren_fmax=7000.0,
    siren_med=41,
    siren_thr=6.0,
    siren_pers=0.6,
    siren_J=6, siren_lam=0.5,
    siren_bw=2,
    h_w=0.0,
    h_lo=0.4,
    h_pow=3.0,
    h_z0=2.5,
    h_fmax=2500.0,
)

# 튜닝 결과(best_params.json)가 이 파일 옆에 있으면 자동으로 기본값에 반영
_CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_params.json")
if os.path.exists(_CFG):
    DEFAULTS.update(json.load(open(_CFG)))


def _noise_psd(P, p):
    S = uniform_filter1d(P, 3, axis=0, mode="nearest")
    S = uniform_filter1d(S, 5, axis=1, mode="nearest")
    win = int(min(p["min_win"], P.shape[1]))
    n_min = minimum_filter1d(S, size=win, axis=1, mode="nearest") * p["min_bias"]
    n_min = np.maximum(gaussian_filter1d(n_min, 5, axis=1, mode="nearest"), 1e-14)

    noise = n_min.copy()
    for _ in range(int(p["n_iter"])):
        speech = (S > p["delta"] * noise).astype(float)
        presence = uniform_filter(speech, size=(3, 5), mode="nearest")
        w = (1.0 - presence) ** 2
        num = gaussian_filter1d(w * P, p["noise_sigma"], axis=1, mode="nearest")
        den = gaussian_filter1d(w, p["noise_sigma"], axis=1, mode="nearest")
        lam = np.clip(den / 0.3, 0.0, 1.0)
        avg = num / np.maximum(den, 1e-9)
        noise = np.maximum(lam * avg + (1 - lam) * n_min, 1e-14)
    return noise * p["noise_scale"]


def _tonal_suppress(x, sr, thr_db, max_cut_db, max_peaks):
    """Sparse, prominent, time-continuous spectral peaks (siren/alarm, also sweeping)
    are notched out. Voiced speech has many harmonic peaks per frame -> protected."""
    n_fft, hop = 1024, 128
    _, _, Y = stft(x, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
    mag_db = 20 * np.log10(np.abs(Y) + 1e-9)
    K, T = mag_db.shape
    pad = np.pad(mag_db, ((24, 24), (0, 0)), mode="edge")
    offs = [*range(-24, -5), *range(6, 25)]
    side = np.median(np.stack([pad[24 + o:24 + o + K] for o in offs]), axis=0)
    prom = mag_db - side
    peak = (mag_db >= np.roll(mag_db, 1, 0)) & (mag_db >= np.roll(mag_db, -1, 0)) & (prom >= thr_db)
    peak[:4] = False
    sparse = peak.sum(axis=0) <= max_peaks
    peak &= sparse[None, :]
    near = uniform_filter(peak.astype(float), size=(7, 1), mode="constant") > 0
    persistent = uniform_filter1d(near.astype(float), size=5, axis=1, mode="constant") >= 0.6
    cand = peak & persistent
    cut = np.where(cand, np.minimum(prom - thr_db + 6.0, max_cut_db), 0.0)
    cut = uniform_filter(cut, size=(5, 1), mode="constant") * 1.5
    cut = np.minimum(gaussian_filter1d(cut, 1.0, axis=1), max_cut_db)
    _, y = istft(Y * 10 ** (-cut / 20), fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
    y = y[:len(x)]
    return np.pad(y, (0, len(x) - len(y))) if len(y) < len(x) else y


def _ceps_xi(gamma, n_fft, sr, win, q_lo_ms, xi_min):
    """Cepstral smoothing of the ML SNR estimate (Breithaupt et al. 2007/2008, non-causal version)."""
    lg = np.log(np.maximum(gamma, 1e-6))
    c = np.fft.irfft(lg, n=n_fft, axis=0)
    n = n_fft
    q_lo = max(int(q_lo_ms * 1e-3 * sr), 2)
    keep = np.zeros_like(c, dtype=bool)
    keep[:q_lo + 1] = True
    keep[n - q_lo:] = True
    qa, qb = int(sr / 400), int(sr / 80)
    qb = min(qb, n // 2 - 3)
    qp = qa + np.argmax(c[qa:qb + 1], axis=0)
    T = c.shape[1]
    idx = np.arange(T)
    for d in range(-2, 3):
        for mult in (1, 2):
            q = np.clip(mult * qp + d, 0, n // 2)
            keep[q, idx] = True
            keep[n - q, idx] = True
    c_s = np.where(keep, c, uniform_filter1d(c, int(win), axis=1, mode="nearest"))
    lg_s = np.fft.rfft(c_s, axis=0).real
    gs = np.exp(lg_s + 0.5772156649)
    return np.maximum(gs - 1.0, xi_min)


def _siren_track(P, fr, p):
    """Track a (sweeping) harmonic siren/alarm: spectral-peak prominence -> harmonic-comb salience over a
    fundamental grid -> Viterbi path with a bounded sweep rate.  Returns (S, info); S = siren power estimate."""
    K, T = P.shape
    df = fr[1] - fr[0]
    base = _medfilt(P, size=(int(p["siren_med"]), 1), mode="nearest")
    prom = np.minimum(np.maximum(10 * np.log10(P / np.maximum(base, 1e-14)), 0.0), 25.0)
    prom = maximum_filter1d(prom, 3, axis=0)
    f0s = np.arange(p["siren_lo"], p["siren_hi"], df / 2.0)
    nC = len(f0s)
    num = np.zeros((nC, T)); den = np.zeros(nC)
    for h in range(1, int(p["siren_fmax"] / p["siren_lo"]) + 1):
        kb = np.rint(h * f0s / df).astype(int)
        km = np.rint((h + 0.5) * f0s / df).astype(int)
        ok = (km < K - 1) & ((h + 0.5) * f0s <= p["siren_fmax"])
        w = 1.0 / np.sqrt(h)
        num[ok] += w * (prom[kb[ok]] - prom[km[ok]])
        den[ok] += w
    sal = num / np.maximum(den, 1e-9)[:, None]
    J = int(p["siren_J"]); lam = p["siren_lam"]
    dp = np.zeros((nC, T)); bp = np.zeros((nC, T), dtype=int)
    dp[:, 0] = sal[:, 0]
    shifts = list(range(-J, J + 1))
    for t in range(1, T):
        cand = np.full((len(shifts), nC), -1e9)
        for si, d in enumerate(shifts):
            if d >= 0:
                cand[si, d:] = dp[:nC - d, t - 1] - lam * abs(d) if d else dp[:, t - 1]
            else:
                cand[si, :nC + d] = dp[-d:, t - 1] - lam * abs(d)
        best = np.argmax(cand, axis=0)
        dp[:, t] = sal[:, t] + cand[best, np.arange(nC)]
        bp[:, t] = np.arange(nC) - np.array(shifts)[best]
    path = np.zeros(T, dtype=int)
    path[-1] = int(np.argmax(dp[:, -1]))
    for t in range(T - 1, 0, -1):
        path[t - 1] = bp[path[t], t]
    ps = sal[path, np.arange(T)]
    on = uniform_filter1d((ps > p["siren_thr"]).astype(float), 5, mode="nearest") > 0.5
    info = dict(frac_on=float(on.mean()), ps_med=float(np.median(ps)), f0=f0s[path])
    S = np.zeros_like(P)
    if info["frac_on"] >= p["siren_pers"]:
        f0t = uniform_filter1d(f0s[path], 3, mode="nearest")
        hop_s = float(p["hop"]) / 16000.0
        rate = np.abs(np.gradient(f0t)) / hop_s
        twin = float(p["n_fft"]) / 16000.0
        for t in np.where(on)[0]:
            for h in range(1, int(p["siren_fmax"] / p["siren_lo"]) + 1):
                fc = h * f0t[t]
                if fc > p["siren_fmax"] or fc / df >= K - 1:
                    break
                half = (p["siren_bw"] * df + 0.5 * h * rate[t] * twin) / df
                lo_, hi_ = max(int(np.floor(fc / df - half)), 0), min(int(np.ceil(fc / df + half)) + 1, K)
                S[lo_:hi_, t] = np.maximum(P[lo_:hi_, t] - base[lo_:hi_, t], 0.0)
    return S, info


def siren_flag(x, sr=16000, **overrides):
    """진단용: 이 클립에서 사이렌이 검출되는가 -> (검출 여부, frac_on, ps_med)"""
    p = {**DEFAULTS, **overrides}
    x = np.asarray(x, dtype=np.float64)
    xh = sosfiltfilt(butter(4, p["hp_hz"], btype="highpass", fs=sr, output="sos"), x) if p["hp_hz"] > 0 else x
    n_fft, hop = int(p["n_fft"]), int(p["hop"])
    fr, _, Y = stft(xh, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
    S, info = _siren_track(np.abs(Y) ** 2 + 1e-14, fr, p)
    return info["frac_on"] >= p["siren_pers"], info["frac_on"], info["ps_med"]


def _harmonic_mult(P, fr, sr, n_fft, h_lo, h_pow, z0, fmax):
    """Pitch-based harmonic comb (cepstral F0 per frame). m(K,T) in [h_lo,1], voicing confidence v(T)."""
    c = np.fft.irfft(np.log(P), n=n_fft, axis=0)
    qa, qb = int(sr / 400), int(sr / 80)
    qb = min(qb, n_fft // 2 - 3)
    seg = c[qa:qb + 1]
    seg = seg - uniform_filter1d(seg, 15, axis=0, mode="nearest")
    Q, T = seg.shape
    idx = np.arange(T)
    i = np.argmax(seg, axis=0)
    peak = seg[i, idx]
    z = (peak - seg.mean(axis=0)) / (seg.std(axis=0) + 1e-9)
    y0, y2 = seg[np.clip(i - 1, 0, Q - 1), idx], seg[np.clip(i + 1, 0, Q - 1), idx]
    den = y0 - 2 * peak + y2
    delta = np.clip(np.where(np.abs(den) > 1e-12, 0.5 * (y0 - y2) / den, 0.0), -1, 1)
    q = _medfilt(qa + i + delta, size=5, mode="nearest")
    f0 = sr / q
    v = uniform_filter1d(np.clip((z - z0) / 2.0, 0.0, 1.0), 3, mode="nearest")
    comb = (0.5 * (1 + np.cos(2 * np.pi * fr[:, None] / f0[None, :]))) ** h_pow
    m = h_lo + (1 - h_lo) * comb
    m = np.where(fr[:, None] <= fmax, m, 1.0)
    return m, v


def _lsa(xi, gamma):
    r = xi / (1.0 + xi)
    v = np.maximum(r * gamma, 1e-8)
    return r * np.exp(0.5 * exp1(v))


def denoise_channel(x, sr=16000, **overrides):
    p = {**DEFAULTS, **overrides}
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < p["n_fft"] or not np.any(x):
        return x.copy()

    if p["hp_hz"] > 0:
        sos = butter(4, p["hp_hz"], btype="highpass", fs=sr, output="sos")
        xh = sosfiltfilt(sos, x)
    else:
        xh = x

    if p["tonal_db"] > 0:
        xh = _tonal_suppress(xh, sr, p["tonal_db"], p["tonal_max_cut_db"], int(p["tonal_max_peaks"]))

    n_fft, hop = int(p["n_fft"]), int(p["hop"])
    fr, _, Y = stft(xh, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
    P = np.abs(Y) ** 2 + 1e-14
    N = _noise_psd(P, p)
    if p["siren_w"] > 0:
        S_, _ = _siren_track(P, fr, p)
        N = N + p["siren_w"] * S_

    floor = 10 ** (p["gain_floor_db"] / 20)
    K, T = P.shape
    xi_min = 10 ** (-25 / 10)
    XC = None
    if p["ceps_w"] > 0:
        XC = _ceps_xi(np.minimum(P / N, 1000.0), n_fft, sr, p["ceps_win"], p["ceps_q_lo_ms"], xi_min)
    HM = None
    if p["h_w"] > 0:
        m_, v_ = _harmonic_mult(P, fr, sr, n_fft, p["h_lo"], p["h_pow"], p["h_z0"], p["h_fmax"])
        HM = 1.0 - p["h_w"] * v_[None, :] * (1.0 - m_)
    fp_db = p["floor_pause_db"]
    if fp_db is None:
        floor_t = np.full(T, floor)
    else:
        band = (fr >= 200) & (fr <= 3500)
        m = np.minimum(P / N, 30.0)[band].mean(axis=0)
        pres = np.clip((m - 1.0) / max(p["pause_thr"], 1e-3), 0.0, 1.0)
        pres = uniform_filter1d(pres, 7, mode="nearest")
        floor_t = 10 ** ((p["gain_floor_db"] * pres + fp_db * (1 - pres)) / 20)
    G = np.empty_like(P)
    prev = np.zeros(K)
    for t in range(T):
        gamma = np.minimum(P[:, t] / N[:, t], 1000.0)
        xi = np.maximum(p["alpha_dd"] * prev + (1 - p["alpha_dd"]) * np.maximum(gamma - 1, 0), xi_min)
        if XC is not None:
            w = p["ceps_w"]
            xi = np.exp((1 - w) * np.log(xi) + w * np.log(XC[:, t]))
        if HM is not None:
            xi = np.maximum(xi * HM[:, t], xi_min)
        g = np.clip(_lsa(xi, gamma), floor_t[t], 1.0)
        G[:, t] = g
        prev = g ** 2 * gamma

    if p["tsnr"]:
        xi2 = G ** 2 * P / N
        G = np.clip(xi2 / (1.0 + xi2), floor_t[None, :], 1.0)

    if p["gain_smooth"]:
        Gs = uniform_filter(G, size=(3, 3), mode="nearest")
        G = np.where(G < 0.4, Gs, G)

    _, y = istft(Y * G, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
    y = y[:n]
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    if p["leak"] > 0:
        y = y + p["leak"] * xh
    return np.nan_to_num(y)


def denoise(x, sr=16000, **overrides):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return denoise_channel(x, sr, **overrides)
    return np.stack([denoise_channel(x[:, c], sr, **overrides) for c in range(x.shape[1])], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="input")
    ap.add_argument("--output_dir", default="output")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    for name in sorted(os.listdir(args.input_dir)):
        if not name.lower().endswith(".wav"):
            continue
        src = os.path.join(args.input_dir, name)
        x, sr = sf.read(src, dtype="float64")
        y = np.clip(denoise(x, sr), -1.0, 1.0)
        assert y.shape == x.shape, (name, y.shape, x.shape)
        sf.write(os.path.join(args.output_dir, name), y, sr, subtype=sf.info(src).subtype)


if __name__ == "__main__":
    main()
