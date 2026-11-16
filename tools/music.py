"""Synthesize the 30s background music for the Willog evolution video (original, royalty-free).

120 BPM, 1 bar = 2 s. Section boundaries line up with the video:
  0-4   intro  : soft pad + bell
  4-12  01 기록 : muted music-box arpeggio, half-time kick from 8 s
  12-20 02 실시간: full beat (kick, clap, hats, bass), brighter arps
  20-26 03 예측 : 16th arps + shimmer, riser and snare roll into 26 s
  26-30 ending : impact, lush Fmaj7 -> Cadd9 with bell chime on the logo (27.95 s)
writes ../bgm.wav (44.1 kHz, 16-bit stereo)
"""
import wave
from pathlib import Path
import numpy as np

SR = 44100
DUR = 30.0
N = int(SR * DUR)
BEAT = 0.5
BAR = 2.0
rng = np.random.default_rng(7)

buses = {k: np.zeros((2, N)) for k in ("pad", "main", "send")}


def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def tt(n):
    return np.arange(n) / SR


def add(sig, t0, gain=1.0, pan=0.0, bus="main", send=0.0):
    i = int(round(t0 * SR))
    if i >= N or i + len(sig) <= 0:
        return
    s = sig[: N - i] * gain
    l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
    buses[bus][0, i:i + len(s)] += s * l
    buses[bus][1, i:i + len(s)] += s * r
    if send:
        buses["send"][0, i:i + len(s)] += s * l * send
        buses["send"][1, i:i + len(s)] += s * r * send


def env_adsr(n, a, d, s, r):
    e = np.ones(n) * s
    na, nd, nr = int(a * SR), int(d * SR), int(r * SR)
    na = max(na, 1)
    e[:na] = np.linspace(0, 1, na)
    e[na:na + nd] = np.linspace(1, s, len(e[na:na + nd]))
    if nr:
        e[-nr:] *= np.linspace(1, 0, nr)
    return e


def saw(f, n, harms=14, bright=1.0):
    t = tt(n)
    out = np.zeros(n)
    for k in range(1, harms + 1):
        if k * f > 15000:
            break
        out += ((-1) ** (k + 1)) / k * np.exp(-(k - 1) / (harms * bright)) * np.sin(2 * np.pi * k * f * t)
    return out


def lowpass_ma(x, w):
    if w <= 1:
        return x
    return np.convolve(x, np.ones(w) / w, mode="same")


# ---------------- instruments ----------------
def pad_note(m, dur, bright):
    n = int((dur + 0.9) * SR)
    sig = sum(saw(mtof(m) * 2 ** (c / 1200), n, harms=10, bright=bright) for c in (-8, 0, 8)) / 3
    return sig * env_adsr(n, 0.45, 0.3, 0.85, 0.9)


def pluck(m, decay=7.0, n_sec=0.9):
    n = int(n_sec * SR)
    t = tt(n)
    f = mtof(m)
    s = np.sin(2 * np.pi * f * t) + 0.32 * np.sin(4 * np.pi * f * t) + 0.12 * np.sin(6 * np.pi * f * t)
    e = np.exp(-t * decay)
    e[: int(0.003 * SR)] *= np.linspace(0, 1, int(0.003 * SR))
    return s * e


def bell(m, n_sec=2.5):
    n = int(n_sec * SR)
    t = tt(n)
    f = mtof(m)
    s = np.sin(2 * np.pi * f * t) + 0.45 * np.sin(2 * np.pi * f * 2.76 * t) * np.exp(-t * 3) + 0.2 * np.sin(2 * np.pi * f * 5.4 * t) * np.exp(-t * 6)
    return s * np.exp(-t * 1.6)


def kick(strength=1.0):
    n = int(0.5 * SR)
    t = tt(n)
    f = 46 + 100 * np.exp(-t * 28)
    ph = 2 * np.pi * np.cumsum(f) / SR
    s = np.sin(ph) * np.exp(-t * 7.5) + lowpass_ma(rng.standard_normal(n), 8) * np.exp(-t * 380) * 0.1
    return np.tanh(s * 1.6 * strength) * 0.9


def clap():
    n = int(0.3 * SR)
    t = tt(n)
    nz = rng.standard_normal(n)
    nz = nz - lowpass_ma(nz, 12)
    e = np.zeros(n)
    for o in (0, 0.011, 0.022):
        i = int(o * SR)
        e[i:] += np.exp(-(t[: n - i]) * (70 if o < 0.02 else 16))
    return nz * e * 0.5 + np.sin(2 * np.pi * 190 * t) * np.exp(-t * 25) * 0.2


def hat(open_=False):
    n = int((0.25 if open_ else 0.07) * SR)
    t = tt(n)
    nz = np.diff(rng.standard_normal(n + 1))
    return nz * np.exp(-t * (14 if open_ else 75)) * 0.5


def bass(m, dur=0.24):
    n = int(dur * SR)
    f = mtof(m)
    s = saw(f, n, harms=8, bright=0.45) + 0.6 * np.sin(2 * np.pi * f * tt(n))
    return s * env_adsr(n, 0.004, 0.12, 0.7, 0.04)


def noise_sweep(dur, w0, w1, a0, a1):
    n = int(dur * SR)
    nz = rng.standard_normal(n)
    out = np.zeros(n)
    blk = 2048
    for i in range(0, n, blk):
        k = i / n
        w = int(w0 + (w1 - w0) * k)
        seg = nz[max(0, i - 64): i + blk + 64]
        f = lowpass_ma(seg, max(1, w))
        out[i:i + blk] = f[(i - max(0, i - 64)):(i - max(0, i - 64)) + len(out[i:i + blk])]
    amp = np.linspace(a0, a1, n) ** 2
    return out * amp


# ---------------- arrangement ----------------
CH = {
    "Am": ([57, 60, 64], 33), "F": ([53, 57, 60], 29), "C": ([55, 60, 64], 36), "G": ([55, 59, 62], 31),
    "Fmaj7": ([53, 57, 60, 64], 29), "Cadd9": ([48, 55, 60, 62, 64], 36),
}
BARS = ["Am", "F", "Am", "F", "C", "G", "Am", "F", "C", "G", "F", "G", "G", "Fmaj7", "Cadd9"]
kicks = []

for b, name in enumerate(BARS):
    t0 = b * BAR
    notes, root = CH[name]
    # ---- pad ----
    bright = 0.25 if t0 < 4 else 0.32 if t0 < 12 else 0.6 if t0 < 20 else 0.9
    pad_gain = 0.20 if t0 < 12 else 0.085
    for m in notes:
        add(pad_note(m, BAR, bright), t0, pad_gain, pan=(m % 5 - 2) * 0.15, bus="pad", send=0.5)
    if t0 >= 20:  # shimmer an octave up
        for m in notes[:3]:
            add(pad_note(m + 12, BAR, 0.4), t0, 0.03, pan=0.3, bus="pad", send=0.9)

    # ---- intro bells ----
    if t0 < 4:
        add(bell(notes[-1] + 12), t0 + 0.02, 0.20, pan=-0.2, send=0.8)
        add(bell(notes[0] + 12), t0 + 1.0, 0.14, pan=0.25, send=0.8)

    # ---- arpeggios ----
    arp = notes[:3] + [notes[0] + 12]
    pattern = [0, 1, 2, 3, 2, 1, 2, 3]
    if 4 <= t0 < 12:
        for i, p in enumerate(pattern):
            add(pluck(arp[p] + 12, decay=6.0), t0 + i * BEAT / 2, 0.22, pan=-0.25 + 0.5 * (i % 2), send=0.55)
    elif 12 <= t0 < 20:
        for i in range(16):
            m = arp[[0, 1, 2, 3, 2, 3, 1, 2][i % 8]] + 12
            add(pluck(m, decay=10.0, n_sec=0.5), t0 + i * BEAT / 4, 0.085, pan=-0.3 + 0.6 * (i % 2), send=0.35)
            add(pluck(m, decay=10.0, n_sec=0.5), t0 + i * BEAT / 4 + 0.375, 0.03, pan=0.5 - (i % 2), send=0.3)
    elif 20 <= t0 < 26:
        for i in range(16):
            m = arp[[0, 2, 3, 1, 2, 3, 2, 1][i % 8]] + 24
            add(pluck(m, decay=12.0, n_sec=0.45), t0 + i * BEAT / 4, 0.075, pan=-0.35 + 0.7 * (i % 2), send=0.5)
            add(pluck(m, decay=12.0, n_sec=0.45), t0 + i * BEAT / 4 + 0.375, 0.03, pan=0.4 - 0.8 * (i % 2), send=0.5)

    # ---- drums & bass ----
    if 8 <= t0 < 12:  # half-time heartbeat
        for k in (0, 2):
            add(kick(0.7), t0 + k * BEAT, 0.55); kicks.append(t0 + k * BEAT)
    if 12 <= t0 < 25:
        for k in range(4):
            tk = t0 + k * BEAT
            if tk >= 25:
                break
            add(kick(), tk, 0.8); kicks.append(tk)
            add(hat(), tk + BEAT / 2, 0.12, pan=0.3)
            if t0 >= 20:
                add(hat(), tk + BEAT / 4, 0.06, pan=-0.3)
                add(hat(), tk + 3 * BEAT / 4, 0.06, pan=-0.3)
        for k in (1, 3):
            if t0 + k * BEAT < 25:
                add(clap(), t0 + k * BEAT, 0.32, pan=0.05, send=0.25)
        for i in range(8):
            tb = t0 + i * BEAT / 2
            if tb >= 25:
                break
            add(bass(root + (12 if i in (3, 7) else 0)), tb, 0.30)

# snare roll + riser into the impact
for i in range(16):
    tr = 25.0 + i * (1.0 / 16)
    add(lowpass_ma(clap(), 4), tr, 0.05 + 0.12 * i / 15, pan=0.0, send=0.2)
add(noise_sweep(2.0, 60, 10, 0.0, 1.0), 24.0, 0.12, send=0.4)
# whooshes into each chapter
for tw, g in ((4.0, 0.12), (12.0, 0.12), (20.0, 0.2)):
    add(noise_sweep(0.7, 30, 4, 0.0, 1.0), tw - 0.7, g, pan=0.2, send=0.4)
# opening hit: soft boom on frame 1, bell when the question lands
add(kick(0.7), 0.0, 0.55); kicks.append(0.0)
add(bass(33, 1.4), 0.0, 0.18)
add(pluck(57, decay=4.0, n_sec=1.4), 0.0, 0.22, send=0.5)   # 220 Hz body so phone speakers hear the hit
add(pluck(64, decay=4.0, n_sec=1.4), 0.0, 0.14, pan=0.2, send=0.5)
add(bell(76), 0.5, 0.16, pan=0.1, send=0.8)

# impact at 26 s
add(kick(1.3), 26.0, 1.0); kicks.append(26.0)
add(bass(29, 1.6), 26.0, 0.35)
crash_n = int(1.6 * SR)
crash = lowpass_ma(np.diff(rng.standard_normal(crash_n + 1)), 6) * np.exp(-tt(crash_n) * 3.6)
add(crash, 26.0, 0.07, pan=0.1, send=0.3)
# logo chime when the logo pops (27.95 s)
for i, m in enumerate([72, 76, 79, 84]):
    add(bell(m, 2.0), 27.95 + i * 0.07, 0.13, pan=-0.3 + 0.2 * i, send=0.8)
# realtime alert cue (15.5 s), call (16.6 s) and all-clear (17.8 s) to match the manager's phone
for k in range(2):
    add(pluck(88, decay=14.0, n_sec=0.25), 15.5 + k * 0.16, 0.10, pan=-0.4, send=0.3)
add(pluck(81, decay=10.0, n_sec=0.35), 16.6, 0.08, pan=-0.4, send=0.3)
add(pluck(84, decay=8.0, n_sec=0.5), 17.8, 0.07, pan=-0.4, send=0.3)
add(pluck(91, decay=8.0, n_sec=0.5), 17.88, 0.06, pan=-0.4, send=0.3)

# ---------------- mix ----------------
duck = np.ones(N)
for tk in kicks:
    i = int(tk * SR)
    n = min(int(0.35 * SR), N - i)
    duck[i:i + n] = np.minimum(duck[i:i + n], 1 - 0.45 * np.exp(-tt(n) / 0.09))
buses["pad"] *= duck

# reverb: stereo exponentially decaying noise impulse, FFT convolution
ir_n = int(2.4 * SR)
irt = tt(ir_n)
pre = int(0.02 * SR)
wet = np.zeros((2, N))
for ch in range(2):
    ir = rng.standard_normal(ir_n) * np.exp(-irt / 0.5)
    ir[:pre] = 0
    ir = lowpass_ma(ir, 6)
    ir /= np.sqrt(np.sum(ir ** 2))
    L = 1 << int(np.ceil(np.log2(N + ir_n)))
    wet[ch] = np.fft.irfft(np.fft.rfft(buses["send"][ch], L) * np.fft.rfft(ir, L), L)[:N]

mix = buses["main"] + buses["pad"] + wet * 0.45
# section gain: lift the end of 01 a little and trim 02/03 so the 12 s jump stays near 3 dB
gt = tt(N)
sec_gain = np.interp(gt, [0, 10, 11.95, 12.05, 25.9, 26.1, 30], [1, 1, 1.18, 0.86, 0.86, 1, 1])
mix *= sec_gain
mix -= mix.mean(axis=1, keepdims=True)
mix = np.tanh(mix * 1.15) / np.tanh(1.15)
mix /= np.max(np.abs(mix)) / 0.70   # headroom so the AAC-encoded file stays under -1 dBTP
fade = np.ones(N)
fo = int(0.7 * SR)
fade[-fo:] = np.cos(np.linspace(0, np.pi / 2, fo)) ** 2
fade[: int(0.02 * SR)] = np.linspace(0, 1, int(0.02 * SR))
mix *= fade

out = Path(__file__).resolve().parent.parent / "bgm.wav"
pcm = (np.clip(mix.T, -1, 1) * 32767).astype("<i2")
with wave.open(str(out), "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes(pcm.tobytes())

# loudness per section (dBFS RMS) as a sanity check
for a, b, nm in [(0, 4, "intro"), (4, 12, "01 기록"), (12, 20, "02 실시간"), (20, 26, "03 예측"), (26, 30, "ending")]:
    seg = mix[:, int(a * SR):int(b * SR)]
    print(f"{nm:8s} {20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-9):6.1f} dBFS")
print("wrote", out)
