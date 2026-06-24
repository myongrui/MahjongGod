"""
Procedural sound effects for the 3D game — short wooden "clack" tones synthesised with
numpy and written to temporary .wav files, so there are no binary assets to ship.

Three sounds: a soft tick when a tile is drawn, a clack when one is thrown to the table,
and a sharper knock when sliding tiles collide. Each has a few pitch-varied variants that
round-robin so repeats don't sound identical.

    import cracked.ursina_audio as ua
    ua.init()          # after Ursina() is created
    ua.play("throw")   # anywhere
"""
from __future__ import annotations

import random
import tempfile
import wave
from pathlib import Path

import numpy as np

_SR = 44100
_bank: dict[str, list] = {}     # name -> [Audio, ...] variants
_rr: dict[str, int] = {}        # name -> round-robin index

# freqs = [(hz, amplitude), ...]; dur seconds; decay 1/s; noise 0..1; vol playback gain
_SPECS = {
    "draw":    dict(freqs=[(820, 1.0), (1400, 0.4)], dur=0.07, decay=70, noise=0.25, vol=0.30),
    "throw":   dict(freqs=[(480, 1.0), (760, 0.6), (1150, 0.3)], dur=0.13, decay=42, noise=0.5, vol=0.55),
    "collide": dict(freqs=[(1150, 1.0), (1850, 0.7), (2500, 0.4)], dur=0.06, decay=95, noise=0.6, vol=0.40),
    "meld":    dict(freqs=[(380, 1.0), (620, 0.6), (950, 0.3)], dur=0.15, decay=36, noise=0.35, vol=0.60),
}

# a short ascending pentatonic chime for a win (C5–E5–G5–C6)
_WIN_NOTES = [523, 659, 784, 1047]
_WIN_VOL = 0.6


def _clack(freqs, dur, decay, noise, seed) -> np.ndarray:
    """A damped percussive blip: summed partials + a noise transient under an exponential
    decay envelope, normalised to int16."""
    rng = np.random.default_rng(seed)
    n = int(_SR * dur)
    t = np.arange(n) / _SR
    sig = np.zeros(n)
    for f, a in freqs:
        sig += a * np.sin(2 * np.pi * f * t)
    sig += noise * rng.uniform(-1.0, 1.0, n)
    sig *= np.exp(-t * decay)
    atk = max(1, int(_SR * 0.002))                      # 2ms fade-in so the start doesn't pop
    sig[:atk] *= np.linspace(0, 1, atk)
    sig /= np.max(np.abs(sig)) + 1e-9
    return (sig * 0.95 * 32767).astype("<i2")


def _jingle(notes, note_dur=0.2, step=0.09, decay=10) -> np.ndarray:
    """An ascending bell-like arpeggio: each note is a few harmonics under a slow decay,
    placed `step` seconds apart so they ring and overlap. Used for the win sound."""
    total = int(_SR * (step * (len(notes) - 1) + note_dur))
    out = np.zeros(total)
    n = int(_SR * note_dur)
    t = np.arange(n) / _SR
    for i, f in enumerate(notes):
        note = np.sin(2 * np.pi * f * t) + 0.5 * np.sin(2 * np.pi * 2 * f * t) + 0.25 * np.sin(2 * np.pi * 3 * f * t)
        note *= np.exp(-t * decay)
        start = int(i * step * _SR)
        out[start:start + n] += note[:max(0, total - start)]
    atk = max(1, int(_SR * 0.002))
    out[:atk] *= np.linspace(0, 1, atk)
    out /= np.max(np.abs(out)) + 1e-9
    return (out * 0.95 * 32767).astype("<i2")


def _write_wav(path: Path, samples: np.ndarray):
    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(_SR)
        w.writeframes(samples.tobytes())


def init():
    """Build the sound bank. Safe to call once after Ursina() exists; a no-op if audio or
    ursina is unavailable, so the game still runs silently rather than crashing."""
    if _bank:
        return
    try:
        from ursina import Audio
    except Exception:
        return
    out = Path(tempfile.mkdtemp(prefix="cracked_sfx_"))

    def _register(name, samples_for_variant, n_variants, vol):
        clips = []
        for v in range(n_variants):
            p = out / f"{name}_{v}.wav"
            _write_wav(p, samples_for_variant(v))
            try:
                clips.append(Audio(Path(p), volume=vol, autoplay=False, auto_destroy=False))
            except Exception:
                pass
        if clips:
            _bank[name] = clips
            _rr[name] = 0

    for name, spec in _SPECS.items():
        _register(name, lambda v, s=spec: _clack(s["freqs"], s["dur"], s["decay"], s["noise"], seed=v),
                  3, spec["vol"])
    _register("win", lambda v: _jingle(_WIN_NOTES), 1, _WIN_VOL)


def play(name: str, pitch_jitter: float = 0.12):
    """Play a sound by name with a little random pitch variation; no-op if uninitialised."""
    clips = _bank.get(name)
    if not clips:
        return
    i = _rr[name]
    _rr[name] = (i + 1) % len(clips)
    a = clips[i]
    try:
        a.pitch = 1.0 + random.uniform(-pitch_jitter, pitch_jitter)
        a.play()
    except Exception:
        pass
