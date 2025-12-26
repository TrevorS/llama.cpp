#!/usr/bin/env python3
"""Analyze audio quality of generated WAV file."""

import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import sys

# Get audio file path from command line or use default
audio_path = sys.argv[1] if len(sys.argv) > 1 else "/models/fix29-test.wav"

# Load the audio file
sr, audio = wav.read(audio_path)
print(f"Sample rate: {sr} Hz")
print(f"Duration: {len(audio)/sr:.2f} seconds")
print(f"Samples: {len(audio)}")

# Convert to float
if audio.dtype == np.int16:
    audio = audio.astype(np.float32) / 32768.0
elif audio.dtype == np.int32:
    audio = audio.astype(np.float32) / 2147483648.0

# Basic stats
print(f"\n=== Basic Stats ===")
print(f"Max amplitude: {np.max(np.abs(audio)):.4f}")
print(f"RMS: {np.sqrt(np.mean(audio**2)):.4f}")

# Compute RMS variation (speech indicator)
frame_size = int(sr * 0.025)  # 25ms frames
hop_size = int(sr * 0.01)     # 10ms hop
n_frames = (len(audio) - frame_size) // hop_size + 1

frame_rms = []
for i in range(n_frames):
    start = i * hop_size
    end = start + frame_size
    frame = audio[start:end]
    rms = np.sqrt(np.mean(frame**2))
    frame_rms.append(rms)

frame_rms = np.array(frame_rms)
rms_mean = np.mean(frame_rms)
rms_std = np.std(frame_rms)
rms_variation = rms_std / rms_mean if rms_mean > 0 else 0

print(f"\n=== RMS Variation Analysis ===")
print(f"Frame RMS mean: {rms_mean:.4f}")
print(f"Frame RMS std: {rms_std:.4f}")
print(f"RMS variation ratio: {rms_variation:.4f}")
print(f"Interpretation: {'Speech-like (>0.3)' if rms_variation > 0.3 else 'Constant/noise (<0.3)'}")

# Pitch detection using autocorrelation
print(f"\n=== Pitch Analysis ===")
# Take a segment from the middle
mid = len(audio) // 2
segment = audio[mid:mid+sr//4]  # 0.25 seconds

# Autocorrelation
correlation = np.correlate(segment, segment, mode="full")
correlation = correlation[len(correlation)//2:]

# Find first peak after minimum pitch period
min_period = sr // 400  # 400 Hz max
max_period = sr // 50   # 50 Hz min

# Find the first significant peak
peaks = []
for i in range(min_period, min(max_period, len(correlation)-1)):
    if correlation[i] > correlation[i-1] and correlation[i] > correlation[i+1]:
        peaks.append((i, correlation[i]))

if peaks:
    peaks.sort(key=lambda x: -x[1])
    best_period = peaks[0][0]
    pitch_hz = sr / best_period
    print(f"Detected pitch: {pitch_hz:.1f} Hz")
    print(f"Interpretation: {'Human voice range (85-255 Hz)' if 85 < pitch_hz < 255 else 'Outside typical voice range'}")
else:
    print("No clear pitch detected")

# Zero crossing rate
zc = np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio))
print(f"\n=== Zero Crossing Rate ===")
print(f"ZCR: {zc:.4f}")
print(f"Interpretation: {'Speech-like (0.02-0.15)' if 0.02 < zc < 0.15 else 'Unusual ZCR'}")

# Spectral analysis
print(f"\n=== Spectral Analysis ===")
f, psd = signal.welch(audio, sr, nperseg=2048)
# Find dominant frequencies
peak_indices = np.argsort(psd)[-5:][::-1]
print("Top 5 frequency components:")
for idx in peak_indices:
    print(f"  {f[idx]:.1f} Hz: {10*np.log10(psd[idx]+1e-10):.1f} dB")

# Check for formants
f1_range = psd[(f > 200) & (f < 800)]
f2_range = psd[(f > 800) & (f < 2500)]
if len(f1_range) > 0 and len(f2_range) > 0:
    f1_energy = np.max(f1_range)
    f2_energy = np.max(f2_range)
    print(f"\nF1 range (200-800 Hz) max energy: {10*np.log10(f1_energy+1e-10):.1f} dB")
    print(f"F2 range (800-2500 Hz) max energy: {10*np.log10(f2_energy+1e-10):.1f} dB")
