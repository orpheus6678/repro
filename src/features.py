# Source Generated with Decompyle++
# File: features.cpython-311.pyc (Python 3.11)

"""Spectral bandpower and simple statistics feature extraction.

Default bands: delta/theta/alpha/beta/gamma. For each time window, compute Welch
PSD per channel, integrate band powers and aggregate {mean, std} across channels
plus broadband energy and time-domain RMS statistics to form frame features.
"""
from typing import Tuple
import numpy as np
from scipy.signal import welch

EEG_BANDS = {
	'delta': (0.5, 4.0),
	'theta': (4.0, 8.0),
	'alpha': (8.0, 13.0),
	'beta': (13.0, 30.0),
	'gamma': (30.0, 45.0),
}


def _bandpower_from_psd(freqs: np.ndarray, psd: np.ndarray, band: Tuple[float, float]) -> np.ndarray:
	idx = np.logical_and(freqs >= band[0], freqs < band[1])
	if not np.any(idx):
		return np.zeros(psd.shape[:-1], dtype=psd.dtype)
	return np.trapezoid(psd[..., idx], freqs[idx], axis=-1)


def extract_features_multichannel(
	signals: np.ndarray,
	fs: float,
	window_sec: float,
	hop_sec: float,
) -> Tuple[np.ndarray, np.ndarray]:
	"""Extract per-window features aggregated across channels.

	Returns (features [T,F], frame_centers_sec [T])
	Base features: for each band {mean, std} across channels + broadband {mean, std}
	Extra features: simple time-domain RMS {mean, std} across channels
	"""
	if signals.ndim != 2:
		raise ValueError("signals must be 2D [C, N]")
	n_channels, n_samples = signals.shape
	win = int(round(window_sec * fs))
	hop = int(round(hop_sec * fs))
	if win <= 0 or hop <= 0 or win > n_samples:
		raise ValueError("Invalid window/hop settings")
	starts = np.arange(0, n_samples - win + 1, hop, dtype=int)
	centers = (starts + win // 2) / float(fs)

	# 🔥 OOM fix: pre-allocate the feature array to avoid dynamic growth
	n_frames = len(starts)
	n_features = len(EEG_BANDS) * 2 + 2 + 2  # 5 bands*2 + broadband*2 + RMS*2 = 14
	X = np.zeros((n_frames, n_features), dtype=np.float32)
	
	# 🔥 Memory optimization: batch processing instead of frame-by-frame
	nperseg = min(win, 256)
	for i, s in enumerate(starts.tolist()):
		seg = signals[:, s:s + win]  # [C, win]
		
		# 🔥 Memory optimization: use a pre-allocated array
		psd_array = np.zeros((n_channels, nperseg // 2 + 1), dtype=np.float32)
		
		# PSD per channel - memory-optimized version
		for c in range(n_channels):
			f, p = welch(seg[c], fs=fs, nperseg=nperseg, noverlap=nperseg // 2, scaling='density')
			psd_array[c] = p.astype(np.float32)  # Ensure consistent dtype
		
		# Fill the feature array directly
		feat_idx = 0
		
		# Band power mean/std across channels
		for band in EEG_BANDS.values():
			bp = _bandpower_from_psd(f, psd_array, band)  # [C]
			X[i, feat_idx] = np.mean(bp)
			X[i, feat_idx + 1] = np.std(bp)
			feat_idx += 2
		
		# Broadband power
		broad = (0.5, 45.0)
		bp_broad = _bandpower_from_psd(f, psd_array, broad)
		X[i, feat_idx] = np.mean(bp_broad)
		X[i, feat_idx + 1] = np.std(bp_broad)
		feat_idx += 2
		
		# Time-domain RMS mean/std
		rms = np.sqrt(np.mean(seg.astype(np.float32) ** 2, axis=1))  # [C]
		X[i, feat_idx] = np.mean(rms)
		X[i, feat_idx + 1] = np.std(rms)
		
		# 🔥 Memory optimization: free temporary arrays promptly
		del psd_array, seg, rms

	return X, centers

