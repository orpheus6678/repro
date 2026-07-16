"""Utility helpers: pairing files, JSON I/O, patient splits, and label tools."""

import os
import json
import re
from typing import List, Tuple, Dict, Optional

import numpy as np


def pair_edf_tse(data_dir: str) -> List[Tuple[str, Optional[str], str]]:
	"""Scan a directory (including subdirectories) and pair up .edf and .tse
	files by record_id. Returns (edf_path, tse_path_or_None, record_id).

	Supports two layouts:
	- Flat directory (TUSZ style): all .edf/.tse files live directly under data_dir;
	- Per-patient subdirectories (CHB-MIT style): data_dir/chb01/chb01_03.edf etc.
	  Both layouts are scanned recursively, and record_id is taken from the
	  filename (without extension), so all .edf filenames used in a given
	  training run must be unique (CHB-MIT's chbXX_YY naming naturally
	  satisfies this).
	"""
	edfs: Dict[str, str] = {}
	tses: Dict[str, str] = {}
	for root, _dirs, files in os.walk(data_dir, followlinks=True):
		for name in files:
			base, ext = os.path.splitext(name)
			path = os.path.join(root, name)
			ext_l = ext.lower()
			if ext_l == ".edf":
				if base in edfs:
					raise ValueError(
						f"duplicate record_id '{base}': {edfs[base]} and {path} share the "
						"same basename; make sure all .edf filenames under data_dir are unique."
					)
				edfs[base] = path
			elif ext_l == ".tse":
				tses[base] = path
	pairs: List[Tuple[str, Optional[str], str]] = []
	for base, edf_path in edfs.items():
		pairs.append((edf_path, tses.get(base), base))
	return pairs


def scan_label_set(tse_paths: List[str], background_label: str) -> List[str]:
	labels = set()
	for p in tse_paths:
		if not p or not os.path.exists(p):
			continue
		with open(p, "r", encoding="utf-8", errors="ignore") as f:
			for line in f:
				parts = line.strip().split()
				if len(parts) >= 3:
					lab = parts[2]
					# skip version/header tokens
					if lab.lower().startswith("tse_v"):
						continue
					labels.add(lab)
	labels.discard(background_label)
	return [background_label] + sorted(labels)


def ensure_dir(path: str) -> None:
	os.makedirs(path, exist_ok=True)


def save_json(obj, path: str) -> None:
	ensure_dir(os.path.dirname(path) or ".")
	with open(path, "w", encoding="utf-8") as f:
		json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str):
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def parse_patient_id(record_id: str) -> str:
	"""Extract the patient ID from a record_id, used for patient-level splits
	and LOSOCV grouping.

	Supports:
	- TUSZ style: '00000001_s001_t000' -> '00000001'
	- CHB-MIT style: 'chb01_03' -> 'chb01' (case-insensitive; tolerates a small
	  number of variant suffixes like 'chb01a_03'. Note e.g. chb21 is a
	  companion record to chb01 but lives under a different directory name,
	  so it is intentionally NOT merged into the same patient group.)
	"""
	m = re.match(r"^(\d{8})_", record_id)
	if m:
		return m.group(1)
	m = re.match(r"^(chb\d+)", record_id, re.IGNORECASE)
	if m:
		return m.group(1).lower()
	return record_id


def split_records_by_patient(
	pairs: List[Tuple[str, Optional[str], str]],
	val_ratio: float,
	test_ratio: float,
	seed: int = 42,
) -> Dict[str, List[Tuple[str, Optional[str], str]]]:
	labeled = [p for p in pairs if p[1]]
	rng = np.random.RandomState(seed)
	patients: Dict[str, List[Tuple[str, Optional[str], str]]] = {}
	for _, _, rec in labeled:
		pid = parse_patient_id(rec)
		patients.setdefault(pid, [])
	for item in labeled:
		pid = parse_patient_id(item[2])
		patients[pid].append(item)
	pids = list(patients.keys())
	rng.shuffle(pids)
	n = len(pids)
	n_test = int(round(n * test_ratio))
	n_val = int(round(n * val_ratio))
	val_ids = set(pids[:n_val])
	test_ids = set(pids[n_val:n_val + n_test])
	train_ids = set(pids[n_val + n_test:])
	def collect(idset):
		res: List[Tuple[str, Optional[str], str]] = []
		for pid in idset:
			res.extend(patients[pid])
		return res
	return {
		"train": collect(train_ids),
		"val": collect(val_ids),
		"test": collect(test_ids),
	}


def merge_non_background_segments(
	frame_centers_sec: np.ndarray,
	probs: np.ndarray,
	label_names: List[str],
	background_label: str,
	prob_threshold: float = 0.5,
	min_duration_sec: float = 0.0,
) -> List[Dict]:
	"""Generate a list of segments from per-frame probabilities."""
	bg_idx = label_names.index(background_label)
	fg_prob = 1.0 - probs[:, bg_idx]
	is_fg = fg_prob >= prob_threshold
	segments: List[Dict] = []
	start = None
	for i, flag in enumerate(is_fg):
		if flag and start is None:
			start = i
		elif (not flag) and start is not None:
			end = i - 1
			dur = frame_centers_sec[end] - frame_centers_sec[start]
			if dur >= min_duration_sec:
				# majority class
				sum_probs = probs[start:end + 1].sum(axis=0)
				cls = int(np.argmax(sum_probs))
				segments.append({
					"start": float(frame_centers_sec[start]),
					"end": float(frame_centers_sec[end]),
					"label": label_names[cls],
					"score": float(fg_prob[start:end + 1].max()),
				})
			start = None
	if start is not None:
		end = len(frame_centers_sec) - 1
		dur = frame_centers_sec[end] - frame_centers_sec[start]
		if dur >= min_duration_sec:
			sum_probs = probs[start:end + 1].sum(axis=0)
			cls = int(np.argmax(sum_probs))
			segments.append({
				"start": float(frame_centers_sec[start]),
				"end": float(frame_centers_sec[end]),
				"label": label_names[cls],
				"score": float(fg_prob[start:end + 1].max()),
			})
	return segments


def normalize_label_with_alias(label: str, aliases: Optional[Dict[str, str]]) -> str:
	"""Normalize a label name using an alias table (case-insensitive). Returns
	the label unchanged if no alias is found."""
	if not aliases:
		return label
	return aliases.get(label.lower(), label)