"""Convert CHB-MIT scalp EEG annotations into this project's `.tse` format.

CHB-MIT (https://physionet.org/content/chbmit/) ships one folder per patient
(chb01/, chb02/, ...), each containing the raw `.edf` recordings plus a
`chbXX-summary.txt` file that lists, per recording, whether it contains
seizures and their start/end offsets in seconds, e.g.:

    File Name: chb01_03.edf
    File Start Time: 13:43:04
    File End Time: 14:43:04
    Number of Seizures in File: 1
    Seizure Start Time: 2996 seconds
    Seizure End Time: 3036 seconds

(Some later patients, e.g. chb24, use a numbered variant such as
`Seizure 1 Start Time: ... seconds` when a file has more than one seizure —
both forms are handled here.)

This script parses every `*-summary.txt` under `--data_dir` and writes one
`.tse` file next to each corresponding `.edf`, with the same record_id
(basename without extension), so the existing `pair_edf_tse` / `tse_parser`
/ `dataset.py` pipeline picks them up completely unmodified. Nothing is
copied or deleted; only new `.tse` sidecar files are added.

Usage:
    python -m src.chbmit_prepare --data_dir data/chb-mit-scalp-eeg-database-1.0.0

Each generated `.tse` line follows the project's convention:
    <start_sec> <end_sec> <label> <confidence>
with label `bckg` for background and `seiz` for seizure activity.
"""

import argparse
import os
import re
from typing import Dict, List, Tuple

import pyedflib

FILE_NAME_RE = re.compile(r"^File Name:\s*(\S+)", re.IGNORECASE)
NUM_SEIZ_RE = re.compile(r"^Number of Seizures in File:\s*(\d+)", re.IGNORECASE)
SEIZ_START_RE = re.compile(r"^Seizure(?:\s+\d+)?\s*Start Time:\s*(\d+)\s*seconds?", re.IGNORECASE)
SEIZ_END_RE = re.compile(r"^Seizure(?:\s+\d+)?\s*End Time:\s*(\d+)\s*seconds?", re.IGNORECASE)


def parse_summary(summary_path: str) -> Dict[str, List[Tuple[float, float]]]:
	"""Parse one chbXX-summary.txt into {edf_filename: [(seiz_start, seiz_end), ...]}.

	Files with 'Number of Seizures in File: 0' get an empty list (still
	included, so a pure-background .tse is produced for them).
	"""
	result: Dict[str, List[Tuple[float, float]]] = {}
	current_file = None
	pending_start = None
	with open(summary_path, "r", encoding="utf-8", errors="ignore") as f:
		for raw_line in f:
			line = raw_line.strip()
			if not line:
				continue
			m = FILE_NAME_RE.match(line)
			if m:
				current_file = m.group(1)
				result.setdefault(current_file, [])
				pending_start = None
				continue
			if NUM_SEIZ_RE.match(line):
				# nothing to do, just informational; presence of seizure lines below governs parsing
				continue
			m = SEIZ_START_RE.match(line)
			if m and current_file is not None:
				pending_start = float(m.group(1))
				continue
			m = SEIZ_END_RE.match(line)
			if m and current_file is not None and pending_start is not None:
				result[current_file].append((pending_start, float(m.group(1))))
				pending_start = None
				continue
	return result


def edf_duration_sec(edf_path: str) -> float:
	with pyedflib.EdfReader(edf_path) as r:
		n = r.getNSamples()[0]
		fs = r.getSampleFrequency(0)
	return float(n) / float(fs)


def write_tse(tse_path: str, duration: float, seizures: List[Tuple[float, float]]) -> None:
	"""Write a .tse covering [0, duration) with seiz spans carved out of bckg."""
	seizures = sorted(seizures)
	lines = []
	cursor = 0.0
	for s, e in seizures:
		s = max(0.0, min(s, duration))
		e = max(0.0, min(e, duration))
		if e <= s:
			continue
		if s > cursor:
			lines.append((cursor, s, "bckg"))
		lines.append((s, e, "seiz"))
		cursor = e
	if cursor < duration:
		lines.append((cursor, duration, "bckg"))
	if not lines:
		lines = [(0.0, duration, "bckg")]
	with open(tse_path, "w", encoding="utf-8") as f:
		for s, e, lab in lines:
			f.write(f"{s:.4f} {e:.4f} {lab} 1.0000\n")


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--data_dir", type=str, required=True, help="Root of the CHB-MIT dataset (contains chb01/, chb02/, ...)")
	ap.add_argument("--overwrite", action="store_true", help="Overwrite existing .tse files")
	args = ap.parse_args()

	n_written = 0
	n_skipped = 0
	n_missing_edf = 0
	for root, _dirs, files in os.walk(args.data_dir):
		summary_files = [f for f in files if f.lower().endswith("-summary.txt")]
		for sf in summary_files:
			summary_path = os.path.join(root, sf)
			parsed = parse_summary(summary_path)
			for edf_name, seizures in parsed.items():
				edf_path = os.path.join(root, edf_name)
				if not os.path.exists(edf_path):
					print(f"[WARN] listed in {sf} but .edf not found on disk: {edf_path}")
					n_missing_edf += 1
					continue
				base, _ext = os.path.splitext(edf_name)
				tse_path = os.path.join(root, base + ".tse")
				if os.path.exists(tse_path) and not args.overwrite:
					n_skipped += 1
					continue
				duration = edf_duration_sec(edf_path)
				write_tse(tse_path, duration, seizures)
				n_written += 1
	print(f"done. wrote {n_written} .tse files, skipped {n_skipped} existing, {n_missing_edf} referenced .edf files missing.")


if __name__ == "__main__":
	raise SystemExit(main())
