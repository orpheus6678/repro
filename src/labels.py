from typing import Dict, List, Tuple, Optional
import re


def _read_excel_pairs(path: str) -> List[Tuple[str, Optional[str]]]:
	"""Read a list of key-value pairs from an Excel file (best-effort, fault-tolerant).

	Rules:
	- Uses the first worksheet;
	- For each row, the first two non-empty string cells are taken as (key, value); if only one exists, (key, None);
	"""
	import openpyxl  # Imported only when used, to avoid a compile-time dependency
	wb = openpyxl.load_workbook(path, data_only=True)
	ws = wb.worksheets[0]
	pairs: List[Tuple[str, Optional[str]]] = []
	rec_id_pattern = re.compile(r"^\d{8}_s\d{3}_t\d{3}$", re.IGNORECASE)
	# Note: the Chinese terms below (规范名/别名/类型/类别 etc.) are Chinese-language header
	# keywords ("canonical name", "alias", "type", "category") matched against real Excel
	# header cells, so they are left as-is rather than translated.
	header_pattern = re.compile(r"^(class( code)?|label|alias|规范名|别名|类型|类别|name|code)$", re.IGNORECASE)
	noise_word_pattern = re.compile(r"(file(name)?|record|patient|start|end|onset|offset|duration|confidence|备注|说明|sheet|table|index|序号|编号|id)$", re.IGNORECASE)
	for row in ws.iter_rows(values_only=True):
		cells = [c for c in row if c is not None]
		cells = [str(c).strip() for c in cells if str(c).strip()]
		if not cells:
			continue
		k0 = cells[0]
		v0 = cells[1] if len(cells) > 1 else None
		# Skip header rows / noise keywords
		if header_pattern.match(k0) or (v0 is not None and header_pattern.match(v0)):
			continue
		if noise_word_pattern.search(k0):
			continue
		# Filter out rows that look like a record ID, or a plain number/timestamp
		if rec_id_pattern.match(k0):
			continue
		if k0.replace(".", "", 1).isdigit():
			continue
		# Filter out cases where the alias column is a record ID / plain number
		if v0 is not None:
			if rec_id_pattern.match(v0):
				v0 = None
			elif v0.replace(".", "", 1).isdigit():
				v0 = None
			elif noise_word_pattern.search(v0):
				v0 = None
		if v0 is None:
			pairs.append((k0, None))
		else:
			pairs.append((k0, v0))
	return pairs


def build_labels_from_excels(
	types_xlsx: Optional[str],
	periods_xlsx: Optional[str],
	background: str = "bckg",
) -> Tuple[List[str], Dict[str, str]]:
	"""Build the label list and alias mapping from Excel tables.

	Returns:
	- label_names: an ordered label list (with background placed first)
	- aliases: a dict mapping alias -> canonical name (case-insensitive, normalized to lowercase internally)
	"""
	canon: Dict[str, None] = {}
	alias_map: Dict[str, str] = {}

	# The class set is defined only from the "seizure type table"
	if types_xlsx:
		for k, v in _read_excel_pairs(types_xlsx):
			k_norm = k.strip()
			if not k_norm:
				continue
			canon.setdefault(k_norm, None)
			# The name's own alias (lowercased)
			alias_map.setdefault(k_norm.lower(), k_norm)
			# The second column is treated as an alias mapping to the canonical name
			if v is not None and v.strip() and v.strip().lower() != k_norm.lower():
				alias_map[v.strip().lower()] = k_norm

	# The "period type table" is only used to add extra alias mappings; it does not add new classes
	if periods_xlsx and canon:
		canon_l2c = {c.lower(): c for c in canon.keys()}
		for k, v in _read_excel_pairs(periods_xlsx):
			k_s = k.strip() if k else ""
			v_s = v.strip() if (v is not None) else ""
			# If the second column (alias -> canonical name) is already in the class set, add the alias mapping
			if v_s and v_s.lower() in canon_l2c:
				alias_map[k_s.lower()] = canon_l2c[v_s.lower()]
			# Or if the first column is itself a canonical name, accept the second column as its alias
			elif k_s and k_s.lower() in canon_l2c and v_s:
				alias_map[v_s.lower()] = canon_l2c[k_s.lower()]

	# Make sure background is placed first
	labels = [background]
	for name in canon.keys():
		if name.lower() == background.lower():
			continue
		labels.append(name)
	return labels, alias_map


def write_labels_json(out_path: str, label_names: List[str], aliases: Dict[str, str], background: str) -> None:
	import json, os
	os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
	with open(out_path, "w", encoding="utf-8") as f:
		json.dump({
			"background": background,
			"label_names": label_names,
			"aliases": aliases,
		}, f, ensure_ascii=False, indent=2)


