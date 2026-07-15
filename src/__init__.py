# Source Generated with Decompyle++
# File: __init__.cpython-311.pyc (Python 3.11)

'''Source package for the EEG seizure prediction project.

This package contains:
- Data reading and preprocessing: `edf_reader`, `tse_parser`, `features`, `dataset`
- Training and inference: `train`, `predict`, `model`
- Experiment utilities: `scan_thresholds`, `compare`, `utils`

Conventions:
- Training and inference parameters are configured uniformly via `configs/config.yaml`;
- Frame-level classification uses `"bckg"` as the background class name;
- All I/O and intermediate results are kept reproducible where possible (e.g. `data_cache/`, `outputs*/`).
'''
__all__ = [
    'compare',
    'dataset',
    'edf_reader',
    'features',
    'model',
    'predict',
    'scan_thresholds',
    'train',
    'tse_parser',
    'utils']
