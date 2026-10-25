"""Model fitting: learned pair weights (train companies), calibration and thresholds (calibration companies).

This package reads ground truth through orgchart.evaluation.labels and is NOT
part of the inference runtime. Its only output is model JSON files that
inference reads. Test companies are never used here.
"""
