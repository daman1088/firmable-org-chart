"""Parquet + DuckDB storage.

`orgchart.storage.observed` exposes observed data only and is safe for inference.
`orgchart.storage.truth` exposes ground truth and may only be imported by
evaluation code; tests/test_isolation.py enforces this.
"""
