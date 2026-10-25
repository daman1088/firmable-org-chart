"""Schema-checked Parquet I/O."""

import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def write_table(table: pa.Table, path: Path, schema: pa.Schema) -> None:
    if not table.schema.equals(schema, check_metadata=False):
        raise ValueError(f"schema mismatch writing {path}:\n{table.schema}\n!=\n{schema}")
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


def read_table(path: Path, schema: pa.Schema) -> pa.Table:
    table = pq.read_table(path)
    if not table.schema.equals(schema, check_metadata=False):
        raise ValueError(f"schema mismatch reading {path}:\n{table.schema}\n!=\n{schema}")
    return table


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
