#!/usr/bin/env python3
"""Rebuild a SwanLab datastore while skipping corrupted framing regions.

The input is never modified. The output is a new datastore containing every
logical protobuf record that can be recovered with a valid LevelDB-style CRC.
"""

from __future__ import annotations

import argparse
import json
import struct
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Optional

from google.protobuf.message import DecodeError

from swanlab.proto.swanlab.record.v1.record_pb2 import Record
from swanlab.sdk.internal.core_python.store import (
    DataStoreReader,
    DataStoreWriter,
    LEVELDBLOG_BLOCK_LEN,
    LEVELDBLOG_DATA_LEN,
    LEVELDBLOG_FIRST,
    LEVELDBLOG_FULL,
    LEVELDBLOG_HEADER_IDENT,
    LEVELDBLOG_HEADER_LEN,
    LEVELDBLOG_HEADER_MAGIC,
    LEVELDBLOG_HEADER_VERSION,
    LEVELDBLOG_LAST,
    LEVELDBLOG_MIDDLE,
    _CRC,
)


@dataclass
class RepairReport:
    input_file: str
    output_file: str
    input_bytes: int
    output_bytes: int = 0
    recovered_records: int = 0
    recovered_scalars: int = 0
    maximum_scalar_step: int = -1
    start_records: int = 0
    finish_records: int = 0
    skipped_regions: int = 0
    skipped_bytes: int = 0
    incomplete_tail_bytes: int = 0


def _parse_record(payload: bytes) -> Optional[Record]:
    record = Record()
    try:
        record.ParseFromString(payload)
    except DecodeError:
        return None
    if record.WhichOneof("record_type") is None:
        return None
    return record


def _physical_record(data: bytes, position: int) -> Optional[tuple[int, bytes, int]]:
    """Return (type, payload, end) for one CRC-valid physical record."""
    if position + LEVELDBLOG_HEADER_LEN > len(data):
        return None
    block_offset = position % LEVELDBLOG_BLOCK_LEN
    if LEVELDBLOG_BLOCK_LEN - block_offset < LEVELDBLOG_HEADER_LEN:
        return None
    checksum, length, record_type = struct.unpack_from("<IHB", data, position)
    if record_type not in (LEVELDBLOG_FULL, LEVELDBLOG_FIRST, LEVELDBLOG_MIDDLE, LEVELDBLOG_LAST):
        return None
    if length > LEVELDBLOG_DATA_LEN:
        return None
    end = position + LEVELDBLOG_HEADER_LEN + length
    if end > len(data) or end - position > LEVELDBLOG_BLOCK_LEN - block_offset:
        return None
    payload = data[position + LEVELDBLOG_HEADER_LEN : end]
    if zlib.crc32(payload, _CRC[record_type]) & 0xFFFFFFFF != checksum:
        return None
    return record_type, payload, end


def _next_full_record(data: bytes, start: int) -> Optional[int]:
    """Find a high-confidence resynchronization point after corrupt bytes."""
    last = len(data) - LEVELDBLOG_HEADER_LEN
    for position in range(start, last + 1):
        physical = _physical_record(data, position)
        if physical is None:
            continue
        record_type, payload, _ = physical
        if record_type == LEVELDBLOG_FULL and _parse_record(payload) is not None:
            return position
    return None


def _recover_payloads(data: bytes, report: RepairReport) -> Iterator[bytes]:
    expected_header = struct.pack(
        "<4sHB", LEVELDBLOG_HEADER_IDENT, LEVELDBLOG_HEADER_MAGIC, LEVELDBLOG_HEADER_VERSION
    )
    if data[:LEVELDBLOG_HEADER_LEN] != expected_header:
        raise ValueError("input does not have a compatible SwanLab datastore header")

    position = LEVELDBLOG_HEADER_LEN
    fragments: list[bytes] = []
    while position < len(data):
        block_remaining = LEVELDBLOG_BLOCK_LEN - position % LEVELDBLOG_BLOCK_LEN
        if block_remaining < LEVELDBLOG_HEADER_LEN:
            padding_end = min(position + block_remaining, len(data))
            if data[position:padding_end] == b"\x00" * (padding_end - position):
                position = padding_end
                continue

        physical = _physical_record(data, position)
        if physical is None:
            recovery = _next_full_record(data, position + 1)
            fragments.clear()
            if recovery is None:
                report.incomplete_tail_bytes = len(data) - position
                break
            report.skipped_regions += 1
            report.skipped_bytes += recovery - position
            position = recovery
            continue

        record_type, payload, end = physical
        position = end
        logical: Optional[bytes] = None
        if record_type == LEVELDBLOG_FULL:
            fragments.clear()
            logical = payload
        elif record_type == LEVELDBLOG_FIRST:
            fragments = [payload]
        elif record_type == LEVELDBLOG_MIDDLE:
            if fragments:
                fragments.append(payload)
            else:
                recovery = _next_full_record(data, position)
                if recovery is None:
                    report.incomplete_tail_bytes = len(data) - position
                    break
                report.skipped_regions += 1
                report.skipped_bytes += recovery - (end - LEVELDBLOG_HEADER_LEN - len(payload))
                position = recovery
        elif record_type == LEVELDBLOG_LAST:
            if fragments:
                fragments.append(payload)
                logical = b"".join(fragments)
                fragments.clear()
            else:
                recovery = _next_full_record(data, position)
                if recovery is None:
                    report.incomplete_tail_bytes = len(data) - position
                    break
                report.skipped_regions += 1
                report.skipped_bytes += recovery - (end - LEVELDBLOG_HEADER_LEN - len(payload))
                position = recovery

        if logical is None:
            continue
        record = _parse_record(logical)
        if record is None:
            recovery = _next_full_record(data, position)
            if recovery is None:
                report.incomplete_tail_bytes = len(data) - position
                break
            report.skipped_regions += 1
            report.skipped_bytes += recovery - (end - LEVELDBLOG_HEADER_LEN - len(payload))
            position = recovery
            continue
        yield logical


def repair(input_file: Path, output_file: Path) -> RepairReport:
    if output_file.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_file}")
    data = input_file.read_bytes()
    report = RepairReport(str(input_file), str(output_file), len(data))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    writer = DataStoreWriter()
    writer.open(output_file)
    try:
        for payload in _recover_payloads(data, report):
            record = _parse_record(payload)
            assert record is not None
            writer.write(payload)
            report.recovered_records += 1
            kind = record.WhichOneof("record_type")
            if kind == "scalar":
                report.recovered_scalars += 1
                report.maximum_scalar_step = max(report.maximum_scalar_step, record.scalar.step)
            elif kind == "start":
                report.start_records += 1
            elif kind == "finish":
                report.finish_records += 1
    finally:
        writer.close()
    report.output_bytes = output_file.stat().st_size
    return report


def validate(output_file: Path, expected: RepairReport) -> None:
    reader = DataStoreReader()
    reader.open(output_file)
    count = scalars = starts = finishes = 0
    max_step = -1
    try:
        while True:
            payload = reader.scan()
            if payload is None:
                break
            record = _parse_record(payload)
            if record is None:
                raise ValueError(f"invalid protobuf record at logical record {count + 1}")
            count += 1
            kind = record.WhichOneof("record_type")
            if kind == "scalar":
                scalars += 1
                max_step = max(max_step, record.scalar.step)
            elif kind == "start":
                starts += 1
            elif kind == "finish":
                finishes += 1
    finally:
        reader.close()
    actual = (count, scalars, max_step, starts, finishes)
    wanted = (
        expected.recovered_records,
        expected.recovered_scalars,
        expected.maximum_scalar_step,
        expected.start_records,
        expected.finish_records,
    )
    if actual != wanted:
        raise ValueError(f"validation mismatch: actual={actual}, expected={wanted}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=Path)
    parser.add_argument("output_file", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = repair(args.input_file, args.output_file)
    validate(args.output_file, report)
    rendered = json.dumps(asdict(report), indent=2, ensure_ascii=False)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
