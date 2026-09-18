"""Build id,indices,key,state from FINAL PUMP indices and root-level JSON events.

Indices are zero-based positions in the unfiltered CSV, excluding the header.
They describe slices [start:end]. Nothing is resampled or removed.
Keep generate_pump_combined_meta.py beside this script. Run --help for options.
"""

import argparse
import json
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from pathlib import Path

import numpy as np
import pandas as pd

from generate_pump_combined_meta import (
    BOUNDARY_COUNTS, KEY_COLUMNS, META_COLUMNS, SIGNAL_FILES, STATE_SIGNALS,
    basename, check_new_sample, load_final_regions, read_state_events,
    read_table, save_table, validate_meta,
)


UNITS = {"s": Decimal("1"), "ms": Decimal("0.001"), "us": Decimal("0.000001"),
         "ns": Decimal("0.000000001"), "ticks": Decimal("0.0000001")}
TIME_COLUMNS = ("original_timestamp", "Timestamp", "timestamp", "firmware_timestamp", "F_Timestamp")
OUTPUT_COLUMNS = ["id", "indices", "key", "state"]


def parse_events(values, description):
    try:
        result = [Decimal(str(v).strip()) for v in values]
    except (InvalidOperation, ValueError) as error:
        raise ValueError("Invalid timestamp in {}".format(description)) from error
    if len(result) < 2 or any(not x.is_finite() for x in result):
        raise ValueError("Expected finite timestamps in {}".format(description))
    if any(b <= a for a, b in zip(result, result[1:])):
        raise ValueError("JSON timestamps must strictly increase: {}".format(description))
    return result


def select_positions(events, state, explicit, config):
    count = BOUNDARY_COUNTS[state]
    expected_events = config.get(state, {}).get("no_changepoints", count + 1)
    if explicit is None:
        if len(events) != count + 1 or expected_events != count + 1:
            raise ValueError(
                "{} has {} JSON entries; FINAL {} requires {} boundary indices. "
                "Config no_changepoints={}. Supply --{}-json-positions with exactly {} "
                "event positions (zero-based), selected according to the collection procedure. "
                "No events were chosen automatically.".format(
                    state, len(events), state, count, expected_events, state, count))
        return list(range(1, count + 1))
    try:
        positions = [int(x.strip()) for x in explicit.split(",")]
    except ValueError as error:
        raise ValueError("Use comma-separated integer event positions.") from error
    if (len(positions) != count or any(i < 0 or i >= len(events) for i in positions)
            or any(b <= a for a, b in zip(positions, positions[1:]))):
        raise ValueError("--{}-json-positions needs {} increasing positions from 0 through {}.".format(
            state, count, len(events) - 1))
    return positions


def timestamp_column(path, requested=None):
    header = list(pd.read_csv(path, nrows=0).columns)
    if requested:
        if requested not in header:
            raise ValueError("{} has no timestamp column {!r}".format(path, requested))
        return requested
    for name in TIME_COLUMNS:
        if name in header:
            return name
    raise ValueError("No supported timestamp column in {}. Use --timestamp-column.".format(path))


def row_indices(path, events, positions, json_unit="us", csv_unit="us",
                time_reference="elapsed", time_column=None, chunksize=250000,
                zero_timestamps="keep"):
    """Use actual CSV timestamps; keep integer epochs exact before subtraction.

    elapsed: JSON entry 0 is acquisition start, aligned with the first usable
             timestamp of EACH stream (the supplied FAN script convention).
    absolute: JSON and CSV share an epoch; preserve analog/digital start offsets.
    No timestamp unit or clock alignment is guessed from numerical magnitude.
    """
    if chunksize <= 0:
        raise ValueError("--chunk-size must be positive.")
    path = Path(path)
    column = timestamp_column(path, time_column)
    baseline = events[0] if time_reference == "elapsed" else Decimal(0)
    targets = [(events[i] - baseline) * UNITS[json_unit] / UNITS[csv_unit] for i in positions]
    found = [None] * len(targets)
    origin, previous, offset, final_value = None, None, 0, None
    for chunk in pd.read_csv(path, usecols=[column], dtype=str, keep_default_na=False, chunksize=chunksize):
        # Reading strings first avoids float rounding of 17-digit JSON/CSV epochs.
        numeric = pd.to_numeric(chunk[column], errors="raise")
        values = numeric.to_numpy()
        if not np.isfinite(values).all():
            raise ValueError("Missing/non-finite timestamp in {}".format(path))
        if values.dtype.kind == "f" and np.any(np.abs(values) > 2 ** 53):
            raise ValueError("Large epoch timestamps must be stored as integer text: {}".format(path))
        if values.dtype.kind == "u":
            if len(values) and values.max() > np.iinfo(np.int64).max:
                raise ValueError("Timestamp exceeds int64 in {}".format(path))
            values = values.astype(np.int64)
        row_positions = np.arange(len(values), dtype=np.int64) + offset
        offset += len(values)
        if zero_timestamps == "ignore":
            mask = values != 0
            values, row_positions = values[mask], row_positions[mask]
        if not len(values):
            continue
        if previous is not None and values[0] < previous:
            raise ValueError("Timestamp moved backwards across chunks: {}".format(path))
        if np.any(values[1:] < values[:-1]):
            raise ValueError("Timestamp moved backwards inside a chunk: {}".format(path))
        previous = values[-1].item()
        if origin is None:
            origin = values[0].item()
            if time_reference == "absolute" and targets[0] < Decimal(str(origin)):
                raise ValueError("First selected JSON boundary precedes the CSV: {}".format(path))
        comparisons = values - origin if time_reference == "elapsed" else values
        final_value = comparisons[-1].item()
        for i, target in enumerate(targets):
            if found[i] is not None:
                continue
            # ceil makes 'first row >= target' exact even for fractional unit conversion.
            if comparisons.dtype.kind in "iu":
                needle = int(target.to_integral_value(rounding=ROUND_CEILING))
                if needle > np.iinfo(np.int64).max:
                    continue
                if needle < np.iinfo(np.int64).min:
                    raise ValueError("Boundary is below the supported timestamp range.")
            else:
                needle = float(target)
            position = int(np.searchsorted(comparisons, needle, side="left"))
            if position < len(comparisons):
                found[i] = int(row_positions[position])
        if all(x is not None for x in found):
            break
    if any(x is None for x in found):
        raise ValueError("CSV does not cover selected boundaries: {}. Targets={} {}; last={} ({}).".format(
            path, [str(x) for x in targets], csv_unit, final_value, time_reference))
    if any(b <= a for a, b in zip(found, found[1:])):
        raise ValueError("Selected boundaries collapse to the same CSV row: {}".format(path))
    return found, column


def new_sample_root(row):
    """Detect analog/digital layout from paths, without a source column."""
    signals = STATE_SIGNALS[row["state"]]
    matches = [basename(row[s]).lower() == SIGNAL_FILES[s] for s in signals]
    if not any(matches):
        return None
    if not all(matches):
        raise ValueError("Mixed signal-file and analog/digital paths in one metadata row.")
    files = {s: Path(row[s]) for s in signals}
    parents = {p.parent for p in files.values()}
    if len(parents) != 1:
        raise ValueError("All signal paths for a state must share one state directory.")
    state_folder = next(iter(parents))
    if state_folder.name.lower() != row["state"]:
        raise ValueError("State directory does not match metadata: {}".format(state_folder))
    for signal, path in files.items():
        if path != state_folder / SIGNAL_FILES[signal]:
            raise ValueError("Unexpected path for {}: {}".format(signal, path))
    return state_folder.parent


def make_record(row, signal, points):
    # State is included in the ID so id remains unique, like the FAN output.
    identity = "{}_{}@{}@{}@{}".format(
        signal, row["fault_id"], row["pump_id"], row["sample_id"], row["state"])
    return {"id": identity, "indices": "#".join(str(p) for p in points), "key": signal, "state": row["state"]}


def generate(args):
    meta = read_table(args.combined_meta, META_COLUMNS)
    validate_meta(meta)
    regions = load_final_regions(args.air_regions, args.water_regions)
    config = {}
    if args.config:
        with Path(args.config).open(encoding="utf-8-sig") as handle:
            config = json.load(handle)
        if not isinstance(config, dict) or not all(s in config for s in STATE_SIGNALS):
            raise ValueError("--config must be a PUMP configuration containing air and water.")
    outputs, checked_roots, stream_cache = [], set(), {}
    old_count = new_count = 0
    # Fail on unknown boundary selection before reading any large new CSV.
    selections = {}
    for row in meta.to_dict("records"):
        root = new_sample_root(row)
        if root is not None:
            state = row["state"]
            events = parse_events(read_state_events(root, state), "{} / {}".format(root, state))
            explicit = args.air_json_positions if state == "air" else args.water_json_positions
            selections[(root, state)] = (events, select_positions(events, state, explicit, config))
    for row in meta.to_dict("records"):
        state = row["state"]
        root = new_sample_root(row)
        if root is None:
            for signal in STATE_SIGNALS[state]:
                key = (row["fault_id"], row["pump_id"], row["sample_id"], state, signal)
                if key not in regions:
                    raise ValueError("Metadata row has no FINAL indices: {}".format(key))
                outputs.append(make_record(row, signal, regions[key]))
                old_count += 1
            continue
        if root not in checked_roots:
            check_new_sample(root)
            checked_roots.add(root)
        events, positions = selections[(root, state)]
        for signal in STATE_SIGNALS[state]:
            path = Path(row[signal])
            cache_key = (path, state)
            if cache_key not in stream_cache:
                stream_cache[cache_key], column = row_indices(
                    path, events, positions, args.json_unit, args.csv_time_unit,
                    args.time_reference, args.timestamp_column, args.chunk_size,
                    args.zero_timestamps)
                print("{} | {} | rows {}".format(path, column, stream_cache[cache_key]))
            outputs.append(make_record(row, signal, stream_cache[cache_key]))
            new_count += 1
    result = pd.DataFrame(outputs, columns=OUTPUT_COLUMNS)
    if result.id.duplicated().any():
        raise ValueError("Duplicate output signal IDs.")
    expected = sum(len(STATE_SIGNALS[s]) for s in meta.state)
    if len(result) != expected:
        raise ValueError("Signalwise output does not match metadata coverage.")
    save_table(result, args.output)
    print("Saved {}: {} signal rows ({} preserved; {} converted).".format(
        args.output, len(result), old_count, new_count))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined-meta", default="pump_meta_combined_all_faults_mapped.csv")
    parser.add_argument("--air-regions", default="PUMP_DATA__FINAL__timestamps_signal_samplewise_regions_air.csv")
    parser.add_argument("--water-regions", default="PUMP_DATA__FINAL__timestamps_signal_samplewise_regions_water.csv")
    parser.add_argument("--output", default="pump_signalwise_timestamp_combined.csv")
    parser.add_argument("--config", help="Optional PUMP_config_inference.json; no class filtering or channel remapping.")
    parser.add_argument("--air-json-positions", help="Exactly 3 zero-based event positions, e.g. a,b,c replaced with actual positions.")
    parser.add_argument("--water-json-positions", help="Exactly 5 zero-based event positions; normally 1,2,3,4,5.")
    parser.add_argument("--json-unit", choices=tuple(UNITS), default="us")
    parser.add_argument("--csv-time-unit", choices=tuple(UNITS), default="us")
    parser.add_argument("--time-reference", choices=("elapsed", "absolute"), default="elapsed")
    parser.add_argument("--timestamp-column", help="Override automatic column selection, e.g. F_Timestamp.")
    parser.add_argument("--zero-timestamps", choices=("keep", "ignore"), default="keep",
                        help="ignore skips zeros during lookup but preserves original row numbering.")
    parser.add_argument("--chunk-size", type=int, default=250000)
    args = parser.parse_args()
    try:
        generate(args)
    except (ValueError, OSError, KeyError, InvalidOperation) as error:
        parser.exit(1, "ERROR: {}\n".format(error))


if __name__ == "__main__":
    main()
