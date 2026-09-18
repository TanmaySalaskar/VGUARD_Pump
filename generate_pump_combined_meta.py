"""Build FAN-style PUMP metadata from Ahmedabad records and Air/Water folders.

Python 3.9+; dependencies: pandas, numpy. Run --help for CLI options.
Keep this file beside generate_pump_signalwise_timestamp.py.
The channel mapping below is the mapping confirmed on 17 September 2026.
"""

import argparse
import json
import ntpath
import re
from pathlib import Path, PureWindowsPath

import pandas as pd


SIGNALS = ("AC1", "GYR", "CTB", "CTR", "CV1", "VRY", "PRS", "SMG", "WTF")
STATE_SIGNALS = {
    "air": tuple(s for s in SIGNALS if s not in ("PRS", "WTF")),
    "water": SIGNALS,
}
BOUNDARY_COUNTS = {"air": 3, "water": 5}
SIGNAL_FILES = {s: "digital.csv" if s in ("AC1", "GYR") else "analog.csv" for s in SIGNALS}
CHANNEL_MAP = {
    "AC1": ("ax", "ay", "az"),
    "GYR": ("gx", "gy", "gz"),
    "CTB": ("ct3",),
    "CTR": ("ct1",),
    "CV1": ("ct4", "vt3"),
    "VRY": ("vt1", "vt2"),
    "PRS": ("prs",),
    "SMG": ("mag",),
    "WTF": ("wtf",),
}  # CTY / ct2 deliberately omitted.
DIGITAL_ALIASES = dict(zip(("ax", "ay", "az", "gx", "gy", "gz"),
                           ("ac1", "ac2", "ac3", "gy1", "gy2", "gy3")))
META_COLUMNS = ["fault_id", "pump_id", "sample_id", "state", *SIGNALS]
KEY_COLUMNS = META_COLUMNS[:4]
MAP_COLUMNS = ["fault_id", "input_pump_id", "pump_id", "input_sample_id", "sample_id", "sample_root"]
DEFAULT_TEST_REGEX = r"^(?P<pump_id>\d+)_.*_test_(?P<sample_id>\d+)$"


def fault_id(value):
    value = str(value).strip().rstrip("_")
    if not re.fullmatch(r"\d+", value):
        raise ValueError("Invalid fault ID: {!r}".format(value))
    return value + "_"


def integer(value, label="ID"):
    value = str(value).strip()
    if not re.fullmatch(r"\d+", value):
        raise ValueError("{} must be a nonnegative integer: {!r}".format(label, value))
    return int(value)


def read_table(path, required):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError("{} is missing columns {}".format(path, sorted(missing)))
    return frame


def save_table(frame, path):
    """Replace an output only after the whole CSV has been written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def basename(value):
    return re.split(r"[\\/]", str(value).strip())[-1]


def join_path(root, name):
    # Keep Windows recording paths valid even when building metadata on Linux.
    if "\\" in root or re.match(r"^[A-Za-z]:", root):
        return str(PureWindowsPath(root) / name)
    return str(Path(root) / name)


def load_final_regions(air_path, water_path):
    """Return exact supplied indices, keyed by fault/pump/sample/state/signal."""
    regions = {}
    for state, path in (("air", air_path), ("water", water_path)):
        columns = [str(i) for i in range(BOUNDARY_COUNTS[state])]
        frame = read_table(path, ["key", *columns])
        for row in frame.to_dict("records"):
            match = re.fullmatch(r"(\d+)_([0-9]+)_([0-9]+)_([A-Za-z0-9]+)", row["key"])
            if not match:
                raise ValueError("Invalid region key: {}".format(row["key"]))
            f, pump, sample, signal = match.groups()
            if signal not in STATE_SIGNALS[state]:
                raise ValueError("Unexpected signal {} in {}".format(signal, path))
            key = (fault_id(f), int(pump), int(sample), state, signal)
            points = tuple(integer(row[c], "row index") for c in columns)
            if any(b <= a for a, b in zip(points, points[1:])):
                raise ValueError("Region boundaries must increase: {}".format(key))
            if key in regions:
                raise ValueError("Duplicate region key: {}".format(key))
            regions[key] = points
    return regions


def validate_meta(frame):
    if list(frame.columns) != META_COLUMNS:
        raise ValueError("Expected metadata columns: {}".format(META_COLUMNS))
    if frame.empty:
        raise ValueError("No complete samples are available.")
    frame["fault_id"] = frame["fault_id"].map(fault_id)
    for c in ("pump_id", "sample_id"):
        frame[c] = frame[c].map(integer)
    frame["state"] = frame["state"].str.lower()
    if frame.duplicated(KEY_COLUMNS).any():
        raise ValueError("Duplicate fault/pump/sample/state in metadata.")
    for row in frame.to_dict("records"):
        state = row["state"]
        if state not in STATE_SIGNALS:
            raise ValueError("Unknown state: {}".format(state))
        for signal in SIGNALS:
            if bool(str(row[signal]).strip()) != (signal in STATE_SIGNALS[state]):
                raise ValueError("Unexpected missing/present path for {} in {}".format(signal, row))
    states = frame.groupby(KEY_COLUMNS[:3])["state"].agg(set)
    if any(x != {"air", "water"} for x in states):
        raise ValueError("Every sample must have both Air and Water metadata rows.")
    return frame


def read_old_meta(path, regions, old_root=None, old_root_from=None, check_files=False):
    source = read_table(path, ["id", "filename", "filepaths", "state"])
    records, seen, all_ids = [], set(), {}
    for row in source.to_dict("records"):
        state = row["state"].strip().lower()
        match = re.search(r"_(\d+)_(\d+)_In\s+(air|water)(?:_|\s|$)", row["filename"], re.I)
        if not match or state not in STATE_SIGNALS or match[3].lower() != state:
            raise ValueError("Cannot parse pump/sample/state: {}".format(row["filename"]))
        f, pump, sample = fault_id(row["id"]), int(match[1]), int(match[2])
        all_ids[f] = max(all_ids.get(f, -1), pump)
        key = (f, pump, sample, state)
        if key in seen:
            raise ValueError("Duplicate input metadata identity: {}".format(key))
        seen.add(key)
        # Completeness comes from the FINAL files, with no hand-written sample exclusions.
        if not all((f, pump, sample, st, sig) in regions
                   for st in STATE_SIGNALS for sig in STATE_SIGNALS[st]):
            continue
        root = row["filepaths"].strip()
        if old_root is not None:
            original = ntpath.normpath(root)
            prefix = ntpath.normpath(old_root_from)
            if not original.lower().startswith(prefix.lower().rstrip("\\") + "\\"):
                raise ValueError("Path is outside --old-root-from: {}".format(root))
            relative = ntpath.relpath(original, prefix)
            root = str(Path(old_root).joinpath(*PureWindowsPath(relative).parts))
        output = dict(zip(KEY_COLUMNS, key))
        for signal in SIGNALS:
            output[signal] = (join_path(root, signal + " " + row["filename"] + ".csv")
                              if signal in STATE_SIGNALS[state] else "")
            if check_files and output[signal] and not Path(output[signal]).is_file():
                raise FileNotFoundError(output[signal])
        records.append(output)
    # Also reject a region sample which cannot be matched to original metadata.
    orphaned = {k[:4] for k in regions} - seen
    if orphaned:
        raise ValueError("FINAL regions have unmatched metadata keys: {}".format(sorted(orphaned)[:5]))
    result = pd.DataFrame(records, columns=META_COLUMNS)
    validate_meta(result)
    print("Old metadata: {} state rows retained; {} excluded by FINAL region coverage.".format(
        len(result), len(source) - len(result)))
    return result, all_ids


def state_directories(root):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    result = {}
    for folder in root.iterdir():
        if folder.is_dir() and folder.name.lower() in STATE_SIGNALS:
            state = folder.name.lower()
            if state in result:
                raise ValueError("Duplicate case variants for {} under {}".format(state, root))
            result[state] = folder
    if set(result) != set(STATE_SIGNALS):
        raise ValueError("Both Air and Water folders are required: {}".format(root))
    return result


def read_state_events(root, state):
    """Match testname_Air/testname_Water exactly, allowing a #suffix on the folder."""
    root = Path(root)
    with (root / "statewise_changepoints.json").open(encoding="utf-8-sig") as handle:
        data = json.load(handle, parse_float=str)
    if not isinstance(data, dict):
        raise ValueError("Expected an object in statewise_changepoints.json")
    names = {root.name.lower(), root.name.split("#", 1)[0].lower()}
    candidates = [k for k in data if k.lower() in {n + "_" + state for n in names}]
    if len(candidates) != 1:
        raise ValueError("Expected exactly one matching {} JSON key for {}; found {}".format(
            state, root, candidates))
    value = data[candidates[0]]
    if isinstance(value, str):
        return [v.strip() for v in value.strip().rstrip(",").split(",")]
    if isinstance(value, list):
        return value
    raise ValueError("Expected comma-separated timestamps or a timestamp array.")


def check_new_sample(root):
    folders = state_directories(root)
    for state, folder in folders.items():
        read_state_events(root, state)
        for filename in ("analog.csv", "digital.csv"):
            path = folder / filename
            header = set(pd.read_csv(path, nrows=0).columns)
            for signal in STATE_SIGNALS[state]:
                if SIGNAL_FILES[signal] != filename:
                    continue
                for i, channel in enumerate(CHANNEL_MAP[signal]):
                    accepted = {channel, "{}_column_{}".format(signal, i)}
                    if channel in DIGITAL_ALIASES:
                        accepted.add(DIGITAL_ALIASES[channel])
                    if not header.intersection(accepted):
                        raise ValueError("{} missing {} (accepted headers: {})".format(path, channel, sorted(accepted)))
    return folders


def scan_new_data(roots, manifest, test_regex):
    samples = []
    pattern = re.compile(test_regex, re.I)
    for root in map(Path, roots):
        if not root.is_dir():
            raise FileNotFoundError(root)
        faults = [root] if re.fullmatch(r"\d+_?", root.name) else sorted(
            p for p in root.iterdir() if p.is_dir() and re.fullmatch(r"\d+_?", p.name))
        for fault_folder in faults:
            for sample_folder in sorted(p for p in fault_folder.iterdir() if p.is_dir()):
                match = pattern.fullmatch(sample_folder.name.split("#", 1)[0])
                if not match:
                    raise ValueError("Cannot extract IDs from {}. Use --test-regex or --new-manifest.".format(sample_folder))
                samples.append({"fault_id": fault_id(fault_folder.name),
                                "input_pump_id": integer(match["pump_id"]),
                                "input_sample_id": integer(match["sample_id"]),
                                "sample_root": str(sample_folder.resolve())})
    if manifest:
        path = Path(manifest).resolve()
        frame = read_table(path, ["fault_id", "pump_id", "sample_id", "sample_root"])
        for row in frame.to_dict("records"):
            root = Path(row["sample_root"])
            if not root.is_absolute():
                root = path.parent / root
            samples.append({"fault_id": fault_id(row["fault_id"]),
                            "input_pump_id": integer(row["pump_id"]),
                            "input_sample_id": integer(row["sample_id"]),
                            "sample_root": str(root.resolve())})
    by_path = {}
    for row in samples:
        root = row["sample_root"]
        if root in by_path and row != by_path[root]:
            raise ValueError("Conflicting identities for {}".format(root))
        by_path[root] = row
    return sorted(by_path.values(), key=lambda r: (r["fault_id"], r["input_pump_id"], r["input_sample_id"], r["sample_root"]))


def assign_ids(samples, old_max, mapping_path):
    """Persist numeric assignments so adding another sample does not renumber previous samples."""
    mapping_path = Path(mapping_path)
    mapping = read_table(mapping_path, MAP_COLUMNS)[MAP_COLUMNS] if mapping_path.exists() else pd.DataFrame(columns=MAP_COLUMNS)
    rows = mapping.to_dict("records")
    by_path, pump_lookup, allocated_pumps, used_samples = {}, {}, {}, set()
    for row in rows:
        row["fault_id"] = fault_id(row["fault_id"])
        for c in ("input_pump_id", "pump_id", "input_sample_id", "sample_id"):
            row[c] = integer(row[c])
        f, original, pump, sample = (row[c] for c in ("fault_id", "input_pump_id", "pump_id", "sample_id"))
        if pump <= old_max.get(f, -1):
            raise ValueError("Saved pump_id map overlaps old pump IDs for {}. Review the map.".format(f))
        if (f, original) in pump_lookup and pump_lookup[(f, original)] != pump:
            raise ValueError("One input pump maps to multiple output pumps.")
        if (f, pump) in allocated_pumps and allocated_pumps[(f, pump)] != original:
            raise ValueError("Multiple input pumps map to one output pump.")
        if (f, pump, sample) in used_samples or row["sample_root"] in by_path:
            raise ValueError("Duplicate sample in pump_id_map.csv.")
        pump_lookup[(f, original)], allocated_pumps[(f, pump)] = pump, original
        used_samples.add((f, pump, sample))
        by_path[row["sample_root"]] = row
    selected = []
    for item in samples:
        f, original, root = item["fault_id"], item["input_pump_id"], item["sample_root"]
        if root in by_path:
            row = by_path[root]
            if any(row[c] != item[c] for c in ("fault_id", "input_pump_id", "input_sample_id")):
                raise ValueError("Sample identity changed for {}".format(root))
        else:
            if (f, original) not in pump_lookup:
                pump_lookup[(f, original)] = max([old_max.get(f, -1)] + [p for (fid, _), p in pump_lookup.items() if fid == f]) + 1
            pump = pump_lookup[(f, original)]
            sample = max([-1] + [s for fid, p, s in used_samples if fid == f and p == pump]) + 1
            row = dict(item, pump_id=pump, sample_id=sample)
            rows.append(row)
            by_path[root] = row
            used_samples.add((f, pump, sample))
        selected.append(row)
    return selected, pd.DataFrame(rows, columns=MAP_COLUMNS)


def generate(args):
    regions = load_final_regions(args.air_regions, args.water_regions)
    old, old_max = read_old_meta(args.old_meta, regions, args.old_root, args.old_root_from, args.check_old_files)
    output = Path(args.output_dir)
    mapping_path = Path(args.id_map) if args.id_map else output / "pump_id_map.csv"
    samples = [] if args.data_mode == "old-only" else scan_new_data(args.new_root, args.new_manifest, args.test_regex)
    folders = {row["sample_root"]: check_new_sample(row["sample_root"]) for row in samples}
    assigned, mapping = assign_ids(samples, old_max, mapping_path)
    records = []
    for item in assigned:
        for state, folder in folders[item["sample_root"]].items():
            row = {c: item[c] for c in KEY_COLUMNS[:3]}
            row["state"] = state
            row.update({signal: str(folder / SIGNAL_FILES[signal]) if signal in STATE_SIGNALS[state] else "" for signal in SIGNALS})
            records.append(row)
    new = pd.DataFrame(records, columns=META_COLUMNS)
    if args.data_mode == "prefer-new":
        old = old[~old.fault_id.isin(new.fault_id.unique())]
    combined = pd.concat([old, new], ignore_index=True)
    validate_meta(combined)
    combined = combined.sort_values(KEY_COLUMNS, kind="stable").reset_index(drop=True)
    # Save the identity map first: a retry can then reuse every assignment.
    if samples:
        save_table(mapping, mapping_path)
    path = output / "pump_meta_combined_all_faults_mapped.csv"
    save_table(combined, path)
    print("Saved {}: {} state rows, {} samples, {} fault IDs; {} new state rows.".format(
        path, len(combined), len(combined) // 2, combined.fault_id.nunique(), len(new)))
    if samples:
        print("Keep {} for stable IDs on subsequent runs.".format(mapping_path))
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-meta", default="PUMP_DATA__meta_ahmedabad.csv")
    parser.add_argument("--air-regions", default="PUMP_DATA__FINAL__timestamps_signal_samplewise_regions_air.csv")
    parser.add_argument("--water-regions", default="PUMP_DATA__FINAL__timestamps_signal_samplewise_regions_water.csv")
    parser.add_argument("--new-root", action="append", default=[], help="Root containing fault-ID folders; may be repeated.")
    parser.add_argument("--new-manifest", help="Optional CSV: fault_id,pump_id,sample_id,sample_root; supports arbitrary test names.")
    parser.add_argument("--test-regex", default=DEFAULT_TEST_REGEX, help="Folder regex with named pump_id and sample_id groups.")
    parser.add_argument("--data-mode", choices=("all", "old-only", "prefer-new"), default="all")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--id-map", help="Existing persistent numeric-ID mapping CSV; default: output-dir/pump_id_map.csv.")
    parser.add_argument("--old-root", help="Optional current location of the old recording root.")
    parser.add_argument("--old-root-from", default=r"D:\VGuard Ahemedebad Data")
    parser.add_argument("--check-old-files", action="store_true", help="Verify old signal paths exist on this computer.")
    args = parser.parse_args()
    try:
        generate(args)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, "ERROR: {}\n".format(error))


if __name__ == "__main__":
    main()
