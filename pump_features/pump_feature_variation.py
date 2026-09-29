"""Describe sample, pump and fault variation in extracted PUMP feature CSVs.

Python 3.9+; dependencies: numpy, pandas, matplotlib.
Run: python pump_feature_variation.py --data-dir . --output-dir variation_results
See README.md for definitions, thresholds, and model-selection precautions.
"""
import argparse
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE_FILES = ["ctvt_singlefault_peaks_freq", "digital_feats_updt",
              "physics_features", "stats_features_2"]
KEYS = ["fault_id", "pump_id", "sample_id"]
META = set(KEYS + ["sample_id_key", "pump_id_key", "fan_id", "fan_id_key",
                  "fault_type", "source_namespace", "source", "dataset",
                  "state", "region", "division", "test_name", "folder_name"])


def discover_files(folder):
    """Accept original names or one numbered download copy; reject ambiguity."""
    files = []
    for base in BASE_FILES:
        matches = sorted(p for p in Path(folder).glob("*.csv")
                         if re.fullmatch(re.escape(base) + r"(?:\s*\(\d+\))?", p.stem))
        if len(matches) != 1:
            raise ValueError("Expected exactly one {} CSV in {}; found {}. "
                             "Use --files to choose explicitly.".format(base, folder, matches))
        files.append(matches[0])
    return files


def load_features(files, keys_csv=None):
    """Outer-align by verified run identifiers; never merge by row position."""
    blocks, origins, audit, expected_keys = [], {}, [], None
    for path in files:
        path = Path(path)
        d = pd.read_csv(path, dtype={c: "string" for c in META})
        missing = set(KEYS + ["sample_id_key"]) - set(d.columns)
        if missing:
            raise ValueError("{} missing identifiers: {}".format(path.name, missing))
        for c in KEYS + ["sample_id_key"]:
            if d[c].isna().any():
                raise ValueError("Missing {} in {}".format(c, path.name))
            d[c] = d[c].str.strip()
        # Canonicalize numeric pump/sample IDs, preserving fault leading zeros.
        for c in ["pump_id", "sample_id"]:
            d[c] = d[c].str.replace(r"^(\d+)\.0+$", r"\1", regex=True)
        canonical = d[KEYS].agg("@".join, axis=1)
        if not canonical.eq(d["sample_id_key"]).all():
            raise ValueError("sample_id_key disagrees with fault/pump/sample IDs in " + path.name)
        if canonical.duplicated().any():
            raise ValueError("Duplicate run IDs in " + path.name +
                             "; aggregate divisions explicitly before this analysis.")
        candidates = [c for c in d.columns if c not in META and not c.startswith("Unnamed:")]
        x = d[candidates].apply(pd.to_numeric, errors="coerce")
        bad = d[candidates].notna() & x.isna()
        if bad.any().any():
            raise ValueError("Non-numeric feature values in {}: {}".format(
                path.name, bad.columns[bad.any()].tolist()))
        overlap = set(candidates) & set(origins)
        if overlap:
            raise ValueError("Duplicate feature names across files: " + str(sorted(overlap)))
        current = set(canonical)
        if expected_keys is None:
            expected_keys = current
        audit.append({"file": path.name, "rows": len(d), "features": len(candidates),
                      "missing_keys_vs_first": len(expected_keys - current),
                      "extra_keys_vs_first": len(current - expected_keys),
                      "infinite_values": int(np.isinf(x.to_numpy()).sum())})
        x = x.replace([np.inf, -np.inf], np.nan)
        x.index = pd.MultiIndex.from_frame(d[KEYS])
        blocks.append(x)
        origins.update({c: path.name for c in candidates})
    x = pd.concat(blocks, axis=1, join="outer").sort_index()
    if keys_csv:
        keep = pd.read_csv(keys_csv, dtype="string")
        if "sample_id_key" not in keep:
            raise ValueError("--keys-csv requires a sample_id_key column")
        wanted = set(keep["sample_id_key"].dropna().str.strip())
        actual = pd.Index(["@".join(k) for k in x.index])
        if wanted - set(actual):
            raise ValueError("Some requested training IDs do not exist in the feature files")
        x = x.loc[actual.isin(wanted)]
    if x.empty or x.index.get_level_values("fault_id").nunique() < 2:
        raise ValueError("At least two faults and nonempty samples are required")
    return x, origins, pd.DataFrame(audit)


def decompose(x):
    """Exact descriptive variance decomposition with equal hierarchical weights.

    Each sample gets 1/(F * pumps_in_fault * samples_in_pump) weight.
    All variances use ddof=0. Missing observations are omitted feature by feature.
    Report support separately: a singleton is not evidence of repeatability.
    """
    group = x.groupby(level=["fault_id", "pump_id"])
    count = group.count()
    pump_mean = group.mean()
    sample_var = group.var(ddof=0)
    fault_mean = pump_mean.groupby(level="fault_id").mean()
    pump_var = pump_mean.groupby(level="fault_id").var(ddof=0)
    sample_var_fault = sample_var.groupby(level="fault_id").mean()
    sample_component = sample_var_fault.mean(axis=0)
    pump_component = pump_var.mean(axis=0)
    fault_component = fault_mean.var(axis=0, ddof=0)
    return dict(count=count, pump_mean=pump_mean, sample_var=sample_var,
                fault_mean=fault_mean, pump_var=pump_var,
                sample_var_fault=sample_var_fault, sample_component=sample_component,
                pump_component=pump_component, fault_component=fault_component)


def rank_features(x, origins, parts, args):
    s, p, f = (parts[k] for k in ["sample_component", "pump_component", "fault_component"])
    total = s + p + f
    denom = total.where(total > 0)
    result = pd.DataFrame({"source_file": pd.Series(origins),
                           "sample_sd": np.sqrt(s), "pump_sd": np.sqrt(p),
                           "fault_sd": np.sqrt(f), "total_sd": np.sqrt(total),
                           "sample_variation_pct": 100 * s / denom,
                           "pump_variation_pct": 100 * p / denom,
                           "fault_variation_pct": 100 * f / denom})
    result.index.name = "feature"
    result["separation_ratio"] = f / (s + p).where((s + p) > 0)
    result.loc[((s + p) == 0) & (f > 0), "separation_ratio"] = np.inf
    result["finite_coverage_pct"] = 100 * x.notna().mean()
    result["min_fault_coverage_pct"] = 100 * x.notna().groupby(level="fault_id").mean().min()
    result["min_samples_per_pump"] = parts["count"].min()
    result["min_pumps_per_fault"] = parts["pump_mean"].notna().groupby(level="fault_id").sum().min()
    result["n_faults_present"] = parts["fault_mean"].notna().sum()
    result["unique_values"] = x.nunique()
    result["eligible"] = ((result["unique_values"] > 1)
                          & (result["min_fault_coverage_pct"] >= 100 * args.min_coverage)
                          & (result["min_samples_per_pump"] >= args.min_samples)
                          & (result["min_pumps_per_fault"] >= args.min_pumps)
                          & (result["n_faults_present"] == len(parts["fault_mean"]))
                          & (total > 0))
    result["candidate"] = (result["eligible"]
                            & (result["sample_variation_pct"] <= args.max_sample_pct)
                            & (result["pump_variation_pct"] <= args.max_pump_pct)
                            & (result["fault_variation_pct"] >= args.min_fault_pct))
    def reasons(row):
        r = []
        if row.unique_values <= 1:
            r.append("constant_or_all_missing")
        if row.min_fault_coverage_pct < 100 * args.min_coverage:
            r.append("low_coverage")
        if row.min_samples_per_pump < args.min_samples:
            r.append("too_few_samples")
        if row.min_pumps_per_fault < args.min_pumps:
            r.append("too_few_pumps")
        if row.n_faults_present != len(parts["fault_mean"]):
            r.append("missing_fault")
        return ";".join(r)
    result["exclusion_reason"] = result.apply(reasons, axis=1)
    result = result.sort_values(["eligible", "fault_variation_pct", "pump_variation_pct"],
                                ascending=[False, False, True], na_position="last")
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    return result


def pairwise_rankings(parts, ranking, top_n):
    """Per-pair standardized mean distance; an effect size, not accuracy."""
    rows = []
    mu = parts["fault_mean"]
    within = parts["sample_var_fault"] + parts["pump_var"]
    eligible = ranking.index[ranking.eligible]
    for a, b in itertools.combinations(mu.index, 2):
        delta = (mu.loc[a, eligible] - mu.loc[b, eligible]).abs()
        pooled = (within.loc[a, eligible] + within.loc[b, eligible]) / 2
        effect = delta / np.sqrt(pooled.where(pooled > 0))
        effect.loc[(pooled == 0) & (delta > 0)] = np.inf
        effect.loc[(pooled == 0) & (delta == 0)] = 0
        best = effect.sort_values(ascending=False, kind="stable").head(top_n)
        for i, (feature, value) in enumerate(best.items(), 1):
            rows.append({"fault_a": a, "fault_b": b, "pair_rank": i,
                         "feature": feature, "standardized_mean_gap": value,
                         "mean_a": mu.loc[a, feature], "mean_b": mu.loc[b, feature],
                         "global_candidate": bool(ranking.loc[feature, "candidate"])})
    return pd.DataFrame(rows, columns=["fault_a", "fault_b", "pair_rank", "feature",
                                      "standardized_mean_gap", "mean_a", "mean_b",
                                      "global_candidate"])


def write_details(parts, x, out):
    def long(df, name):
        frame = df.rename_axis(columns="feature")
        try:
            return frame.stack(future_stack=True).rename(name)
        except TypeError:  # pandas before 2.1
            return frame.stack(dropna=False).rename(name)
    pump = pd.concat([long(parts["count"], "n_samples"),
                      long(parts["pump_mean"], "pump_mean"),
                      long(np.sqrt(parts["sample_var"]), "sample_sd")], axis=1)
    pump.to_csv(out / "variation_within_each_pump.csv")
    fault = pd.concat([long(parts["fault_mean"], "fault_mean"),
                       long(np.sqrt(parts["sample_var_fault"]), "sample_sd"),
                       long(np.sqrt(parts["pump_var"]), "pump_sd"),
                       long(parts["pump_mean"].notna().groupby(level="fault_id").sum(), "n_pumps")], axis=1)
    fault.to_csv(out / "variation_within_each_fault.csv")
    x.index.to_frame(index=False).groupby(["fault_id", "pump_id"]).size().rename(
        "n_samples").to_csv(out / "sample_counts.csv")


def make_plots(x, parts, ranking, out, top_n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = out / "plots"
    folder.mkdir(exist_ok=True)
    top = ranking.loc[ranking.eligible].head(top_n)
    if top.empty:
        return
    fig, ax = plt.subplots(figsize=(13, max(5, len(top) * .34)))
    left = np.zeros(len(top))
    for col, label, color in [("sample_variation_pct", "Between samples", "#e9a33b"),
                              ("pump_variation_pct", "Between pumps", "#db6472"),
                              ("fault_variation_pct", "Between faults", "#338a91")]:
        ax.barh(np.arange(len(top)), top[col], left=left, label=label, color=color)
        left += top[col].to_numpy()
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(top.index, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of total weighted variance (%)")
    ax.set_title("Highest ranked features: larger teal share is better")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.12), ncol=3)
    fig.tight_layout()
    fig.savefig(folder / "top_feature_variation.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # Within-feature z scores, using the same hierarchical weighting as ranking.
    means = parts["fault_mean"][top.index]
    z = (means - means.mean()) / ranking.loc[top.index, "total_sd"]
    fig, ax = plt.subplots(figsize=(12, max(5, len(top) * .34)))
    limit = max(1., float(np.nanmax(np.abs(z.to_numpy()))))
    im = ax.imshow(z.T, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_xticks(np.arange(len(z)))
    ax.set_xticklabels(z.index, rotation=65, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(top.index, fontsize=8)
    ax.set_title("Fault means for highest ranked features")
    fig.colorbar(im, ax=ax, label="Centered fault mean / weighted total SD")
    fig.tight_layout()
    fig.savefig(folder / "fault_feature_heatmap.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # Six original-unit plots expose sample scatter and individual pump means.
    rng = np.random.default_rng(42)
    faults = parts["fault_mean"].index.tolist()
    for number, feature in enumerate(top.index[:6], 1):
        fig, ax = plt.subplots(figsize=(13, 5))
        for j, fault in enumerate(faults):
            values = x.xs(fault, level="fault_id")[feature]
            pumps = values.index.get_level_values("pump_id").unique()
            for k, pump in enumerate(pumps):
                v = values.xs(pump, level="pump_id").dropna()
                xpos = j + (k - (len(pumps) - 1) / 2) * .10
                color = plt.get_cmap("tab10")(k % 10)
                ax.scatter(xpos + rng.uniform(-.015, .015, len(v)), v,
                           color=color, s=15, alpha=.65)
                ax.scatter(xpos, v.mean(), color=color, marker="_", s=130)
            ax.scatter(j, parts["fault_mean"].loc[fault, feature],
                       color="black", marker="D", s=22, zorder=4)
        ax.set_xticks(np.arange(len(faults)))
        ax.set_xticklabels(faults, rotation=65, ha="right", fontsize=8)
        ax.set_ylabel("Original feature value")
        ax.set_title(feature + "\nDots: samples; colored bars: pump means; black diamonds: fault means")
        ax.grid(axis="y", alpha=.2)
        fig.tight_layout()
        fig.savefig(folder / ("feature_{:02d}.png".format(number)), dpi=160, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--files", type=Path, nargs="+", help="Explicit input CSVs; overrides discovery")
    parser.add_argument("--output-dir", type=Path, default=Path("variation_results"))
    parser.add_argument("--keys-csv", type=Path, help="Optional training-only sample_id_key list")
    parser.add_argument("--min-coverage", type=float, default=.95)
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument("--min-pumps", type=int, default=2)
    parser.add_argument("--max-sample-pct", type=float, default=20.)
    parser.add_argument("--max-pump-pct", type=float, default=20.)
    parser.add_argument("--min-fault-pct", type=float, default=60.)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--pair-top-n", type=int, default=10)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if not 0 < args.min_coverage <= 1:
        parser.error("--min-coverage must be in (0, 1]")
    if args.min_samples < 2 or args.min_pumps < 2:
        parser.error("At least two samples per pump and two pumps per fault are needed")
    if args.top_n < 1 or args.pair_top_n < 1:
        parser.error("Plot and pair counts must be positive")
    if not all(0 <= v <= 100 for v in [args.max_sample_pct, args.max_pump_pct, args.min_fault_pct]):
        parser.error("Variation thresholds must be percentages between 0 and 100")
    files = args.files or discover_files(args.data_dir)
    x, origins, audit = load_features(files, args.keys_csv)
    print("Loaded {} runs, {} fault labels, {} features".format(
        len(x), x.index.get_level_values("fault_id").nunique(), x.shape[1]))
    parts = decompose(x)
    ranking = rank_features(x, origins, parts, args)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(out / "feature_variation_ranking.csv")
    ranking.loc[ranking.candidate].to_csv(out / "candidate_features.csv")
    selected = ranking.index[ranking.candidate].tolist()
    (out / "candidate_feature_names.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    audit.to_csv(out / "input_audit.csv", index=False)
    write_details(parts, x, out)
    pairwise_rankings(parts, ranking, args.pair_top_n).to_csv(out / "pairwise_top_features.csv", index=False)
    if not args.no_plots:
        make_plots(x, parts, ranking, out, args.top_n)
    summary = {"rows": len(x), "features": x.shape[1],
               "fault_labels": x.index.get_level_values("fault_id").nunique(),
               "fault_pump_groups": len(parts["pump_mean"]),
               "eligible_features": int(ranking.eligible.sum()),
               "constant_or_all_missing_features": int((ranking.unique_values <= 1).sum()),
               "candidate_features": len(selected),
               "settings": {k: str(v) if isinstance(v, Path) else v
                            for k, v in vars(args).items() if k != "files"},
               "input_files": [str(p) for p in files],
               "analysis_scope": "training IDs only" if args.keys_csv else "all supplied runs (exploratory)",
               "method": "Equal-fault, equal-pump empirical variance decomposition; ddof=0",
               "interpretation": "Variance shares, not accuracy, CV%, or probabilities"}
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Eligible: {}; candidates: {}; constants/all missing: {}".format(
        summary["eligible_features"], len(selected), summary["constant_or_all_missing_features"]))
    print(ranking.loc[ranking.eligible, ["sample_variation_pct", "pump_variation_pct",
                                       "fault_variation_pct"]].head(10).round(2).to_string())
    print("Saved results to " + str(out.resolve()))


if __name__ == "__main__":
    main()
