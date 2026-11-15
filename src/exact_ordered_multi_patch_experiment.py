import json
import math
import os
import time
from collections import Counter, defaultdict

import pandas as pd

import build_level3_role_basket_completion as lvl3
import build_weak_slice_assoc_reranker as weak
import investigate_ordered_context as ordered


BASE = os.path.dirname(os.path.abspath(__file__))
TRAIN_BASKETS = os.path.join(BASE, "train_baskets.csv")
TEST_INSTANCES = os.path.join(BASE, "test_instances.csv")
MENU_ITEMS = os.path.join(BASE, "menu_items.csv")

OUT_CSV = "candidate_exact_ordered_multi_patch_oldbase.csv"
REPORT_JSON = "exact_ordered_multi_patch_report.json"
REPORT_MD = "exact_ordered_multi_patch_report.md"

REC_COLS = [f"rank_{i}" for i in range(1, 11)]
EXPECTED_COLS = ["instance_id"] + REC_COLS

EXACT_BASE_FILE = "exact_context_blend_oldbase.csv"

PARAM_GRID = [
    {"name": "main_sauce_strict_20_008", "min_sup": 20, "min_gain": 0.08, "exact": 0.35, "first": 0.02, "last": 0.06, "pair": 0.12, "role": 0.05, "main_sauce_only": True, "topk_scope": 5},
    {"name": "main_sauce_strict_12_006", "min_sup": 12, "min_gain": 0.06, "exact": 0.32, "first": 0.02, "last": 0.06, "pair": 0.14, "role": 0.06, "main_sauce_only": True, "topk_scope": 5},
    {"name": "main_sauce_top10_12_005", "min_sup": 12, "min_gain": 0.05, "exact": 0.30, "first": 0.02, "last": 0.06, "pair": 0.15, "role": 0.06, "main_sauce_only": True, "topk_scope": 10},
    {"name": "multi_strict_25_010", "min_sup": 25, "min_gain": 0.10, "exact": 0.40, "first": 0.02, "last": 0.05, "pair": 0.10, "role": 0.03, "main_sauce_only": False, "topk_scope": 5},
    {"name": "multi_strict_18_008", "min_sup": 18, "min_gain": 0.08, "exact": 0.36, "first": 0.02, "last": 0.05, "pair": 0.12, "role": 0.04, "main_sauce_only": False, "topk_scope": 5},
    {"name": "multi_top10_18_007", "min_sup": 18, "min_gain": 0.07, "exact": 0.34, "first": 0.02, "last": 0.05, "pair": 0.14, "role": 0.05, "main_sauce_only": False, "topk_scope": 10},
]

COMPLEMENT_CATEGORIES = {"Beverages", "Sides", "Salads", "Desserts", "Appetizers", "Plates", "Highlanders"}
STRONG_CATEGORIES = {"Sauces", "Wraps"}


def ndcg_at10(recs, target):
    if target not in recs[:10]:
        return 0.0
    return 1.0 / math.log2(recs.index(target) + 2)


def metric_summary(rows, predictor):
    ndcgs = []
    hits = {1: [], 3: [], 5: [], 10: []}
    rank1 = []
    by_segment = defaultdict(list)
    by_category = defaultdict(list)
    by_role = defaultdict(list)
    for row in rows:
        recs = predictor(row)
        target = row["hidden"]
        val = ndcg_at10(recs, target)
        ndcgs.append(val)
        rank1.append(int(recs and recs[0] == target))
        for k in hits:
            hits[k].append(int(target in recs[:k]))
        by_segment[row["segment"]].append(val)
        by_category[row["hidden_category"]].append(val)
        by_role[row["hidden_role"]].append(val)
    mean = lambda xs: float(sum(xs) / len(xs)) if xs else 0.0
    return {
        "overall_ndcg": mean(ndcgs),
        "hit_at": {str(k): mean(v) for k, v in hits.items()},
        "rank1_accuracy": mean(rank1),
        "ndcg_by_segment": {k: mean(v) for k, v in by_segment.items()},
        "ndcg_by_category": {k: mean(v) for k, v in by_category.items()},
        "ndcg_by_role": {k: mean(v) for k, v in by_role.items()},
        "counts_by_segment": {k: len(v) for k, v in by_segment.items()},
        "counts_by_category": {k: len(v) for k, v in by_category.items()},
    }


def context_roles(row, role_map):
    return [role_map.get(item, "dessert_addon") for item in row["ordered_context"]]


def exact_base_recs(row, stats, valid_items):
    return lvl3.exact_context_recs(row["context"], stats, valid_items)


def patch_recs(row, base_recs, ordered_stats, role_map, category_map, params):
    # The defining guard for this experiment: no empty/single changes.
    if len(row["ordered_context"]) < 2:
        return base_recs

    roles = context_roles(row, role_map)
    if params["main_sauce_only"] and not ("main" in roles and "sauce" in roles):
        return base_recs

    scope = base_recs[: params["topk_scope"]]
    old_top = base_recs[0] if base_recs else None
    old_score, old_support = ordered.ordered_score(row, old_top, ordered_stats, role_map, category_map, params) if old_top else (0.0, 0)

    scored = []
    for item in scope:
        cat = category_map.get(item, "unknown")
        score, support = ordered.ordered_score(row, item, ordered_stats, role_map, category_map, params)
        complement = cat in COMPLEMENT_CATEGORIES
        scored.append((item, score, support, complement))

    scored.sort(key=lambda x: (-x[1], -x[2], -int(x[3]), x[0]))
    if not scored:
        return base_recs

    new_top, new_score, new_support, complement = scored[0]
    if new_top == old_top:
        return base_recs
    if new_support < params["min_sup"]:
        return base_recs
    if new_score - old_score < params["min_gain"]:
        return base_recs
    if category_map.get(old_top, "unknown") in STRONG_CATEGORIES and category_map.get(new_top, "unknown") not in STRONG_CATEGORIES:
        if new_score - old_score < params["min_gain"] * 1.75:
            return base_recs
    if not complement and category_map.get(new_top, "unknown") not in STRONG_CATEGORIES:
        return base_recs

    new_order = [item for item, *_ in scored] + [item for item in base_recs if item not in {x[0] for x in scored}]
    return new_order[:10]


def evaluate_change(rows, base_predictor, cand_predictor):
    changed = r1 = r2 = 0
    by_segment = Counter()
    for row in rows:
        base = base_predictor(row)
        cand = cand_predictor(row)
        if base != cand:
            changed += 1
            by_segment[row["segment"]] += 1
        if base[:1] != cand[:1]:
            r1 += 1
        if len(base) > 1 and len(cand) > 1 and base[1] != cand[1]:
            r2 += 1
    return {"changed_rows": changed, "rank_1_changes": r1, "rank_2_changes": r2, "changed_rows_by_segment": dict(by_segment)}


def gate(base_metrics, cand_metrics, change_profile):
    deltas = {
        "overall": cand_metrics["overall_ndcg"] - base_metrics["overall_ndcg"],
        "sauces": cand_metrics["ndcg_by_category"].get("Sauces", 0.0) - base_metrics["ndcg_by_category"].get("Sauces", 0.0),
        "wraps": cand_metrics["ndcg_by_category"].get("Wraps", 0.0) - base_metrics["ndcg_by_category"].get("Wraps", 0.0),
        "multi_with_main": cand_metrics["ndcg_by_segment"].get("multi-with-main", 0.0) - base_metrics["ndcg_by_segment"].get("multi-with-main", 0.0),
        "multi_no_main": cand_metrics["ndcg_by_segment"].get("multi-no-main", 0.0) - base_metrics["ndcg_by_segment"].get("multi-no-main", 0.0),
    }
    checks = {
        "beats_exact_by_clear_margin": deltas["overall"] >= 0.0005,
        "sauce_regression_guard": deltas["sauces"] >= -0.002,
        "wrap_regression_guard": deltas["wraps"] >= -0.002,
        "multi_with_main_improves": deltas["multi_with_main"] >= 0.0,
        "multi_no_main_neutral": deltas["multi_no_main"] >= -0.0001,
        "no_empty_changes": change_profile["changed_rows_by_segment"].get("empty", 0) == 0,
        "no_single_changes": (
            change_profile["changed_rows_by_segment"].get("single-visible-main", 0) == 0
            and change_profile["changed_rows_by_segment"].get("single-visible-addon", 0) == 0
            and change_profile["changed_rows_by_segment"].get("single", 0) == 0
        ),
    }
    return {"deltas": deltas, "checks": checks, "passes": all(checks.values())}


def parse_test_ordered(row, valid_items, role_map):
    raw = [x for x in lvl3.parse_items(row.current_basket_items) if x in valid_items]
    seen = set()
    ordered_ctx = []
    for item in raw:
        if item not in seen:
            ordered_ctx.append(item)
            seen.add(item)
    context = tuple(sorted(ordered_ctx))
    return {
        "instance_id": str(row.instance_id),
        "ordered_context": tuple(ordered_ctx),
        "context": context,
        "segment": lvl3.segment_for_context(context, role_map),
        "hidden": None,
        "hidden_category": None,
        "hidden_role": None,
        "meal_period": str(row.meal_period),
        "user_id": str(row.user_id),
    }


def audit(df, test, valid_items):
    current = {
        str(r.instance_id): set(lvl3.parse_items(r.current_basket_items))
        for r in test.itertuples(index=False)
    }
    dup = invalid = violations = semicolon = 0
    for row in df.itertuples(index=False):
        recs = [str(getattr(row, c)) for c in REC_COLS]
        dup += int(len(recs) != len(set(recs)))
        invalid += sum(1 for item in recs if item not in valid_items)
        violations += sum(1 for item in recs if item in current.get(str(row.instance_id), set()))
        semicolon += sum(1 for item in recs if ";" in item)
    return {
        "shape_ok": tuple(df.shape) == (14147, 11),
        "columns_exact": list(df.columns) == EXPECTED_COLS,
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_instance_id": int(df["instance_id"].duplicated().sum()),
        "duplicate_recommendation_rows": dup,
        "invalid_item_ids": invalid,
        "current_basket_violations": violations,
        "instance_id_matches_test": [str(x) for x in df["instance_id"]] == [str(x) for x in test["instance_id"]],
        "unnamed_columns": [c for c in df.columns if str(c).startswith("Unnamed")],
        "semicolon_cells": semicolon,
    }


def generate_candidate(test, base_file, ordered_stats, valid_items, role_map, category_map, params):
    base = pd.read_csv(os.path.join(BASE, base_file))
    out = base.copy()
    idx_map = {str(iid): idx for idx, iid in out["instance_id"].astype(str).items()}
    changed = r1 = r2 = 0
    by_segment = Counter()
    for test_row in test.itertuples(index=False):
        iid = str(test_row.instance_id)
        idx = idx_map[iid]
        base_recs = [str(out.at[idx, c]) for c in REC_COLS]
        row = parse_test_ordered(test_row, valid_items, role_map)
        recs = patch_recs(row, base_recs, ordered_stats, role_map, category_map, params)
        if recs != base_recs:
            changed += 1
            by_segment["empty" if len(row["context"]) == 0 else ("single" if len(row["context"]) == 1 else "multi")] += 1
        if recs[:1] != base_recs[:1]:
            r1 += 1
        if recs[1:2] != base_recs[1:2]:
            r2 += 1
        for col, item in zip(REC_COLS, recs):
            out.at[idx, col] = item
    out.to_csv(os.path.join(BASE, OUT_CSV), index=False)
    return out, {"changed_rows": changed, "rank_1_changes": r1, "rank_2_changes": r2, "changed_rows_by_segment": dict(by_segment)}


def main():
    t0 = time.time()
    train = pd.read_csv(TRAIN_BASKETS)
    test = pd.read_csv(TEST_INSTANCES)
    menu = pd.read_csv(MENU_ITEMS)
    valid_items = set(menu["item_id"].astype(str))
    category_map, role_map = lvl3.make_menu_maps(menu)
    train_part, val_part = lvl3.split_train_validation(train)
    stats = lvl3.build_stats(train_part, valid_items, role_map)
    ordered_stats = ordered.build_ordered_stats(train_part, valid_items, role_map, category_map)
    cat_stats = weak.category_stats(train_part, valid_items, category_map, role_map)
    val_rows = ordered.make_ordered_validation_rows(val_part, valid_items, role_map, category_map)

    exact_pred = lambda r: exact_base_recs(r, stats, valid_items)
    weak_pred = lambda r: weak.weak_slice_recs(r, stats, cat_stats, valid_items, category_map, role_map, {"name": "graph_cat_small", "graph": 0.10, "cat": 0.10, "user": 0.0, "meal": 0.0, "subset": 0.0}, lvl3.assoc_recs(r["context"], stats, valid_items))
    full_proxy = lambda r: lvl3.assoc_recs(r["context"], stats, valid_items)

    exact_metrics = metric_summary(val_rows, exact_pred)
    weak_metrics = metric_summary(val_rows, weak_pred)
    full_metrics = metric_summary(val_rows, full_proxy)

    sweep = []
    for params in PARAM_GRID:
        cand_pred = lambda r, p=params: patch_recs(r, exact_pred(r), ordered_stats, role_map, category_map, p)
        metrics = metric_summary(val_rows, cand_pred)
        changes = evaluate_change(val_rows, exact_pred, cand_pred)
        g = gate(exact_metrics, metrics, changes)
        row = {"params": params, "metrics": metrics, "change_profile": changes, "gate": g}
        sweep.append(row)
        print(f"{params['name']}: ndcg={metrics['overall_ndcg']:.6f} delta={g['deltas']['overall']:.6f} pass={g['passes']} changes={changes}", flush=True)

    passing = [x for x in sweep if x["gate"]["passes"]]
    best = max(passing, key=lambda x: x["metrics"]["overall_ndcg"]) if passing else max(sweep, key=lambda x: x["metrics"]["overall_ndcg"])

    generated = None
    decision = "NO-GO"
    reason = "Ordered selective multi-only patch did not clear validation gates."
    if best["gate"]["passes"]:
        full_ordered_stats = ordered.build_ordered_stats(train, valid_items, role_map, category_map)
        df, gen_profile = generate_candidate(test, EXACT_BASE_FILE, full_ordered_stats, valid_items, role_map, category_map, best["params"])
        aud = audit(df, test, valid_items)
        audit_ok = (
            aud["shape_ok"] and aud["columns_exact"] and aud["missing_values"] == 0
            and aud["duplicate_instance_id"] == 0 and aud["duplicate_recommendation_rows"] == 0
            and aud["invalid_item_ids"] == 0 and aud["current_basket_violations"] == 0
            and aud["instance_id_matches_test"] and not aud["unnamed_columns"] and aud["semicolon_cells"] == 0
        )
        generated = {"file": OUT_CSV, "change_profile_vs_exact_base": gen_profile, "audit": aud}
        if audit_ok:
            decision = "GO"
            reason = "Ordered selective multi-only patch beat exact-context validation and passed final audit."
        else:
            reason = "Validation gates passed, but final audit failed."

    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "decision": decision,
        "reason": reason,
        "elapsed_seconds": time.time() - t0,
        "base_file": EXACT_BASE_FILE,
        "split": {
            "train_rows": len(train_part),
            "validation_rows": len(val_part),
            "train_max_timestamp": str(pd.to_datetime(train_part["timestamp"]).max()),
            "validation_min_timestamp": str(pd.to_datetime(val_part["timestamp"]).min()),
            "leakage_check": "ordered evidence and exact-context stats built only from earlier training split; validation baskets excluded",
        },
        "metrics": {
            "exact_context_blend_proxy": exact_metrics,
            "weak_slice_proxy": weak_metrics,
            "assoc_full_proxy": full_metrics,
            "ordered_multi_patch": best["metrics"],
        },
        "best": best,
        "sweep": sweep,
        "generated": generated,
        "recommendation_if_no_go": "candidate_lambdarank_meta_oldbase.csv",
    }
    with open(os.path.join(BASE, REPORT_JSON), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    lines = [
        "# Exact Ordered Multi Patch Experiment",
        "",
        f"Decision: **{decision}**",
        "",
        reason,
        "",
        f"Base file: `{EXACT_BASE_FILE}`",
        "",
        "## Validation",
        f"- Exact-context proxy: `{exact_metrics['overall_ndcg']:.6f}`",
        f"- Weak-slice proxy: `{weak_metrics['overall_ndcg']:.6f}`",
        f"- Assoc/full proxy: `{full_metrics['overall_ndcg']:.6f}`",
        f"- Best ordered patch: `{best['metrics']['overall_ndcg']:.6f}`",
        f"- Delta vs exact: `{best['gate']['deltas']['overall']:.6f}`",
        "",
        "## Best Params",
        "```json",
        json.dumps(best["params"], indent=2),
        "```",
        "",
        "## Gate",
        "```json",
        json.dumps(best["gate"], indent=2),
        "```",
        "",
        f"Generated: `{generated['file'] if generated else None}`",
    ]
    with open(os.path.join(BASE, REPORT_MD), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(json.dumps({
        "decision": decision,
        "reason": reason,
        "exact_ndcg": exact_metrics["overall_ndcg"],
        "weak_ndcg": weak_metrics["overall_ndcg"],
        "lambda_reference": "candidate_lambdarank_meta_oldbase.csv",
        "best_patch_ndcg": best["metrics"]["overall_ndcg"],
        "delta_vs_exact": best["gate"]["deltas"]["overall"],
        "best_params": best["params"],
        "generated": generated,
        "report": REPORT_JSON,
    }, indent=2))


if __name__ == "__main__":
    main()
