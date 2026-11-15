import json
import math
import os
import time
from collections import Counter, defaultdict

import lightgbm as lgb
import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except Exception:
    xgb = None

import lambdarank_meta_experiment as meta
import lambdarank_meta_v2_experiment as v2
import exact_ordered_multi_patch_experiment as ordered_patch
import build_level3_role_basket_completion as lvl3
import build_weak_slice_assoc_reranker as weak
import investigate_ordered_context as ordered


BASE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(BASE, "candidate_lambdarank_meta_v2_tuned.csv")
REPORT_JSON = os.path.join(BASE, "lambdarank_meta_toolkit_tuning_report.json")
REPORT_MD = os.path.join(BASE, "lambdarank_meta_toolkit_tuning_report.md")

CHAMPION_CSV = "candidate_lambdarank_meta_oldbase.csv"
CHAMPION_LB = 0.7755
CHAMPION_VALIDATION = 0.770271044898964

REC_COLS = [f"rank_{i}" for i in range(1, 11)]
EXPECTED_COLS = ["instance_id"] + REC_COLS

SEEDS = [13, 29, 42]


def ndcg_at10(recs, target):
    if target not in recs[:10]:
        return 0.0
    return 1.0 / math.log2(recs.index(target) + 2)


def mean(xs):
    return float(np.mean(xs)) if xs else 0.0


def score_groups(scores, groups, matrix_meta):
    out = []
    offset = 0
    for g in groups:
        row = matrix_meta[offset]["row"]
        weak_recs = matrix_meta[offset]["baseline_weak_recs"]
        block = []
        for j in range(g):
            m = matrix_meta[offset + j]
            block.append((m["item"], float(scores[offset + j])))
        ranked = sorted(block, key=lambda x: (-x[1], x[0]))
        raw = [item for item, _ in ranked[:10]]
        margin = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else 0.0
        out.append({"row": row, "raw": raw, "weak": weak_recs, "margin": margin})
        offset += g
    return out


def evaluate_groups(groups):
    ndcgs = []
    hits = {1: [], 3: [], 5: [], 10: []}
    rank1 = []
    by_segment = defaultdict(list)
    by_category = defaultdict(list)
    by_role = defaultdict(list)
    for recs, row in groups:
        target = row["hidden"]
        val = ndcg_at10(recs, target)
        ndcgs.append(val)
        rank1.append(int(recs and recs[0] == target))
        for k in hits:
            hits[k].append(int(target in recs[:k]))
        by_segment[row["segment"]].append(val)
        by_category[row["hidden_category"]].append(val)
        by_role[row["hidden_role"]].append(val)
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


def build_variant(preds, category_map, ordered_stats, role_map, ordered_params, mode):
    groups = []
    for p in preds:
        row = p["row"]
        champion_guarded = v2.guard_recs(p["raw"], p["weak"], category_map, "dessert_top10")
        recs = champion_guarded
        if mode == "dessert_top10":
            recs = champion_guarded
        elif mode == "ordered_patch_protect_top1_bev_plate":
            patched = v2.ordered_patch_for_row(row, champion_guarded, ordered_stats, role_map, category_map, ordered_params)
            ccat = category_map.get(champion_guarded[0], "unknown") if champion_guarded else "unknown"
            recs = champion_guarded if ccat in {"Beverages", "Plates"} else patched
        elif mode == "ordered_patch_protect_top3_bev_plate":
            patched = v2.ordered_patch_for_row(row, champion_guarded, ordered_stats, role_map, category_map, ordered_params)
            top3_cats = {category_map.get(item, "unknown") for item in champion_guarded[:3]}
            recs = champion_guarded if top3_cats & {"Beverages", "Plates"} else patched
        groups.append((recs, row))
    return groups


def compare_groups(new_groups, base_groups):
    changed = r1 = r2 = 0
    by_segment = Counter()
    overlaps = []
    for (new, row), (base, _) in zip(new_groups, base_groups):
        if new != base:
            changed += 1
            by_segment[row["segment"]] += 1
        if new[:1] != base[:1]:
            r1 += 1
        if new[1:2] != base[1:2]:
            r2 += 1
        overlaps.append(len(set(new) & set(base)) / 10)
    return {
        "changed_rows": changed,
        "rank_1_changes": r1,
        "rank_2_changes": r2,
        "top10_overlap": mean(overlaps),
        "changed_rows_by_segment": dict(by_segment),
    }


def gate(metrics, champion_metrics):
    deltas = {
        "overall": metrics["overall_ndcg"] - CHAMPION_VALIDATION,
        "sauces": metrics["ndcg_by_category"].get("Sauces", 0.0) - champion_metrics["ndcg_by_category"].get("Sauces", 0.0),
        "wraps": metrics["ndcg_by_category"].get("Wraps", 0.0) - champion_metrics["ndcg_by_category"].get("Wraps", 0.0),
        "desserts": metrics["ndcg_by_category"].get("Desserts", 0.0) - champion_metrics["ndcg_by_category"].get("Desserts", 0.0),
        "appetizers": metrics["ndcg_by_category"].get("Appetizers", 0.0) - champion_metrics["ndcg_by_category"].get("Appetizers", 0.0),
        "plates": metrics["ndcg_by_category"].get("Plates", 0.0) - champion_metrics["ndcg_by_category"].get("Plates", 0.0),
        "beverages": metrics["ndcg_by_category"].get("Beverages", 0.0) - champion_metrics["ndcg_by_category"].get("Beverages", 0.0),
        "multi_with_main": metrics["ndcg_by_segment"].get("multi-with-main", 0.0) - champion_metrics["ndcg_by_segment"].get("multi-with-main", 0.0),
    }
    checks = {
        "beats_champion_by_0_0005": deltas["overall"] >= 0.0005,
        "sauce_regression_guard": deltas["sauces"] >= -0.002,
        "wrap_regression_guard": deltas["wraps"] >= -0.002,
        "desserts_neutral_or_better": deltas["desserts"] >= -0.0001,
        "appetizers_neutral_or_better": deltas["appetizers"] >= -0.0001,
        "plates_neutral_or_better": deltas["plates"] >= -0.0001,
        "beverages_neutral_or_better": deltas["beverages"] >= -0.0001,
        "multi_with_main_neutral_or_better": deltas["multi_with_main"] >= -0.0001,
    }
    return {"deltas": deltas, "checks": checks, "passes": all(checks.values())}


def train_lgbm(X_tr, y_tr, g_tr, X_va, y_va, g_va, params, seed):
    model = lgb.LGBMRanker(
        objective=params.get("objective", "lambdarank"),
        metric="ndcg",
        ndcg_eval_at=[10],
        n_estimators=params.get("n_estimators", 260),
        learning_rate=params["learning_rate"],
        num_leaves=params["num_leaves"],
        min_child_samples=params["min_child_samples"],
        feature_fraction=params["feature_fraction"],
        bagging_fraction=params["bagging_fraction"],
        bagging_freq=1,
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    try:
        model.fit(
            X_tr,
            y_tr,
            group=g_tr,
            eval_set=[(X_va, y_va)],
            eval_group=[g_va],
            callbacks=[lgb.early_stopping(25, verbose=False)],
        )
    except Exception:
        model.fit(X_tr, y_tr, group=g_tr)
    return model


def train_xgb(X_tr, y_tr, g_tr, X_va, y_va, g_va, params, seed):
    if xgb is None:
        return None
    model = xgb.XGBRanker(
        objective="rank:ndcg",
        eval_metric="ndcg@10",
        n_estimators=params.get("n_estimators", 180),
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        min_child_weight=params["min_child_weight"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )
    try:
        model.fit(X_tr, y_tr, group=g_tr, eval_set=[(X_va, y_va)], eval_group=[g_va], verbose=False)
    except Exception:
        return None
    return model


def audit_submission(path, test_df, valid_items):
    df = pd.read_csv(path)
    current = {
        str(r.instance_id): set(lvl3.parse_items(r.current_basket_items))
        for r in test_df.itertuples(index=False)
    }
    dup = invalid = viol = semicolon = 0
    for row in df.itertuples(index=False):
        recs = [str(getattr(row, c)) for c in REC_COLS]
        dup += int(len(recs) != len(set(recs)))
        invalid += sum(1 for item in recs if item not in valid_items)
        viol += sum(1 for item in recs if item in current.get(str(row.instance_id), set()))
        semicolon += sum(1 for item in recs if ";" in item)
    return {
        "shape_ok": tuple(df.shape) == (14147, 11),
        "columns_exact": list(df.columns) == EXPECTED_COLS,
        "instance_id_matches_test": [str(x) for x in df["instance_id"]] == [str(x) for x in test_df["instance_id"]],
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_instance_id": int(df["instance_id"].duplicated().sum()),
        "duplicate_recommendation_rows": dup,
        "invalid_item_ids": invalid,
        "current_basket_violations": viol,
        "semicolon_cells": semicolon,
        "unnamed_columns": [c for c in df.columns if str(c).startswith("Unnamed")],
    }


def audit_ok(a):
    return (
        a and a["shape_ok"] and a["columns_exact"] and a["instance_id_matches_test"]
        and a["missing_values"] == 0 and a["duplicate_instance_id"] == 0
        and a["duplicate_recommendation_rows"] == 0 and a["invalid_item_ids"] == 0
        and a["current_basket_violations"] == 0 and a["semicolon_cells"] == 0
        and not a["unnamed_columns"]
    )


def generate_csv(best, models_by_key, train, test, valid_items, category_map, role_map, ordered_params):
    full_stats = lvl3.build_stats(train, valid_items, role_map)
    full_cat = weak.category_stats(train, valid_items, category_map, role_map)
    full_seq = meta.build_seq_stats(train, valid_items, category_map)
    full_ordered = ordered.build_ordered_stats(train, valid_items, role_map, category_map)
    test_rows = meta.add_time_fields(meta.parse_test_rows(test, valid_items, role_map, category_map))
    order_lookup = {}
    for row in test.itertuples(index=False):
        seen = set()
        items = []
        for item in lvl3.parse_items(row.current_basket_items):
            if item in valid_items and item not in seen:
                items.append(item)
                seen.add(item)
        order_lookup[str(row.instance_id)] = tuple(items)
    for row in test_rows:
        row["ordered_context"] = order_lookup[row["instance_id"]]

    X, _, groups, matrix_meta, _ = meta.build_matrix(test_rows, full_stats, full_cat, full_seq, valid_items, category_map, role_map, include_labels=False)

    if best["kind"] == "ensemble":
        total = None
        for key, weight in best["weights"]:
            pred = models_by_key[key].predict(X)
            total = weight * pred if total is None else total + weight * pred
        scores = total
    else:
        scores = models_by_key[best["model_key"]].predict(X)

    preds = score_groups(scores, groups, matrix_meta)
    groups_out = build_variant(preds, category_map, full_ordered, role_map, ordered_params, best["variant_mode"])
    records = []
    for recs, row in groups_out:
        records.append({"instance_id": row["instance_id"], **{f"rank_{i+1}": recs[i] for i in range(10)}})
    out = pd.DataFrame(records, columns=EXPECTED_COLS)
    out.to_csv(OUT_CSV, index=False)
    return audit_submission(OUT_CSV, test, valid_items)


def main():
    t0 = time.time()
    train = pd.read_csv(meta.TRAIN_BASKETS)
    test = pd.read_csv(meta.TEST_INSTANCES)
    menu = pd.read_csv(meta.MENU_ITEMS)
    train["timestamp"] = pd.to_datetime(train["timestamp"], errors="coerce")
    train = train.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    valid_items = set(menu["item_id"].astype(str))
    category_map, role_map = lvl3.make_menu_maps(menu)

    n = len(train)
    build_end = int(n * 0.60)
    meta_train_end = int(n * 0.80)
    build_part = train.iloc[:build_end].copy()
    meta_train_part = train.iloc[build_end:meta_train_end].copy()
    val_part = train.iloc[meta_train_end:].copy()
    train80 = train.iloc[:meta_train_end].copy()

    print("Building leakage-safe matrices once...", flush=True)
    build_stats = lvl3.build_stats(build_part, valid_items, role_map)
    build_cat = weak.category_stats(build_part, valid_items, category_map, role_map)
    build_seq = meta.build_seq_stats(build_part, valid_items, category_map)
    val_stats = lvl3.build_stats(train80, valid_items, role_map)
    val_cat = weak.category_stats(train80, valid_items, category_map, role_map)
    val_seq = meta.build_seq_stats(train80, valid_items, category_map)
    val_ordered = ordered.build_ordered_stats(train80, valid_items, role_map, category_map)

    train_rows = meta.add_time_fields(lvl3.build_validation_rows(meta_train_part, valid_items, role_map, category_map))
    val_rows = meta.add_time_fields(lvl3.build_validation_rows(val_part, valid_items, role_map, category_map))
    ordered_rows = ordered.make_ordered_validation_rows(val_part, valid_items, role_map, category_map)
    order_lookup = {r["instance_id"]: r["ordered_context"] for r in ordered_rows}
    for row in val_rows:
        row["ordered_context"] = order_lookup.get(row["instance_id"], row["context"])

    X_tr, y_tr, g_tr, _, feature_names = meta.build_matrix(train_rows, build_stats, build_cat, build_seq, valid_items, category_map, role_map)
    X_va, y_va, g_va, matrix_meta, _ = meta.build_matrix(val_rows, val_stats, val_cat, val_seq, valid_items, category_map, role_map)

    champion_report = json.load(open(os.path.join(BASE, "lambdarank_meta_experiment_report.json"), encoding="utf-8"))
    champion_metrics = champion_report["metrics"]["lambdarank_meta"]
    ordered_report = json.load(open(os.path.join(BASE, "exact_ordered_multi_patch_report.json"), encoding="utf-8"))
    ordered_params = ordered_report["best"]["params"]

    lgb_configs = [
        {"name": "baseline_045_31_30", "learning_rate": 0.045, "num_leaves": 31, "min_child_samples": 30, "feature_fraction": 0.85, "bagging_fraction": 0.85, "reg_alpha": 0.0, "reg_lambda": 1.0},
        {"name": "slow_030_31_50_reg", "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 50, "feature_fraction": 0.85, "bagging_fraction": 0.85, "reg_alpha": 0.1, "reg_lambda": 1.0},
        {"name": "small_030_15_30", "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 30, "feature_fraction": 0.95, "bagging_fraction": 0.85, "reg_alpha": 0.0, "reg_lambda": 0.5},
        {"name": "wide_020_47_50_reg", "learning_rate": 0.02, "num_leaves": 47, "min_child_samples": 50, "feature_fraction": 0.75, "bagging_fraction": 0.85, "reg_alpha": 0.1, "reg_lambda": 2.0},
        {"name": "fast_060_31_80", "learning_rate": 0.06, "num_leaves": 31, "min_child_samples": 80, "feature_fraction": 0.85, "bagging_fraction": 0.95, "reg_alpha": 0.5, "reg_lambda": 1.0},
        {"name": "wide_045_63_80_reg", "learning_rate": 0.045, "num_leaves": 63, "min_child_samples": 80, "feature_fraction": 0.75, "bagging_fraction": 0.75, "reg_alpha": 0.5, "reg_lambda": 2.0},
    ]
    xgb_configs = [
        {"name": "xgb_depth4_030", "learning_rate": 0.03, "max_depth": 4, "min_child_weight": 20, "subsample": 0.85, "colsample_bytree": 0.85, "reg_alpha": 0.1, "reg_lambda": 1.0},
        {"name": "xgb_depth5_045", "learning_rate": 0.045, "max_depth": 5, "min_child_weight": 30, "subsample": 0.85, "colsample_bytree": 0.75, "reg_alpha": 0.1, "reg_lambda": 2.0},
    ]

    models_by_key = {}
    pred_by_key = {}
    results = []
    variant_modes = ["dessert_top10", "ordered_patch_protect_top1_bev_plate"]

    for cfg in lgb_configs:
        for seed in SEEDS:
            key = f"lgbm::{cfg['name']}::seed{seed}"
            print(f"Training {key}", flush=True)
            model = train_lgbm(X_tr, y_tr, g_tr, X_va, y_va, g_va, cfg, seed)
            scores = model.predict(X_va)
            models_by_key[key] = model
            pred_by_key[key] = scores
            preds = score_groups(scores, g_va, matrix_meta)
            for mode in variant_modes:
                groups = build_variant(preds, category_map, val_ordered, role_map, ordered_params, mode)
                metrics = evaluate_groups(groups)
                base_groups = build_variant(score_groups(pred_by_key[key], g_va, matrix_meta), category_map, val_ordered, role_map, ordered_params, "dessert_top10")
                changes = compare_groups(groups, base_groups)
                g = gate(metrics, champion_metrics)
                results.append({"kind": "lgbm", "model_key": key, "variant_mode": mode, "params": cfg, "seed": seed, "metrics": metrics, "change_profile": changes, "gate": g})
                print(f"  {mode}: ndcg={metrics['overall_ndcg']:.6f} delta={g['deltas']['overall']:.6f} pass={g['passes']}", flush=True)

    xgb_available = xgb is not None
    if xgb_available:
        for cfg in xgb_configs:
            for seed in SEEDS:
                key = f"xgb::{cfg['name']}::seed{seed}"
                print(f"Training {key}", flush=True)
                model = train_xgb(X_tr, y_tr, g_tr, X_va, y_va, g_va, cfg, seed)
                if model is None:
                    results.append({"kind": "xgb", "model_key": key, "params": cfg, "seed": seed, "error": "fit_failed"})
                    continue
                scores = model.predict(X_va)
                models_by_key[key] = model
                pred_by_key[key] = scores
                preds = score_groups(scores, g_va, matrix_meta)
                for mode in variant_modes:
                    groups = build_variant(preds, category_map, val_ordered, role_map, ordered_params, mode)
                    metrics = evaluate_groups(groups)
                    g = gate(metrics, champion_metrics)
                    results.append({"kind": "xgb", "model_key": key, "variant_mode": mode, "params": cfg, "seed": seed, "metrics": metrics, "gate": g})
                    print(f"  {mode}: ndcg={metrics['overall_ndcg']:.6f} delta={g['deltas']['overall']:.6f} pass={g['passes']}", flush=True)

    # Simple ensembles: only between the strongest LGBM and strongest XGB if XGB exists.
    scored_results = [r for r in results if "metrics" in r]
    best_lgb = max([r for r in scored_results if r["kind"] == "lgbm"], key=lambda r: r["metrics"]["overall_ndcg"])
    xgb_scored = [r for r in scored_results if r["kind"] == "xgb"]
    if xgb_scored:
        best_xgb = max(xgb_scored, key=lambda r: r["metrics"]["overall_ndcg"])
        for lw, xw in [(0.7, 0.3), (0.6, 0.4), (0.8, 0.2)]:
            key = f"ensemble::{lw:.1f}lgb_{xw:.1f}xgb"
            scores = lw * pred_by_key[best_lgb["model_key"]] + xw * pred_by_key[best_xgb["model_key"]]
            preds = score_groups(scores, g_va, matrix_meta)
            for mode in variant_modes:
                groups = build_variant(preds, category_map, val_ordered, role_map, ordered_params, mode)
                metrics = evaluate_groups(groups)
                g = gate(metrics, champion_metrics)
                results.append({
                    "kind": "ensemble",
                    "model_key": key,
                    "weights": [(best_lgb["model_key"], lw), (best_xgb["model_key"], xw)],
                    "variant_mode": mode,
                    "metrics": metrics,
                    "gate": g,
                })
                print(f"  {key} {mode}: ndcg={metrics['overall_ndcg']:.6f} delta={g['deltas']['overall']:.6f} pass={g['passes']}", flush=True)

    passing = [r for r in results if r.get("gate", {}).get("passes")]
    best = max(passing, key=lambda r: r["metrics"]["overall_ndcg"]) if passing else max(scored_results, key=lambda r: r["metrics"]["overall_ndcg"])

    generated = None
    decision = "NO-GO"
    reason = "No tuned LightGBM/XGBoost/ensemble variant cleared champion hard gates."
    if best.get("gate", {}).get("passes"):
        audit = generate_csv(best, models_by_key, train, test, valid_items, category_map, role_map, ordered_params)
        generated = {"file": os.path.basename(OUT_CSV), "audit": audit}
        if audit_ok(audit):
            decision = "GO"
            reason = "Tuned toolkit variant beat champion validation gates and passed final audit."
        else:
            reason = "Validation gates passed, but generated CSV audit failed."

    importance = []
    if best["kind"] in {"lgbm", "xgb"} and best["model_key"] in models_by_key:
        model = models_by_key[best["model_key"]]
        imp = getattr(model, "feature_importances_", None)
        if imp is not None:
            importance = sorted(zip(feature_names, [int(x) for x in imp]), key=lambda x: -x[1])[:30]

    seed_stability = defaultdict(list)
    for r in scored_results:
        if r["kind"] in {"lgbm", "xgb"} and r.get("variant_mode") == best.get("variant_mode"):
            base_name = r["params"]["name"]
            seed_stability[f"{r['kind']}::{base_name}"].append(r["metrics"]["overall_ndcg"])

    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": time.time() - t0,
        "toolkit_used": True,
        "xgboost_available": xgb_available,
        "decision": decision,
        "reason": reason,
        "champion": {"file": CHAMPION_CSV, "leaderboard_score": CHAMPION_LB, "validation_ndcg": CHAMPION_VALIDATION},
        "split": {
            "build_stats_rows": len(build_part),
            "meta_train_rows": len(meta_train_part),
            "validation_rows": len(val_part),
            "leakage_check": "first 60% stats, next 20% ranker train, last 20% validation; validation baskets excluded; sequential/user stats use earlier history only",
        },
        "lgbm_configs": lgb_configs,
        "xgb_configs": xgb_configs,
        "results": results,
        "best": best,
        "feature_importance": importance,
        "seed_stability": {k: {"scores": v, "mean": mean(v), "std": float(np.std(v)) if v else 0.0} for k, v in seed_stability.items()},
        "generated": generated,
        "recommended_upload": os.path.basename(OUT_CSV) if decision == "GO" else CHAMPION_CSV,
    }
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(
            "# LambdaRank Meta Toolkit Tuning\n\n"
            f"Decision: **{decision}**\n\n{reason}\n\n"
            f"Best kind: `{best.get('kind')}`\n"
            f"Best model: `{best.get('model_key')}`\n"
            f"Best variant: `{best.get('variant_mode')}`\n"
            f"Validation NDCG: `{best['metrics']['overall_ndcg']:.6f}`\n"
            f"Delta vs champion: `{best['gate']['deltas']['overall']:.6f}`\n"
            f"Generated: `{generated['file'] if generated else None}`\n"
        )
    print(json.dumps({
        "decision": decision,
        "reason": reason,
        "best_kind": best.get("kind"),
        "best_model": best.get("model_key"),
        "best_variant": best.get("variant_mode"),
        "validation_ndcg": best["metrics"]["overall_ndcg"],
        "delta_vs_champion": best["gate"]["deltas"]["overall"],
        "generated": generated,
        "recommended_upload": payload["recommended_upload"],
        "report": os.path.basename(REPORT_JSON),
    }, indent=2))


if __name__ == "__main__":
    main()
