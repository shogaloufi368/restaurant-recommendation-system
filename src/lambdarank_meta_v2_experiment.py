import json
import math
import os
import time
from collections import Counter, defaultdict

import lightgbm as lgb
import numpy as np
import pandas as pd

import lambdarank_meta_experiment as meta
import exact_ordered_multi_patch_experiment as ordered_patch
import build_level3_role_basket_completion as lvl3
import build_weak_slice_assoc_reranker as weak
import investigate_ordered_context as ordered


BASE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(BASE, "candidate_lambdarank_meta_v2.csv")
REPORT_JSON = os.path.join(BASE, "lambdarank_meta_v2_report.json")
REPORT_MD = os.path.join(BASE, "lambdarank_meta_v2_report.md")

CHAMPION_CSV = "candidate_lambdarank_meta_oldbase.csv"
ORDERED_PATCH_CSV = "candidate_exact_ordered_multi_patch_oldbase.csv"
CHAMPION_VALIDATION = 0.770271044898964

REC_COLS = [f"rank_{i}" for i in range(1, 11)]
EXPECTED_COLS = ["instance_id"] + REC_COLS


def ndcg_at10(recs, target):
    if target not in recs[:10]:
        return 0.0
    return 1.0 / math.log2(recs.index(target) + 2)


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
    mean = lambda xs: float(np.mean(xs)) if xs else 0.0
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


def predict_rankings_with_scores(model, X, groups, matrix_meta):
    preds = model.predict(X)
    out = []
    offset = 0
    for g in groups:
        block = []
        row = matrix_meta[offset]["row"]
        weak_recs = matrix_meta[offset]["baseline_weak_recs"]
        for j in range(g):
            m = matrix_meta[offset + j]
            block.append((m["item"], float(preds[offset + j])))
        ranked_block = sorted(block, key=lambda x: (-x[1], x[0]))
        recs = [item for item, _ in ranked_block[:10]]
        margin = ranked_block[0][1] - ranked_block[1][1] if len(ranked_block) > 1 else 0.0
        score_map = {item: score for item, score in block}
        out.append({"row": row, "raw": recs, "weak": weak_recs, "margin": margin, "scores": score_map})
        offset += g
    return out


def guard_recs(raw, weak_recs, category_map, mode):
    if mode == "raw":
        return raw
    if mode.startswith("dessert_top"):
        n = int(mode.replace("dessert_top", ""))
        if any(category_map.get(item) == "Desserts" for item in weak_recs[:n]):
            return weak_recs
    if mode == "strong_exact_rank1" and weak_recs and category_map.get(weak_recs[0]) in {"Sauces", "Wraps"}:
        return weak_recs
    return raw


def ordered_patch_for_row(row, base_recs, ordered_stats, role_map, category_map, params):
    oprow = dict(row)
    oprow["ordered_context"] = tuple(row.get("ordered_context", row["context"]))
    return ordered_patch.patch_recs(oprow, base_recs, ordered_stats, role_map, category_map, params)


def build_variant(preds, category_map, ordered_stats, role_map, ordered_params, mode):
    groups = []
    for p in preds:
        row = p["row"]
        champion = guard_recs(p["raw"], p["weak"], category_map, "dessert_top10")
        recs = champion
        if mode == "champion":
            recs = champion
        elif mode == "dessert_top7":
            recs = guard_recs(p["raw"], p["weak"], category_map, "dessert_top7")
        elif mode == "dessert_top8":
            recs = guard_recs(p["raw"], p["weak"], category_map, "dessert_top8")
        elif mode == "dessert_top9":
            recs = guard_recs(p["raw"], p["weak"], category_map, "dessert_top9")
        elif mode == "raw_high_margin_keep_dessert":
            candidate = p["raw"] if p["margin"] >= 0.035 else champion
            recs = guard_recs(candidate, p["weak"], category_map, "dessert_top10")
        elif mode == "raw_very_high_margin_keep_dessert":
            candidate = p["raw"] if p["margin"] >= 0.070 else champion
            recs = guard_recs(candidate, p["weak"], category_map, "dessert_top10")
        elif mode == "ordered_patch_on_champion":
            recs = ordered_patch_for_row(row, champion, ordered_stats, role_map, category_map, ordered_params)
        elif mode == "ordered_patch_on_raw_guarded":
            guarded = guard_recs(p["raw"], p["weak"], category_map, "dessert_top10")
            recs = ordered_patch_for_row(row, guarded, ordered_stats, role_map, category_map, ordered_params)
        elif mode == "ordered_patch_protect_top1_bev_plate":
            patched = ordered_patch_for_row(row, champion, ordered_stats, role_map, category_map, ordered_params)
            ccat = category_map.get(champion[0], "unknown") if champion else "unknown"
            recs = champion if ccat in {"Beverages", "Plates"} else patched
        elif mode == "ordered_patch_protect_top3_bev_plate":
            patched = ordered_patch_for_row(row, champion, ordered_stats, role_map, category_map, ordered_params)
            top3_cats = {category_map.get(item, "unknown") for item in champion[:3]}
            recs = champion if top3_cats & {"Beverages", "Plates"} else patched
        elif mode == "ordered_patch_only_app_high_side":
            patched = ordered_patch_for_row(row, champion, ordered_stats, role_map, category_map, ordered_params)
            if patched[:1] != champion[:1]:
                ncat = category_map.get(patched[0], "unknown") if patched else "unknown"
                recs = patched if ncat in {"Appetizers", "Highlanders", "Sides"} else champion
        elif mode == "ordered_patch_no_demote_bev_plate_top3":
            patched = ordered_patch_for_row(row, champion, ordered_stats, role_map, category_map, ordered_params)
            if patched != champion:
                protected = {item for item in champion[:3] if category_map.get(item, "unknown") in {"Beverages", "Plates"}}
                recs = champion if any((item not in patched[:3]) for item in protected) else patched
        elif mode == "ordered_patch_multi_main_only_protect_top3":
            patched = ordered_patch_for_row(row, champion, ordered_stats, role_map, category_map, ordered_params)
            if row["segment"] != "multi-with-main":
                recs = champion
            else:
                protected = {item for item in champion[:3] if category_map.get(item, "unknown") in {"Beverages", "Plates", "Desserts"}}
                recs = champion if any((item not in patched[:3]) for item in protected) else patched
        elif mode == "weak_slice_selector":
            # If the champion predicts a non-weak category while weak-slice leads with a weak target category,
            # try the safer weak-slice ordering only in multi-with-main rows.
            if row["segment"] == "multi-with-main" and p["weak"]:
                wcat = category_map.get(p["weak"][0], "unknown")
                ccat = category_map.get(champion[0], "unknown") if champion else "unknown"
                if wcat in {"Beverages", "Plates", "Appetizers", "Desserts"} and ccat in {"Wraps", "Sauces"}:
                    recs = p["weak"]
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
        "top10_overlap": float(np.mean(overlaps)) if overlaps else 0.0,
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


def generate_test_csv(model, train, test, valid_items, category_map, role_map, ordered_params, chosen_mode):
    full_stats = lvl3.build_stats(train, valid_items, role_map)
    full_cat_stats = weak.category_stats(train, valid_items, category_map, role_map)
    full_seq = meta.build_seq_stats(train, valid_items, category_map)
    full_ordered = ordered.build_ordered_stats(train, valid_items, role_map, category_map)
    test_rows = meta.add_time_fields(meta.parse_test_rows(test, valid_items, role_map, category_map))
    # Preserve test item order for ordered patch.
    test_order_lookup = {}
    for row in test.itertuples(index=False):
        raw = []
        seen = set()
        for item in lvl3.parse_items(row.current_basket_items):
            if item in valid_items and item not in seen:
                raw.append(item)
                seen.add(item)
        test_order_lookup[str(row.instance_id)] = tuple(raw)
    for row in test_rows:
        row["ordered_context"] = test_order_lookup[row["instance_id"]]

    X, _, groups, matrix_meta, _ = meta.build_matrix(test_rows, full_stats, full_cat_stats, full_seq, valid_items, category_map, role_map, include_labels=False)
    preds = predict_rankings_with_scores(model, X, groups, matrix_meta)
    records = []
    for p in preds:
        recs = build_variant([p], category_map, full_ordered, role_map, ordered_params, chosen_mode)[0][0]
        records.append({"instance_id": p["row"]["instance_id"], **{f"rank_{i+1}": recs[i] for i in range(10)}})
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

    print("Building stats and validation matrices...", flush=True)
    build_stats = lvl3.build_stats(build_part, valid_items, role_map)
    build_cat = weak.category_stats(build_part, valid_items, category_map, role_map)
    build_seq = meta.build_seq_stats(build_part, valid_items, category_map)
    val_stats = lvl3.build_stats(train80, valid_items, role_map)
    val_cat = weak.category_stats(train80, valid_items, category_map, role_map)
    val_seq = meta.build_seq_stats(train80, valid_items, category_map)
    val_ordered = ordered.build_ordered_stats(train80, valid_items, role_map, category_map)

    meta_train_rows = meta.add_time_fields(lvl3.build_validation_rows(meta_train_part, valid_items, role_map, category_map))
    val_rows = meta.add_time_fields(lvl3.build_validation_rows(val_part, valid_items, role_map, category_map))
    ordered_rows = ordered.make_ordered_validation_rows(val_part, valid_items, role_map, category_map)
    order_lookup = {r["instance_id"]: r["ordered_context"] for r in ordered_rows}
    for row in val_rows:
        row["ordered_context"] = order_lookup.get(row["instance_id"], row["context"])

    X_tr, y_tr, g_tr, _, _ = meta.build_matrix(meta_train_rows, build_stats, build_cat, build_seq, valid_items, category_map, role_map)
    X_va, y_va, g_va, matrix_meta, _ = meta.build_matrix(val_rows, val_stats, val_cat, val_seq, valid_items, category_map, role_map)

    print("Training LambdaRank v2 feasibility model...", flush=True)
    model = meta.train_model(X_tr, y_tr, g_tr, 13)
    preds = predict_rankings_with_scores(model, X_va, g_va, matrix_meta)

    ordered_report = json.load(open(os.path.join(BASE, "exact_ordered_multi_patch_report.json"), encoding="utf-8"))
    ordered_params = ordered_report["best"]["params"]

    modes = [
        "champion",
        "dessert_top7",
        "dessert_top8",
        "dessert_top9",
        "raw_high_margin_keep_dessert",
        "raw_very_high_margin_keep_dessert",
        "ordered_patch_on_champion",
        "ordered_patch_on_raw_guarded",
        "ordered_patch_protect_top1_bev_plate",
        "ordered_patch_protect_top3_bev_plate",
        "ordered_patch_only_app_high_side",
        "ordered_patch_no_demote_bev_plate_top3",
        "ordered_patch_multi_main_only_protect_top3",
        "weak_slice_selector",
    ]
    champion_groups = build_variant(preds, category_map, val_ordered, role_map, ordered_params, "champion")
    champion_metrics = evaluate_groups(champion_groups)
    variants = []
    for mode in modes:
        groups = build_variant(preds, category_map, val_ordered, role_map, ordered_params, mode)
        metrics = evaluate_groups(groups)
        changes = compare_groups(groups, champion_groups)
        g = gate(metrics, champion_metrics)
        variants.append({"mode": mode, "metrics": metrics, "change_profile_vs_champion": changes, "gate": g})
        print(f"{mode}: ndcg={metrics['overall_ndcg']:.6f} delta={g['deltas']['overall']:.6f} pass={g['passes']} changes={changes}", flush=True)

    passing = [v for v in variants if v["gate"]["passes"]]
    best = max(passing, key=lambda x: x["metrics"]["overall_ndcg"]) if passing else max(variants, key=lambda x: x["metrics"]["overall_ndcg"])
    generated = None
    decision = "NO-GO"
    reason = "No v2 calibration beat champion validation by +0.0005 while preserving all category guards."

    if best["gate"]["passes"] and best["mode"] != "champion":
        aud = generate_test_csv(model, train, test, valid_items, category_map, role_map, ordered_params, best["mode"])
        generated = {"file": os.path.basename(OUT_CSV), "audit": aud}
        if audit_ok(aud):
            decision = "GO"
            reason = "Small LambdaRank v2 calibration beat champion gates and passed final audit."
        else:
            reason = "Validation gates passed, but generated CSV audit failed."

    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": time.time() - t0,
        "decision": decision,
        "reason": reason,
        "champion": {
            "file": CHAMPION_CSV,
            "leaderboard_score": 0.7755,
            "validation_ndcg_reference": CHAMPION_VALIDATION,
            "recomputed_validation": champion_metrics,
        },
        "feature_importance_context": json.load(open(os.path.join(BASE, "lambdarank_meta_experiment_report.json"), encoding="utf-8")).get("feature_importance", [])[:30],
        "guard_variants_original": json.load(open(os.path.join(BASE, "lambdarank_meta_experiment_report.json"), encoding="utf-8")).get("guard_variants", []),
        "variants": variants,
        "best": best,
        "generated": generated,
        "recommendation": CHAMPION_CSV if decision == "NO-GO" else os.path.basename(OUT_CSV),
    }
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(
            "# LambdaRank Meta V2 Experiment\n\n"
            f"Decision: **{decision}**\n\n{reason}\n\n"
            f"Champion validation: `{CHAMPION_VALIDATION:.6f}`\n"
            f"Best variant: `{best['mode']}` = `{best['metrics']['overall_ndcg']:.6f}`\n"
            f"Delta vs champion: `{best['gate']['deltas']['overall']:.6f}`\n"
            f"Generated: `{generated['file'] if generated else None}`\n"
        )
    print(json.dumps({
        "decision": decision,
        "reason": reason,
        "champion_validation": CHAMPION_VALIDATION,
        "best_mode": best["mode"],
        "best_validation": best["metrics"]["overall_ndcg"],
        "delta_vs_champion": best["gate"]["deltas"]["overall"],
        "generated": generated,
        "recommended_upload": payload["recommendation"],
        "report": os.path.basename(REPORT_JSON),
    }, indent=2))


if __name__ == "__main__":
    main()
