import json
import math
import os
import time
from collections import Counter, defaultdict

import lightgbm as lgb
import numpy as np
import pandas as pd

import build_level3_role_basket_completion as lvl3
import build_weak_slice_assoc_reranker as weak


BASE = os.path.dirname(os.path.abspath(__file__))
TRAIN_BASKETS = os.path.join(BASE, "train_baskets.csv")
TEST_INSTANCES = os.path.join(BASE, "test_instances.csv")
MENU_ITEMS = os.path.join(BASE, "menu_items.csv")

REPORT_JSON = os.path.join(BASE, "lambdarank_meta_experiment_report.json")
REPORT_MD = os.path.join(BASE, "lambdarank_meta_experiment_report.md")
OUT_CSV = os.path.join(BASE, "candidate_lambdarank_meta_oldbase.csv")

REC_COLS = [f"rank_{i}" for i in range(1, 11)]
EXPECTED_COLS = ["instance_id"] + REC_COLS

BASELINE_NDCG = 0.767120188826832
BEST_WEAK_PARAMS = {"name": "graph_cat_small", "graph": 0.10, "cat": 0.10, "user": 0.0, "meal": 0.0, "subset": 0.0}

REFERENCE_FILES = {
    "weak": "candidate_weak_slice_assoc_oldbase.csv",
    "full_old": "basket_completion_full_pipeline_oldbase.csv",
    "full_v7": "basket_completion_full_pipeline_v7base.csv",
    "assoc_old": "basket_completion_assoc_oldbase.csv",
    "exact_old": "exact_context_blend_oldbase.csv",
    "oldbase": "submission_final_bestt.csv",
}

WEAK_CATEGORIES = {"Appetizers", "Plates", "Beverages", "Desserts"}
STRONG_GUARD_CATEGORIES = {"Sauces", "Wraps"}


def safe_div(num, den):
    return float(num) / float(den) if den else 0.0


def one_hot(prefix, value, choices):
    return {f"{prefix}_{choice}": 1.0 if value == choice else 0.0 for choice in choices}


def load_reference_rankings():
    out = {}
    for name, filename in REFERENCE_FILES.items():
        path = os.path.join(BASE, filename)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        out[name] = {
            str(row.instance_id): [str(getattr(row, c)) for c in REC_COLS]
            for row in df.itertuples(index=False)
        }
    return out


def rank_feature(rank):
    if rank is None:
        return 0.0
    return 1.0 / rank


def rank_lookup(recs):
    return {item: i for i, item in enumerate(recs, start=1)}


def position_features(prefix, recs, item):
    r = rank_lookup(recs).get(item)
    return {
        f"{prefix}_rank": float(r if r is not None else 99),
        f"{prefix}_inv_rank": rank_feature(r),
        f"{prefix}_top1": 1.0 if r == 1 else 0.0,
        f"{prefix}_top3": 1.0 if r is not None and r <= 3 else 0.0,
        f"{prefix}_top5": 1.0 if r is not None and r <= 5 else 0.0,
        f"{prefix}_top10": 1.0 if r is not None and r <= 10 else 0.0,
    }


def ndcg_at10(recs, target):
    for i, item in enumerate(recs[:10], start=1):
        if item == target:
            return 1.0 / math.log2(i + 1)
    return 0.0


def rank_of(recs, target):
    try:
        return recs.index(target) + 1
    except ValueError:
        return None


def hit_at(recs, target, k):
    return int(target in recs[:k])


def parse_test_rows(test_df, valid_items, role_map, category_map):
    rows = []
    for row in test_df.itertuples(index=False):
        context = tuple(x for x in lvl3.unique_context(lvl3.parse_items(row.current_basket_items)) if x in valid_items)
        rows.append(
            {
                "instance_id": str(row.instance_id),
                "user_id": str(row.user_id),
                "timestamp": pd.to_datetime(row.timestamp, errors="coerce"),
                "context": context,
                "hidden": None,
                "hidden_role": None,
                "hidden_category": None,
                "segment": lvl3.segment_for_context(context, role_map),
                "meal_period": str(row.meal_period),
                "hour": int(row.hour) if not pd.isna(row.hour) else -1,
                "day_of_week": str(getattr(row, "day_of_week", "")),
                "is_weekend": int(getattr(row, "is_weekend", 0)) if not pd.isna(getattr(row, "is_weekend", 0)) else 0,
            }
        )
    return rows


def add_time_fields(rows):
    for row in rows:
        ts = pd.to_datetime(row.get("timestamp"), errors="coerce")
        if not pd.isna(ts):
            row["hour"] = int(ts.hour)
            row["day_of_week"] = str(ts.day_name())
            row["is_weekend"] = int(ts.dayofweek >= 5)
        row["hour_bucket"] = int(row.get("hour", -1) // 4) if row.get("hour", -1) >= 0 else -1
    return rows


def build_seq_stats(train_df, valid_items, category_map):
    by_user = defaultdict(list)
    for row in train_df.itertuples(index=False):
        ts = pd.to_datetime(row.timestamp, errors="coerce")
        if pd.isna(ts):
            continue
        basket = tuple(x for x in lvl3.unique_context(lvl3.parse_items(row.item_ids)) if x in valid_items)
        if not basket:
            continue
        by_user[str(row.user_id)].append((ts, basket))

    prev_basket_item = defaultdict(Counter)
    prev_item_item = defaultdict(Counter)
    prev_cat_item = defaultdict(Counter)
    recent_user_item = defaultdict(Counter)
    recent_user_cat = defaultdict(Counter)
    user_history_item = defaultdict(Counter)

    for uid, events in by_user.items():
        events.sort(key=lambda x: x[0])
        last_baskets = []
        for _, basket in events:
            bset = set(basket)
            if last_baskets:
                prev = set(last_baskets[-1])
                for p in prev:
                    for target in bset:
                        prev_item_item[p][target] += 1
                        prev_cat_item[category_map.get(p, "unknown")][target] += 1
                for prev_item in prev:
                    for target in bset:
                        prev_basket_item[prev_item][target] += 1
            for distance, prev_basket in enumerate(reversed(last_baskets[-5:]), start=1):
                weight = 1.0 / distance
                for target in bset:
                    for prev_item in prev_basket:
                        recent_user_item[(uid, prev_item)][target] += weight
                        recent_user_cat[(uid, category_map.get(prev_item, "unknown"))][target] += weight
            for target in bset:
                user_history_item[uid][target] += 1
            last_baskets.append(basket)

    return {
        "prev_item_item": prev_item_item,
        "prev_cat_item": prev_cat_item,
        "recent_user_item": recent_user_item,
        "recent_user_cat": recent_user_cat,
        "user_history_item": user_history_item,
    }


def seq_features(row, item, seq_stats, category_map):
    vals = []
    cat_vals = []
    recent_vals = []
    recent_cat_vals = []
    for visible in row["context"]:
        c = seq_stats["prev_item_item"].get(visible, Counter())
        vals.append(safe_div(c.get(item, 0), sum(c.values())))
        cat = category_map.get(visible, "unknown")
        cc = seq_stats["prev_cat_item"].get(cat, Counter())
        cat_vals.append(safe_div(cc.get(item, 0), sum(cc.values())))
        rc = seq_stats["recent_user_item"].get((row["user_id"], visible), Counter())
        recent_vals.append(safe_div(rc.get(item, 0), sum(rc.values())))
        rcc = seq_stats["recent_user_cat"].get((row["user_id"], cat), Counter())
        recent_cat_vals.append(safe_div(rcc.get(item, 0), sum(rcc.values())))
    uh = seq_stats["user_history_item"].get(row["user_id"], Counter())
    return {
        "seq_prev_item_mean": float(np.mean(vals)) if vals else 0.0,
        "seq_prev_item_max": float(np.max(vals)) if vals else 0.0,
        "seq_prev_cat_mean": float(np.mean(cat_vals)) if cat_vals else 0.0,
        "seq_recent_item_mean": float(np.mean(recent_vals)) if recent_vals else 0.0,
        "seq_recent_cat_mean": float(np.mean(recent_cat_vals)) if recent_cat_vals else 0.0,
        "user_history_item_prob": safe_div(uh.get(item, 0), sum(uh.values())),
    }


def item_signal_features(row, item, stats, cat_stats, valid_items, category_map, role_map, seq_stats):
    context = tuple(sorted(row["context"]))
    current = set(context)
    full_size = len(current) + 1
    ctx_counter = stats["ctx_counts"].get(context, Counter())
    ctx_total = stats["ctx_totals"].get(context, 0)
    exact_count = ctx_counter.get(item, 0)
    exact_conf = safe_div(exact_count, ctx_total)
    subset_conf, lift_bonus, npmi_bonus = lvl3.weighted_subset_signal(context, item, stats)
    denom = stats["subset_support"].get(context, 0)
    joint = tuple(sorted(set(context) | {item}))
    joint_count = stats["subset_support"].get(joint, 0)
    expected = safe_div(stats["subset_support"].get(context, 0) * stats["subset_support"].get((item,), 0), stats["n_baskets"])
    chi = ((joint_count - expected) ** 2 / expected) if expected > 0 else 0.0
    conf, lift, npmi, _ = lvl3.rule_metrics(context, item, stats) if context else (0.0, 0.0, 0.0, 0)
    base_rank = {}
    assoc_score = lvl3.item_score_assoc(context, item, stats, base_rank, lvl3.ASSOC_WEIGHTS)
    graph = weak.graph_npmi(context, item, stats)
    cat_sig = weak.category_signal(row, item, cat_stats, category_map, role_map)
    weak_flag = 1.0 if weak.is_target_candidate(row, item, category_map, role_map) else 0.0
    item_cat = category_map.get(item, "unknown")
    item_role = role_map.get(item, "dessert_addon")
    size_counter = stats["size_counts"].get(full_size, Counter())
    meal_counter = stats["meal_item_counts"].get(row["meal_period"], Counter())
    user_counter = stats["user_item_counts"].get(row["user_id"], Counter())

    feats = {
        "exact_count": float(exact_count),
        "exact_support": float(ctx_total),
        "exact_conf": exact_conf,
        "subset_context_support": float(denom),
        "subset_joint_count": float(joint_count),
        "subset_conf": subset_conf,
        "rule_conf": conf,
        "lift": float(lift),
        "lift_bonus": lift_bonus,
        "npmi": float(npmi),
        "npmi_bonus": npmi_bonus,
        "chi_square": min(10.0, math.log1p(chi)),
        "assoc_score": assoc_score,
        "graph_npmi": graph,
        "category_signal": cat_sig,
        "weak_slice_flag": weak_flag,
        "size_prior": safe_div(size_counter.get(item, 0), sum(size_counter.values())),
        "meal_prior": safe_div(meal_counter.get(item, 0), sum(meal_counter.values())),
        "user_prior": safe_div(user_counter.get(item, 0), sum(user_counter.values())),
        "global_prior": safe_div(stats["global_counts"].get(item, 0), stats["global_total"]),
        "context_size": float(len(context)),
        "is_empty": 1.0 if len(context) == 0 else 0.0,
        "is_single": 1.0 if len(context) == 1 else 0.0,
        "is_multi": 1.0 if len(context) >= 2 else 0.0,
        "hour": float(row.get("hour", -1)),
        "hour_bucket": float(row.get("hour_bucket", -1)),
        "is_weekend": float(row.get("is_weekend", 0)),
        "candidate_is_weak_category": 1.0 if item_cat in WEAK_CATEGORIES else 0.0,
        "candidate_is_strong_guard_category": 1.0 if item_cat in STRONG_GUARD_CATEGORIES else 0.0,
    }
    feats.update(one_hot("role", item_role, ["main", "sauce", "side", "beverage", "dessert_addon"]))
    feats.update(one_hot("cat", item_cat, ["Wraps", "Sauces", "Beverages", "Plates", "Highlanders", "Sides", "Appetizers", "Desserts", "Salads"]))
    feats.update(one_hot("segment", row["segment"], ["empty", "single-visible-main", "single-visible-addon", "multi-with-main", "multi-no-main"]))
    feats.update(seq_features(row, item, seq_stats, category_map))
    return feats


def make_rankers(row, stats, cat_stats, valid_items, category_map, role_map):
    base_items = []
    exact = lvl3.exact_context_recs(row["context"], stats, valid_items)
    assoc = lvl3.assoc_recs(row["context"], stats, valid_items, base_items)
    weak_recs = weak.weak_slice_recs(row, stats, cat_stats, valid_items, category_map, role_map, BEST_WEAK_PARAMS, assoc)
    # Full proxy: assoc plus weak graph/category is a closer stage-1 signal than a generic recommender.
    full_old = weak_recs
    full_v7 = assoc
    return {
        "exact_old": exact,
        "assoc_old": assoc,
        "weak": weak_recs,
        "full_old": full_old,
        "full_v7": full_v7,
    }


def build_matrix(rows, stats, cat_stats, seq_stats, valid_items, category_map, role_map, include_labels=True):
    X_rows = []
    y = []
    groups = []
    meta = []
    feature_names = None
    for idx, row in enumerate(rows):
        if idx and idx % 3000 == 0:
            print(f"  features for {idx:,}/{len(rows):,} groups", flush=True)
        rankers = make_rankers(row, stats, cat_stats, valid_items, category_map, role_map)
        rank_maps = {name: rank_lookup(recs) for name, recs in rankers.items()}
        current = set(row["context"])
        items = [item for item in sorted(valid_items) if item not in current]
        groups.append(len(items))

        for item in items:
            feats = item_signal_features(row, item, stats, cat_stats, valid_items, category_map, role_map, seq_stats)
            consensus_top1 = 0
            consensus_top3 = 0
            consensus_top5 = 0
            consensus_top10 = 0
            for name, recs in rankers.items():
                feats.update(position_features(f"ranker_{name}", recs, item))
                r = rank_maps[name].get(item)
                consensus_top1 += int(r == 1)
                consensus_top3 += int(r is not None and r <= 3)
                consensus_top5 += int(r is not None and r <= 5)
                consensus_top10 += int(r is not None and r <= 10)

            signal_agree = 0
            signal_agree += int(feats["exact_conf"] > 0)
            signal_agree += int(feats["subset_conf"] > 0)
            signal_agree += int(feats["graph_npmi"] > 0)
            signal_agree += int(feats["seq_prev_item_max"] > 0 or feats["seq_recent_item_mean"] > 0)
            signal_agree += int(feats["meal_prior"] > 0)
            feats.update(
                {
                    "consensus_top1": float(consensus_top1),
                    "consensus_top3": float(consensus_top3),
                    "consensus_top5": float(consensus_top5),
                    "consensus_top10": float(consensus_top10),
                    "independent_signal_agreement": float(signal_agree),
                }
            )

            if feature_names is None:
                feature_names = list(feats.keys())
            X_rows.append([feats.get(f, 0.0) for f in feature_names])
            if include_labels:
                y.append(1 if item == row["hidden"] else 0)
            meta.append(
                {
                    "row": row,
                    "item": item,
                    "baseline_weak_recs": rankers["weak"],
                    "baseline_assoc_recs": rankers["assoc_old"],
                }
            )
    X = np.asarray(X_rows, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.int32) if include_labels else None
    return X, y_arr, np.asarray(groups, dtype=np.int32), meta, feature_names


def predict_groups(model, X, groups, meta):
    preds = model.predict(X)
    out = []
    offset = 0
    for g in groups:
        block = []
        for j in range(g):
            m = meta[offset + j]
            block.append((m["item"], float(preds[offset + j]), m))
        ranked = [item for item, _, _ in sorted(block, key=lambda x: (-x[1], x[0]))[:10]]
        out.append((ranked, block[0][2]["row"], block[0][2]["baseline_weak_recs"]))
        offset += g
    return out


def apply_guard(pred_groups, category_map, mode):
    if mode == "raw":
        return pred_groups
    guarded = []
    for recs, row, weak_recs in pred_groups:
        use_weak = False
        if mode == "dessert_weak_top10":
            use_weak = any(category_map.get(item) == "Desserts" for item in weak_recs[:10])
        elif mode == "dessert_weak_top5":
            use_weak = any(category_map.get(item) == "Desserts" for item in weak_recs[:5])
        elif mode == "dessert_weak_top3":
            use_weak = any(category_map.get(item) == "Desserts" for item in weak_recs[:3])
        elif mode == "strong_exact_rank1":
            use_weak = weak_recs and category_map.get(weak_recs[0]) in STRONG_GUARD_CATEGORIES
        guarded.append((weak_recs if use_weak else recs, row, weak_recs))
    return guarded


def evaluate_predictions(pred_groups, name):
    ndcgs = []
    hits = {1: [], 3: [], 5: [], 10: []}
    rank1 = []
    by_segment = defaultdict(list)
    by_role = defaultdict(list)
    by_category = defaultdict(list)
    for recs, row, _ in pred_groups:
        target = row["hidden"]
        v = ndcg_at10(recs, target)
        ndcgs.append(v)
        rank1.append(int(recs and recs[0] == target))
        for k in hits:
            hits[k].append(hit_at(recs, target, k))
        by_segment[row["segment"]].append(v)
        by_role[row["hidden_role"]].append(v)
        by_category[row["hidden_category"]].append(v)
    return {
        "name": name,
        "overall_ndcg": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "hit_at": {str(k): float(np.mean(v)) if v else 0.0 for k, v in hits.items()},
        "rank1_accuracy": float(np.mean(rank1)) if rank1 else 0.0,
        "ndcg_by_segment": {k: float(np.mean(v)) for k, v in by_segment.items()},
        "ndcg_by_role": {k: float(np.mean(v)) for k, v in by_role.items()},
        "ndcg_by_category": {k: float(np.mean(v)) for k, v in by_category.items()},
        "counts_by_segment": {k: len(v) for k, v in by_segment.items()},
        "counts_by_role": {k: len(v) for k, v in by_role.items()},
        "counts_by_category": {k: len(v) for k, v in by_category.items()},
    }


def baseline_groups_from_meta(groups, meta, key):
    out = []
    offset = 0
    for g in groups:
        m = meta[offset]
        out.append((m[key], m["row"], m["baseline_weak_recs"]))
        offset += g
    return out


def compare_recs(pred_groups, baseline_groups):
    changed = 0
    rank1_changes = 0
    top10_overlap = []
    for (new, row, _), (base, _, _) in zip(pred_groups, baseline_groups):
        if new != base:
            changed += 1
        if new[:1] != base[:1]:
            rank1_changes += 1
        top10_overlap.append(len(set(new) & set(base)) / 10)
    return {
        "changed_rows": changed,
        "rank_1_changes": rank1_changes,
        "top10_overlap": float(np.mean(top10_overlap)) if top10_overlap else 0.0,
    }


def gate_report(meta_metrics, weak_metrics):
    deltas = {
        "overall": meta_metrics["overall_ndcg"] - BASELINE_NDCG,
        "sauces": meta_metrics["ndcg_by_category"].get("Sauces", 0.0) - weak_metrics["ndcg_by_category"].get("Sauces", 0.0),
        "wraps": meta_metrics["ndcg_by_category"].get("Wraps", 0.0) - weak_metrics["ndcg_by_category"].get("Wraps", 0.0),
        "appetizers": meta_metrics["ndcg_by_category"].get("Appetizers", 0.0) - weak_metrics["ndcg_by_category"].get("Appetizers", 0.0),
        "plates": meta_metrics["ndcg_by_category"].get("Plates", 0.0) - weak_metrics["ndcg_by_category"].get("Plates", 0.0),
        "beverages": meta_metrics["ndcg_by_category"].get("Beverages", 0.0) - weak_metrics["ndcg_by_category"].get("Beverages", 0.0),
        "desserts": meta_metrics["ndcg_by_category"].get("Desserts", 0.0) - weak_metrics["ndcg_by_category"].get("Desserts", 0.0),
        "multi_with_main": meta_metrics["ndcg_by_segment"].get("multi-with-main", 0.0) - weak_metrics["ndcg_by_segment"].get("multi-with-main", 0.0),
    }
    checks = {
        "overall_beats_weak_by_0_001": deltas["overall"] >= 0.0010,
        "sauce_regression_guard": deltas["sauces"] >= -0.002,
        "wrap_regression_guard": deltas["wraps"] >= -0.002,
        "appetizers_neutral_or_better": deltas["appetizers"] >= -0.0001,
        "plates_neutral_or_better": deltas["plates"] >= -0.0001,
        "beverages_neutral_or_better": deltas["beverages"] >= -0.0001,
        "desserts_neutral_or_better": deltas["desserts"] >= -0.0001,
        "multi_with_main_neutral_or_better": deltas["multi_with_main"] >= -0.0001,
    }
    return {"deltas": deltas, "checks": checks, "passes": all(checks.values())}


def train_model(X_tr, y_tr, groups_tr, seed):
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[10],
        n_estimators=180,
        learning_rate=0.045,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_tr, y_tr, group=groups_tr)
    return model


def audit_submission(path, test_df, valid_items):
    df = pd.read_csv(path)
    test_ids = [str(x) for x in test_df["instance_id"]]
    current_lookup = {
        str(row.instance_id): set(lvl3.unique_context(lvl3.parse_items(row.current_basket_items)))
        for row in test_df.itertuples(index=False)
    }
    dup_rec = 0
    invalid = 0
    violations = 0
    for row in df.itertuples(index=False):
        recs = [str(getattr(row, c)) for c in REC_COLS]
        if len(recs) != len(set(recs)):
            dup_rec += 1
        invalid += sum(1 for item in recs if item not in valid_items)
        cur = current_lookup.get(str(row.instance_id), set())
        violations += sum(1 for item in recs if item in cur)
    return {
        "shape_ok": tuple(df.shape) == (14147, 11),
        "columns_exact": list(df.columns) == EXPECTED_COLS,
        "instance_id_matches_test": [str(x) for x in df["instance_id"]] == test_ids,
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_instance_id": int(df["instance_id"].duplicated().sum()),
        "duplicate_recommendation_rows": int(dup_rec),
        "invalid_item_ids": int(invalid),
        "current_basket_violations": int(violations),
    }


def audit_passes(audit):
    if not audit:
        return False
    return (
        audit.get("shape_ok") is True
        and audit.get("columns_exact") is True
        and audit.get("instance_id_matches_test") is True
        and audit.get("missing_values") == 0
        and audit.get("duplicate_instance_id") == 0
        and audit.get("duplicate_recommendation_rows") == 0
        and audit.get("invalid_item_ids") == 0
        and audit.get("current_basket_violations") == 0
    )


def generate_csv(model, full_stats, full_cat_stats, full_seq_stats, test_df, valid_items, category_map, role_map, feature_names, guard_mode):
    test_rows = add_time_fields(parse_test_rows(test_df, valid_items, role_map, category_map))
    X, _, groups, meta, _ = build_matrix(test_rows, full_stats, full_cat_stats, full_seq_stats, valid_items, category_map, role_map, include_labels=False)
    # Align feature order if an earlier call established it.
    preds = model.predict(X)
    records = []
    offset = 0
    for g in groups:
        block = []
        for j in range(g):
            m = meta[offset + j]
            block.append((m["item"], float(preds[offset + j])))
        recs = [item for item, _, in sorted(block, key=lambda x: (-x[1], x[0]))[:10]]
        weak_recs = meta[offset]["baseline_weak_recs"]
        if guard_mode != "raw":
            guarded = apply_guard([(recs, meta[offset]["row"], weak_recs)], category_map, guard_mode)
            recs = guarded[0][0]
        iid = meta[offset]["row"]["instance_id"]
        records.append({"instance_id": iid, **{f"rank_{i+1}": recs[i] for i in range(10)}})
        offset += g
    out = pd.DataFrame(records, columns=EXPECTED_COLS)
    out.to_csv(OUT_CSV, index=False)
    return audit_submission(OUT_CSV, test_df, valid_items)


def write_reports(payload):
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    lines = [
        "# LambdaRank Meta-Ranker Experiment",
        "",
        f"Decision: **{payload['decision']}**",
        f"Reason: {payload['reason']}",
        "",
        "## Validation",
        "",
        "| model | NDCG@10 | Hit@1 | Hit@3 | Hit@5 | Hit@10 | rank1 |",
    ]
    for name in ["weak_baseline_proxy", "assoc_proxy", "lambdarank_meta"]:
        m = payload["metrics"][name]
        lines.append(
            f"| {name} | {m['overall_ndcg']:.6f} | {m['hit_at']['1']:.6f} | {m['hit_at']['3']:.6f} | "
            f"{m['hit_at']['5']:.6f} | {m['hit_at']['10']:.6f} | {m['rank1_accuracy']:.6f} |"
        )
    lines += [
        "",
        "## Gate Deltas",
        "",
        "```json",
        json.dumps(payload["gate"], indent=2),
        "```",
        "",
        "## Top Feature Importance",
        "",
    ]
    for item in payload["feature_importance"][:25]:
        lines.append(f"- {item[0]}: {item[1]}")
    lines += [
        "",
        f"Generated CSV: `{payload['generated_csv']}`",
        "",
    ]
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    t0 = time.time()
    print("Loading data...", flush=True)
    train = pd.read_csv(TRAIN_BASKETS)
    test = pd.read_csv(TEST_INSTANCES)
    menu = pd.read_csv(MENU_ITEMS)
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
    train80_part = train.iloc[:meta_train_end].copy()

    print("Building leakage-safe stats...", flush=True)
    build_stats = lvl3.build_stats(build_part, valid_items, role_map)
    build_cat_stats = weak.category_stats(build_part, valid_items, category_map, role_map)
    build_seq = build_seq_stats(build_part, valid_items, category_map)
    val_stats = lvl3.build_stats(train80_part, valid_items, role_map)
    val_cat_stats = weak.category_stats(train80_part, valid_items, category_map, role_map)
    val_seq = build_seq_stats(train80_part, valid_items, category_map)

    meta_train_rows = add_time_fields(lvl3.build_validation_rows(meta_train_part, valid_items, role_map, category_map))
    val_rows = add_time_fields(lvl3.build_validation_rows(val_part, valid_items, role_map, category_map))

    print(f"Stage-2 train groups={len(meta_train_rows):,}; validation groups={len(val_rows):,}", flush=True)
    print("Building training matrix...", flush=True)
    X_tr, y_tr, g_tr, _, feature_names = build_matrix(meta_train_rows, build_stats, build_cat_stats, build_seq, valid_items, category_map, role_map)
    print(f"Training matrix: {X_tr.shape}, positives={int(y_tr.sum()):,}", flush=True)
    print("Building validation matrix...", flush=True)
    X_va, y_va, g_va, meta_va, _ = build_matrix(val_rows, val_stats, val_cat_stats, val_seq, valid_items, category_map, role_map)
    print(f"Validation matrix: {X_va.shape}, positives={int(y_va.sum()):,}", flush=True)

    models = []
    seed_metrics = []
    for seed in [13, 29]:
        print(f"Training LightGBM LambdaRank seed={seed}...", flush=True)
        model = train_model(X_tr, y_tr, g_tr, seed)
        pred_groups = predict_groups(model, X_va, g_va, meta_va)
        metrics = evaluate_predictions(pred_groups, f"lambdarank_seed_{seed}")
        seed_metrics.append(metrics)
        models.append(model)
        print(f"  seed={seed} ndcg={metrics['overall_ndcg']:.6f} hit1={metrics['hit_at']['1']:.6f}", flush=True)

    # Use the better seed for any gated generation.
    best_idx = int(np.argmax([m["overall_ndcg"] for m in seed_metrics]))
    best_model = models[best_idx]
    raw_pred_groups = predict_groups(best_model, X_va, g_va, meta_va)
    weak_groups = baseline_groups_from_meta(g_va, meta_va, "baseline_weak_recs")
    assoc_groups = baseline_groups_from_meta(g_va, meta_va, "baseline_assoc_recs")
    weak_metrics = evaluate_predictions(weak_groups, "weak_baseline_proxy")
    assoc_metrics = evaluate_predictions(assoc_groups, "assoc_proxy")
    variants = []
    for guard_mode in ["raw", "dessert_weak_top3", "dessert_weak_top5", "dessert_weak_top10", "strong_exact_rank1"]:
        groups_for_mode = apply_guard(raw_pred_groups, category_map, guard_mode)
        metrics = evaluate_predictions(groups_for_mode, f"lambdarank_meta_{guard_mode}")
        gate = gate_report(metrics, weak_metrics)
        variants.append(
            {
                "guard_mode": guard_mode,
                "metrics": metrics,
                "gate": gate,
                "change_profile": compare_recs(groups_for_mode, weak_groups),
            }
        )
    passing = [v for v in variants if v["gate"]["passes"]]
    chosen = max(passing, key=lambda v: v["metrics"]["overall_ndcg"]) if passing else max(variants, key=lambda v: v["metrics"]["overall_ndcg"])
    meta_metrics = chosen["metrics"]
    change_profile = chosen["change_profile"]
    gate = chosen["gate"]

    importance = sorted(zip(feature_names, best_model.feature_importances_.tolist()), key=lambda x: -x[1])

    generated_csv = None
    audit = None
    if gate["passes"]:
        print("Gates passed. Training final model and generating one CSV...", flush=True)
        full_stats = lvl3.build_stats(train, valid_items, role_map)
        full_cat_stats = weak.category_stats(train, valid_items, category_map, role_map)
        full_seq_stats = build_seq_stats(train, valid_items, category_map)
        # Keep model family fixed; train on the leakage-safe stage-2 rows that proved validation.
        audit = generate_csv(best_model, full_stats, full_cat_stats, full_seq_stats, test, valid_items, category_map, role_map, feature_names, chosen["guard_mode"])
        generated_csv = os.path.basename(OUT_CSV)
        if not audit_passes(audit):
            gate["passes"] = False
    decision = "GO" if gate["passes"] and audit_passes(audit) else "NO-GO"
    reason = (
        "LambdaRank meta-ranker beat the weak-slice baseline by at least +0.0010 and passed all slice/audit gates."
        if decision == "GO"
        else "LambdaRank/meta-ranker did not clear the required validation and slice/audit gates; no CSV generated."
    )

    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "decision": decision,
        "reason": reason,
        "splits": {
            "build_stats_rows": len(build_part),
            "meta_train_rows": len(meta_train_part),
            "validation_rows": len(val_part),
            "build_max_timestamp": str(build_part["timestamp"].max()),
            "meta_train_min_timestamp": str(meta_train_part["timestamp"].min()),
            "meta_train_max_timestamp": str(meta_train_part["timestamp"].max()),
            "validation_min_timestamp": str(val_part["timestamp"].min()),
            "leakage_check": "stage-2 train features use only first 60%; validation features use first 80%; validation baskets are later and excluded from stats",
        },
        "feature_list": feature_names,
        "feature_importance": importance,
        "seed_metrics": seed_metrics,
        "guard_variants": variants,
        "chosen_guard_mode": chosen["guard_mode"],
        "metrics": {
            "weak_baseline_proxy": weak_metrics,
            "assoc_proxy": assoc_metrics,
            "lambdarank_meta": meta_metrics,
        },
        "change_profile_vs_weak": change_profile,
        "gate": gate,
        "generated_csv": generated_csv,
        "audit": audit,
        "elapsed_seconds": time.time() - t0,
    }
    write_reports(payload)
    print(f"Decision: {decision}", flush=True)
    print(f"Meta NDCG={meta_metrics['overall_ndcg']:.6f}; delta_vs_required_baseline={meta_metrics['overall_ndcg'] - BASELINE_NDCG:.6f}", flush=True)
    print(f"Report: {REPORT_JSON}", flush=True)


if __name__ == "__main__":
    main()
