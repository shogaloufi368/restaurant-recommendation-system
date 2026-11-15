import json
import math
import os
import time
from collections import Counter, defaultdict
from itertools import combinations

import pandas as pd

import build_level3_role_basket_completion as lvl3


BASE = os.path.dirname(os.path.abspath(__file__))

TRAIN_BASKETS = os.path.join(BASE, "train_baskets.csv")
TEST_INSTANCES = os.path.join(BASE, "test_instances.csv")
MENU_ITEMS = os.path.join(BASE, "menu_items.csv")

REPORT_PATH = os.path.join(BASE, "weak_slice_assoc_diagnosis_report.md")
JSON_PATH = os.path.join(BASE, "weak_slice_assoc_diagnosis_report.json")

OLD_BASE = "basket_completion_assoc_oldbase.csv"
V7_BASE = "basket_completion_assoc_v7base.csv"
OLD_OUT = "candidate_weak_slice_assoc_oldbase.csv"
V7_OUT = "candidate_weak_slice_assoc_v7base.csv"

REC_COLS = [f"rank_{i}" for i in range(1, 11)]
EXPECTED_COLS = ["instance_id"] + REC_COLS

WEAK_ROLES = {"beverage", "dessert_addon"}
WEAK_CATEGORIES = {"Beverages", "Plates", "Desserts", "Appetizers"}
STRONG_GUARD_CATEGORIES = {"Sauces", "Wraps"}

PARAM_GRID = [
    {"name": "graph_only_010", "graph": 0.10, "cat": 0.0, "user": 0.0, "meal": 0.0, "subset": 0.0},
    {"name": "graph_only_025", "graph": 0.25, "cat": 0.0, "user": 0.0, "meal": 0.0, "subset": 0.0},
    {"name": "cat_010", "graph": 0.0, "cat": 0.10, "user": 0.0, "meal": 0.0, "subset": 0.0},
    {"name": "cat_025", "graph": 0.0, "cat": 0.25, "user": 0.0, "meal": 0.0, "subset": 0.0},
    {"name": "subset_025", "graph": 0.0, "cat": 0.0, "user": 0.0, "meal": 0.0, "subset": 0.25},
    {"name": "subset_050", "graph": 0.0, "cat": 0.0, "user": 0.0, "meal": 0.0, "subset": 0.50},
    {"name": "graph_cat_small", "graph": 0.10, "cat": 0.10, "user": 0.0, "meal": 0.0, "subset": 0.0},
    {"name": "graph_cat_mid", "graph": 0.20, "cat": 0.20, "user": 0.0, "meal": 0.0, "subset": 0.0},
    {"name": "graph_subset", "graph": 0.15, "cat": 0.0, "user": 0.0, "meal": 0.0, "subset": 0.35},
    {"name": "cat_subset", "graph": 0.0, "cat": 0.15, "user": 0.0, "meal": 0.0, "subset": 0.35},
    {"name": "tie_user_meal", "graph": 0.0, "cat": 0.0, "user": 0.025, "meal": 0.015, "subset": 0.0},
    {"name": "all_tiny", "graph": 0.08, "cat": 0.08, "user": 0.015, "meal": 0.010, "subset": 0.15},
    {"name": "all_small", "graph": 0.15, "cat": 0.15, "user": 0.020, "meal": 0.015, "subset": 0.25},
]


def category_stats(train_df, valid_items, category_map, role_map):
    ctx_cat_counts = defaultdict(Counter)
    segment_cat_counts = defaultdict(Counter)
    visible_role_cat_counts = defaultdict(Counter)
    meal_cat_counts = defaultdict(Counter)
    user_cat_counts = defaultdict(Counter)

    for row in train_df.itertuples(index=False):
        basket = tuple(x for x in lvl3.unique_context(lvl3.parse_items(row.item_ids)) if x in valid_items)
        if not basket:
            continue
        bset = set(basket)
        user = str(row.user_id)
        meal = str(row.meal_period)
        for target in bset:
            context = tuple(sorted(bset - {target}))
            cat = category_map.get(target, "unknown")
            segment = lvl3.segment_for_context(context, role_map)
            visible_roles = tuple(sorted(role_map.get(x, "dessert_addon") for x in context))
            ctx_cat_counts[context][cat] += 1
            segment_cat_counts[segment][cat] += 1
            visible_role_cat_counts[visible_roles][cat] += 1
            meal_cat_counts[meal][cat] += 1
            user_cat_counts[user][cat] += 1

    return {
        "ctx_cat_counts": ctx_cat_counts,
        "segment_cat_counts": segment_cat_counts,
        "visible_role_cat_counts": visible_role_cat_counts,
        "meal_cat_counts": meal_cat_counts,
        "user_cat_counts": user_cat_counts,
    }


def norm_prob(counter, key):
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    return counter.get(key, 0) / total


def graph_npmi(context, item, stats):
    if not context:
        return 0.0
    support = stats["subset_support"]
    n = stats["n_baskets"]
    vals = []
    for visible in context:
        pair = tuple(sorted((visible, item)))
        pair_count = support.get(pair, 0)
        if pair_count <= 0:
            continue
        p_pair = pair_count / n
        p_v = support.get((visible,), 0) / n
        p_i = support.get((item,), 0) / n
        if p_pair <= 0 or p_v <= 0 or p_i <= 0:
            continue
        pmi = math.log(p_pair / (p_v * p_i))
        npmi = pmi / (-math.log(p_pair)) if p_pair < 1 else 0.0
        vals.append(max(0.0, npmi))
    return sum(vals) / len(vals) if vals else 0.0


def category_signal(row, item, cat_stats, category_map, role_map):
    cat = category_map.get(item, "unknown")
    context = row["context"]
    segment = row["segment"]
    visible_roles = tuple(sorted(role_map.get(x, "dessert_addon") for x in context))
    pieces = [
        (1.20, norm_prob(cat_stats["ctx_cat_counts"].get(tuple(sorted(context)), Counter()), cat)),
        (0.55, norm_prob(cat_stats["segment_cat_counts"].get(segment, Counter()), cat)),
        (0.55, norm_prob(cat_stats["visible_role_cat_counts"].get(visible_roles, Counter()), cat)),
        (0.08, norm_prob(cat_stats["meal_cat_counts"].get(row["meal_period"], Counter()), cat)),
        (0.08, norm_prob(cat_stats["user_cat_counts"].get(row["user_id"], Counter()), cat)),
    ]
    total_w = sum(w for w, _ in pieces)
    return sum(w * p for w, p in pieces) / total_w if total_w else 0.0


def is_target_candidate(row, item, category_map, role_map):
    cat = category_map.get(item, "unknown")
    role = role_map.get(item, "dessert_addon")
    if row["segment"] == "multi-with-main" and cat not in STRONG_GUARD_CATEGORIES:
        return True
    return role in WEAK_ROLES or cat in WEAK_CATEGORIES


def weak_slice_recs(row, stats, cat_stats, valid_items, category_map, role_map, params, base_items=None):
    context = row["context"]
    current = set(context)
    ctx = tuple(sorted(context))
    ctx_counter = stats["ctx_counts"].get(ctx, Counter())
    ctx_total = stats["ctx_totals"].get(ctx, 0)
    base_rank = {item: pos for pos, item in enumerate(base_items or [], start=1)}
    user_counter = stats["user_item_counts"].get(row["user_id"], Counter())
    user_total = sum(user_counter.values())
    meal_counter = stats["meal_item_counts"].get(row["meal_period"], Counter())
    meal_total = sum(meal_counter.values())

    def exact_bucket(item):
        if ctx_total <= 0:
            return 0
        count = ctx_counter.get(item, 0)
        if count <= 0:
            return 0
        conf = count / ctx_total
        return 2 if count >= 3 or conf >= 0.02 else 1

    scored = []
    for item in valid_items:
        if item in current:
            continue
        assoc_score = lvl3.item_score_assoc(context, item, stats, base_rank, lvl3.ASSOC_WEIGHTS)
        targeted = is_target_candidate(row, item, category_map, role_map)
        bonus = 0.0
        graph = 0.0
        cat_sig = 0.0
        subset_conf = 0.0
        if targeted:
            subset_conf, lift_bonus, npmi_bonus = lvl3.weighted_subset_signal(context, item, stats)
            graph = graph_npmi(context, item, stats)
            cat_sig = category_signal(row, item, cat_stats, category_map, role_map)
            user_prior = (user_counter.get(item, 0) / user_total) if user_total else 0.0
            meal_prior = (meal_counter.get(item, 0) / meal_total) if meal_total else 0.0
            bonus = (
                params["graph"] * graph
                + params["cat"] * cat_sig
                + params["subset"] * (subset_conf + 0.35 * lift_bonus + 0.35 * npmi_bonus)
                + params["user"] * user_prior
                + params["meal"] * meal_prior
            )
        scored.append((item, assoc_score + bonus, targeted, graph, cat_sig))

    return [
        item
        for item, _, _, _, _ in sorted(
            scored,
            key=lambda kv: (
                -exact_bucket(kv[0]),
                -kv[1],
                -int(kv[2]),
                -kv[3],
                -kv[4],
                -ctx_counter.get(kv[0], 0),
                -stats["global_counts"].get(kv[0], 0),
                kv[0],
            ),
        )[:10]
    ]


def evaluate(rows, predictor):
    ndcgs = []
    hits = {1: [], 3: [], 5: [], 10: []}
    rank1 = []
    by_segment = defaultdict(list)
    by_role = defaultdict(list)
    by_category = defaultdict(list)
    counts_segment = Counter()
    counts_role = Counter()
    counts_category = Counter()

    for row in rows:
        recs = predictor(row)
        target = row["hidden"]
        ndcg = lvl3.ndcg_at10(recs, target)
        ndcgs.append(ndcg)
        for k in hits:
            hits[k].append(lvl3.hit_at(recs, target, k))
        rank1.append(int(recs and recs[0] == target))
        by_segment[row["segment"]].append(ndcg)
        by_role[row["hidden_role"]].append(ndcg)
        by_category[row["hidden_category"]].append(ndcg)
        counts_segment[row["segment"]] += 1
        counts_role[row["hidden_role"]] += 1
        counts_category[row["hidden_category"]] += 1

    def mean(vals):
        return float(sum(vals) / len(vals)) if vals else 0.0

    return {
        "overall_ndcg": mean(ndcgs),
        "hit_at": {str(k): mean(v) for k, v in hits.items()},
        "rank1_accuracy": mean(rank1),
        "ndcg_by_segment": {k: mean(v) for k, v in by_segment.items()},
        "ndcg_by_role": {k: mean(v) for k, v in by_role.items()},
        "ndcg_by_category": {k: mean(v) for k, v in by_category.items()},
        "counts_by_segment": dict(counts_segment),
        "counts_by_role": dict(counts_role),
        "counts_by_category": dict(counts_category),
    }


def rank_of(recs, target):
    for i, item in enumerate(recs, start=1):
        if item == target:
            return i
    return 999


def diagnose_weak_slices(rows, base_recs_by_id, category_map, role_map):
    focus = [
        row
        for row in rows
        if row["segment"] == "multi-with-main"
        or row["hidden_role"] in WEAK_ROLES
        or row["hidden_category"] in {"Plates"}
    ]
    context_counts = Counter()
    hidden_counts = Counter()
    wrong_rank1 = Counter()
    rank_buckets = Counter()
    failure_type = Counter()
    fixability = Counter()

    for row in focus:
        recs = base_recs_by_id[row["instance_id"]]
        target = row["hidden"]
        context_counts["|".join(row["context"]) if row["context"] else "(empty)"] += 1
        hidden_counts[target] += 1
        r = rank_of(recs, target)
        if r == 1:
            rank_buckets["rank_1"] += 1
        elif r <= 3:
            rank_buckets["rank_2_3"] += 1
        elif r <= 5:
            rank_buckets["rank_4_5"] += 1
        elif r <= 10:
            rank_buckets["rank_6_10"] += 1
        else:
            rank_buckets["missing"] += 1
        if recs and recs[0] != target:
            wrong_rank1[recs[0]] += 1
            if role_map.get(recs[0]) != row["hidden_role"]:
                failure_type["wrong_role"] += 1
            elif category_map.get(recs[0]) != row["hidden_category"]:
                failure_type["right_role_wrong_category"] += 1
            else:
                failure_type["right_category_wrong_item"] += 1

        subset_conf, lift_bonus, npmi_bonus = lvl3.weighted_subset_signal(row["context"], target, diagnose_weak_slices.stats)
        graph = graph_npmi(row["context"], target, diagnose_weak_slices.stats)
        meal_counter = diagnose_weak_slices.stats["meal_item_counts"].get(row["meal_period"], Counter())
        user_counter = diagnose_weak_slices.stats["user_item_counts"].get(row["user_id"], Counter())
        if subset_conf > 0.05:
            fixability["subset_backoff_signal"] += 1
        if graph > 0.10:
            fixability["graph_signal"] += 1
        if meal_counter.get(target, 0) > 0:
            fixability["meal_prior_has_target"] += 1
        if user_counter.get(target, 0) > 0:
            fixability["user_prior_has_target"] += 1

    return {
        "focus_rows": len(focus),
        "top_visible_contexts": context_counts.most_common(20),
        "top_true_hidden_items": hidden_counts.most_common(20),
        "current_predicted_rank_buckets": dict(rank_buckets),
        "common_wrong_rank1": wrong_rank1.most_common(20),
        "failure_type": dict(failure_type),
        "fixability_signals": dict(fixability),
    }


def gate_decision(base_metrics, best_metrics):
    overall = best_metrics["overall_ndcg"] > base_metrics["overall_ndcg"]
    sauce_delta = best_metrics["ndcg_by_category"].get("Sauces", 0.0) - base_metrics["ndcg_by_category"].get("Sauces", 0.0)
    wraps_delta = best_metrics["ndcg_by_category"].get("Wraps", 0.0) - base_metrics["ndcg_by_category"].get("Wraps", 0.0)
    multi_delta = best_metrics["ndcg_by_segment"].get("multi-with-main", 0.0) - base_metrics["ndcg_by_segment"].get("multi-with-main", 0.0)
    beverage_delta = best_metrics["ndcg_by_role"].get("beverage", 0.0) - base_metrics["ndcg_by_role"].get("beverage", 0.0)
    plates_delta = best_metrics["ndcg_by_category"].get("Plates", 0.0) - base_metrics["ndcg_by_category"].get("Plates", 0.0)
    dessert_delta = best_metrics["ndcg_by_role"].get("dessert_addon", 0.0) - base_metrics["ndcg_by_role"].get("dessert_addon", 0.0)
    checks = {
        "overall_beats_assoc": overall,
        "sauce_regression_guard": sauce_delta >= -0.002,
        "wrap_regression_guard": wraps_delta >= -0.002,
        "multi_with_main_improves": multi_delta > 0.0,
        "beverage_neutral_or_better": beverage_delta >= -0.0001,
        "plates_neutral_or_better": plates_delta >= -0.0001,
        "dessert_addon_neutral_or_better": dessert_delta >= -0.0001,
    }
    return all(checks.values()), {
        "checks": checks,
        "deltas": {
            "overall": best_metrics["overall_ndcg"] - base_metrics["overall_ndcg"],
            "sauces": sauce_delta,
            "wraps": wraps_delta,
            "multi_with_main": multi_delta,
            "beverage": beverage_delta,
            "plates": plates_delta,
            "dessert_addon": dessert_delta,
        },
    }


def generate_candidate(base_file, out_file, stats, cat_stats, test_df, valid_items, category_map, role_map, params):
    base_df = pd.read_csv(os.path.join(BASE, base_file))
    if list(base_df.columns) != EXPECTED_COLS:
        raise ValueError(f"{base_file} columns are not valid submission columns")
    test_by_id = {str(r.instance_id): r for r in test_df.itertuples(index=False)}
    out_df = base_df.copy()
    audit = Counter()

    for idx, row in base_df.iterrows():
        iid = str(row["instance_id"])
        test_row = test_by_id[iid]
        context = lvl3.unique_context(lvl3.parse_items(test_row.current_basket_items))
        segment = lvl3.segment_for_context(context, role_map)
        pseudo = {
            "instance_id": iid,
            "user_id": str(test_row.user_id),
            "meal_period": str(test_row.meal_period),
            "context": context,
            "segment": segment,
        }
        base_items = [str(row[col]) for col in REC_COLS]
        recs = weak_slice_recs(pseudo, stats, cat_stats, valid_items, category_map, role_map, params, base_items)
        if recs != base_items:
            audit["changed_rows"] += 1
        if len(set(recs)) != len(recs):
            audit["duplicate_rows"] += 1
        audit["invalid_items"] += sum(1 for item in recs if item not in valid_items)
        if set(recs) & set(context):
            audit["basket_violations"] += 1
        for col, item in zip(REC_COLS, recs):
            out_df.at[idx, col] = item

    out_df.to_csv(os.path.join(BASE, out_file), index=False)
    result = dict(audit)
    result["base_file"] = base_file
    result["output_file"] = out_file
    return result


def fmt(x):
    return f"{x:.6f}"


def write_report(payload):
    base = payload["baseline"]
    best = payload["best"]
    lines = [
        "# Weak-Slice Association Reranker Diagnosis",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "## Decision",
        "",
        f"**{payload['decision']}**",
        "",
        payload["reason"],
        "",
        "## Validation Setup",
        "",
        f"- Train rows: {payload['split']['train_rows']}",
        f"- Validation rows: {payload['split']['validation_rows']}",
        f"- Train max timestamp: {payload['split']['train_max_timestamp']}",
        f"- Validation min timestamp: {payload['split']['validation_min_timestamp']}",
        "",
        "## Overall",
        "",
        "| method | NDCG@10 | Hit@1 | Hit@3 | Hit@5 | Hit@10 | rank_1 acc |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| basket_completion_assoc_oldbase_proxy | {fmt(base['overall_ndcg'])} | {fmt(base['hit_at']['1'])} | {fmt(base['hit_at']['3'])} | {fmt(base['hit_at']['5'])} | {fmt(base['hit_at']['10'])} | {fmt(base['rank1_accuracy'])} |",
        f"| weak_slice_best ({payload['best_params']['name']}) | {fmt(best['overall_ndcg'])} | {fmt(best['hit_at']['1'])} | {fmt(best['hit_at']['3'])} | {fmt(best['hit_at']['5'])} | {fmt(best['hit_at']['10'])} | {fmt(best['rank1_accuracy'])} |",
        "",
        "## Gate Deltas",
        "",
        "| gate | pass | delta |",
        "|---|---:|---:|",
    ]
    for key, ok in payload["gate"]["checks"].items():
        delta_key = {
            "overall_beats_assoc": "overall",
            "sauce_regression_guard": "sauces",
            "wrap_regression_guard": "wraps",
            "multi_with_main_improves": "multi_with_main",
            "beverage_neutral_or_better": "beverage",
            "plates_neutral_or_better": "plates",
            "dessert_addon_neutral_or_better": "dessert_addon",
        }[key]
        lines.append(f"| {key} | {ok} | {fmt(payload['gate']['deltas'][delta_key])} |")

    lines += [
        "",
        "## Weak-Slice Diagnosis",
        "",
        f"Focus rows: {payload['diagnosis']['focus_rows']}",
        "",
        "### Current Predicted Rank Buckets",
        "",
        json.dumps(payload["diagnosis"]["current_predicted_rank_buckets"], indent=2),
        "",
        "### Failure Type",
        "",
        json.dumps(payload["diagnosis"]["failure_type"], indent=2),
        "",
        "### Fixability Signals",
        "",
        json.dumps(payload["diagnosis"]["fixability_signals"], indent=2),
        "",
        "### Top Visible Contexts",
        "",
    ]
    for item, count in payload["diagnosis"]["top_visible_contexts"]:
        lines.append(f"- `{item}`: {count}")
    lines += ["", "### Top True Hidden Items", ""]
    for item, count in payload["diagnosis"]["top_true_hidden_items"]:
        lines.append(f"- `{item}`: {count}")
    lines += ["", "### Common Wrong Rank 1", ""]
    for item, count in payload["diagnosis"]["common_wrong_rank1"]:
        lines.append(f"- `{item}`: {count}")

    lines += [
        "",
        "## Parameter Sweep",
        "",
        "| params | NDCG@10 | Hit@1 | multi-with-main | beverage | plates | dessert_addon | sauces delta | wraps delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["sweep"]:
        metrics = row["metrics"]
        deltas = row["gate"]["deltas"]
        lines.append(
            f"| {row['params']['name']} | {fmt(metrics['overall_ndcg'])} | {fmt(metrics['hit_at']['1'])} | "
            f"{fmt(metrics['ndcg_by_segment'].get('multi-with-main', 0.0))} | "
            f"{fmt(metrics['ndcg_by_role'].get('beverage', 0.0))} | "
            f"{fmt(metrics['ndcg_by_category'].get('Plates', 0.0))} | "
            f"{fmt(metrics['ndcg_by_role'].get('dessert_addon', 0.0))} | "
            f"{fmt(deltas['sauces'])} | {fmt(deltas['wraps'])} |"
        )

    lines += ["", "## Candidate CSV Generation", ""]
    if payload["generated_candidates"]:
        for item in payload["generated_candidates"]:
            lines.append(
                f"- `{item['output_file']}` from `{item['base_file']}`: "
                f"changed={item.get('changed_rows', 0)}, dups={item.get('duplicate_rows', 0)}, "
                f"invalid={item.get('invalid_items', 0)}, basket_violations={item.get('basket_violations', 0)}"
            )
    else:
        lines.append("No CSV generated.")
    lines.append("")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    t0 = time.time()
    print("Loading data...")
    train = pd.read_csv(TRAIN_BASKETS)
    test = pd.read_csv(TEST_INSTANCES)
    menu = pd.read_csv(MENU_ITEMS)
    valid_items = set(menu["item_id"].astype(str))
    category_map, role_map = lvl3.make_menu_maps(menu)

    train_part, val_part = lvl3.split_train_validation(train)
    val_rows = lvl3.build_validation_rows(val_part, valid_items, role_map, category_map)
    print(f"Validation rows: {len(val_rows):,}")

    print("Building train-only stats...")
    stats = lvl3.build_stats(train_part, valid_items, role_map)
    cat_stats = category_stats(train_part, valid_items, category_map, role_map)

    print("Evaluating assoc proxy baseline...")
    base_recs_by_id = {}
    for row in val_rows:
        base_recs_by_id[row["instance_id"]] = lvl3.assoc_recs(row["context"], stats, valid_items)
    base_metrics = evaluate(val_rows, lambda r: base_recs_by_id[r["instance_id"]])

    diagnose_weak_slices.stats = stats
    diagnosis = diagnose_weak_slices(val_rows, base_recs_by_id, category_map, role_map)

    print("Sweeping targeted rerankers...")
    sweep = []
    best = None
    for params in PARAM_GRID:
        metrics = evaluate(
            val_rows,
            lambda r, p=params: weak_slice_recs(r, stats, cat_stats, valid_items, category_map, role_map, p),
        )
        passes, gate = gate_decision(base_metrics, metrics)
        row = {"params": params, "metrics": metrics, "passes": passes, "gate": gate}
        sweep.append(row)
        print(
            f"  {params['name']}: ndcg={metrics['overall_ndcg']:.6f} "
            f"multi={metrics['ndcg_by_segment'].get('multi-with-main', 0.0):.6f} "
            f"bev={metrics['ndcg_by_role'].get('beverage', 0.0):.6f} "
            f"plates={metrics['ndcg_by_category'].get('Plates', 0.0):.6f} "
            f"pass={passes}"
        )
        if best is None or (passes, metrics["overall_ndcg"]) > (best["passes"], best["metrics"]["overall_ndcg"]):
            best = row

    gates_pass = bool(best["passes"])
    generated = []
    if gates_pass:
        print("Hard gates passed. Generating candidate CSVs from assoc bases...")
        full_stats = lvl3.build_stats(train, valid_items, role_map)
        full_cat_stats = category_stats(train, valid_items, category_map, role_map)
        generated.append(generate_candidate(OLD_BASE, OLD_OUT, full_stats, full_cat_stats, test, valid_items, category_map, role_map, best["params"]))
        generated.append(generate_candidate(V7_BASE, V7_OUT, full_stats, full_cat_stats, test, valid_items, category_map, role_map, best["params"]))
        audit_ok = all(
            item.get("duplicate_rows", 0) == 0
            and item.get("invalid_items", 0) == 0
            and item.get("basket_violations", 0) == 0
            for item in generated
        )
        if not audit_ok:
            gates_pass = False
            reason = "Validation gates passed, but final candidate audit failed."
        else:
            reason = "All hard validation gates passed and final candidate audits passed."
    else:
        print("No targeted rule passed all hard gates. No CSVs will be generated.")
        reason = "No targeted weak-slice rule improved validation while preserving sauce/wrap guards and required weak-slice gates."

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": time.time() - t0,
        "decision": "GO" if gates_pass else "NO-GO",
        "reason": reason,
        "split": {
            "train_rows": int(len(train_part)),
            "validation_rows": int(len(val_rows)),
            "train_max_timestamp": str(train_part["timestamp"].max()),
            "validation_min_timestamp": str(val_part["timestamp"].min()),
        },
        "baseline": base_metrics,
        "diagnosis": diagnosis,
        "best_params": best["params"],
        "best": best["metrics"],
        "gate": best["gate"],
        "sweep": sweep,
        "generated_candidates": generated if gates_pass else [],
    }

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    write_report(payload)

    print(f"\nDecision: {payload['decision']}")
    print(reason)
    print(f"Report: {REPORT_PATH}")
    print(f"JSON: {JSON_PATH}")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
