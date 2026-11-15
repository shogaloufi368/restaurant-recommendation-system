import json
import math
import os
import time
from collections import Counter, defaultdict
from itertools import combinations

import pandas as pd


BASE = os.path.dirname(os.path.abspath(__file__))

TRAIN_BASKETS = os.path.join(BASE, "train_baskets.csv")
TEST_INSTANCES = os.path.join(BASE, "test_instances.csv")
MENU_ITEMS = os.path.join(BASE, "menu_items.csv")

REPORT_PATH = os.path.join(BASE, "level3_role_diagnosis_report.md")
JSON_PATH = os.path.join(BASE, "level3_role_diagnosis_report.json")

REC_COLS = [f"rank_{i}" for i in range(1, 11)]
EXPECTED_COLS = ["instance_id"] + REC_COLS

OLD_BASE = "submission_final_bestt.csv"
V7_BASE = "candidate_v7_single_highconf_rank12_frozen.csv"
OLD_OUT = "candidate_level3_role_oldbase.csv"
V7_OUT = "candidate_level3_role_v7base.csv"

MIN_ASSOC_JOINT = 20
MIN_ASSOC_CONF = 0.03
MAX_LIFT_BONUS = 0.35
MAX_NPMI_BONUS = 0.35

ASSOC_WEIGHTS = {
    "exact_conf": 8.00,
    "subset_conf": 0.85,
    "lift": 0.07,
    "npmi": 0.05,
    "size_prior": 0.24,
    "base_prior": 0.05,
}

LEVEL3_BASE_WEIGHTS = {
    "exact_conf": 8.00,
    "subset_conf": 0.85,
    "lift": 0.07,
    "npmi": 0.05,
    "size_prior": 0.20,
    "base_prior": 0.04,
    "user_item": 0.20,
    "meal_item": 0.035,
}

ROLE_WEIGHT_GRID = [0.0, 0.15, 0.30, 0.50, 0.75, 1.00, 1.30, 1.70, 2.20]
ROLE_SOURCES = {
    "exact": 2.20,
    "subset": 0.95,
    "segment": 0.70,
    "user": 0.55,
    "meal": 0.20,
    "global": 0.15,
}


def parse_items(value):
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [x for x in text.split("|") if x]


def unique_context(items):
    return tuple(sorted(set(str(x) for x in items if str(x))))


def powerset_nonempty(items):
    items = tuple(sorted(items))
    for r in range(1, len(items) + 1):
        for sub in combinations(items, r):
            yield sub


def ndcg_at10(recs, target):
    for i, item in enumerate(recs[:10], start=1):
        if item == target:
            return 1.0 / math.log2(i + 1)
    return 0.0


def hit_at(recs, target, k):
    return int(target in recs[:k])


def role_for_category(category):
    category = str(category)
    if category in {"Wraps", "Plates", "Highlanders"}:
        return "main"
    if category == "Sauces":
        return "sauce"
    if category in {"Sides", "Salads"}:
        return "side"
    if category == "Beverages":
        return "beverage"
    return "dessert_addon"


def make_menu_maps(menu):
    category = dict(zip(menu["item_id"].astype(str), menu["category"].astype(str)))
    role = {item: role_for_category(cat) for item, cat in category.items()}
    return category, role


def segment_for_context(context, role_map):
    roles = [role_map.get(item, "dessert_addon") for item in context]
    if len(context) == 0:
        return "empty"
    if len(context) == 1:
        return "single-visible-main" if roles[0] == "main" else "single-visible-addon"
    return "multi-with-main" if "main" in roles else "multi-no-main"


def ordered(counter, global_counts):
    return [
        item
        for item, _ in sorted(
            counter.items(),
            key=lambda kv: (-kv[1], -global_counts.get(kv[0], 0), kv[0]),
        )
    ]


def normalize_counter(counter):
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in counter.items()}


def split_train_validation(train):
    train = train.copy()
    train["timestamp"] = pd.to_datetime(train["timestamp"], errors="coerce")
    train = train.dropna(subset=["timestamp", "item_ids"]).sort_values("timestamp")
    split_idx = int(len(train) * 0.80)
    train_part = train.iloc[:split_idx].copy()
    val_part = train.iloc[split_idx:].copy()
    return train_part, val_part


def deterministic_hidden_item(transaction_id, items):
    items = sorted(set(items))
    if not items:
        return None
    seed = sum(ord(ch) for ch in str(transaction_id))
    return items[seed % len(items)]


def build_validation_rows(val_part, valid_items, role_map, category_map):
    rows = []
    for row in val_part.itertuples(index=False):
        items = [x for x in unique_context(parse_items(row.item_ids)) if x in valid_items]
        if not items:
            continue
        hidden = deterministic_hidden_item(row.transaction_id, items)
        if hidden is None:
            continue
        context = tuple(x for x in items if x != hidden)
        rows.append(
            {
                "instance_id": str(row.transaction_id),
                "user_id": str(row.user_id),
                "timestamp": row.timestamp,
                "context": context,
                "hidden": hidden,
                "hidden_role": role_map.get(hidden, "dessert_addon"),
                "hidden_category": category_map.get(hidden, "unknown"),
                "segment": segment_for_context(context, role_map),
                "meal_period": str(row.meal_period),
                "hour": int(row.hour) if not pd.isna(row.hour) else -1,
            }
        )
    return rows


def build_stats(train_df, valid_items, role_map):
    ctx_counts = defaultdict(Counter)
    ctx_role_counts = defaultdict(Counter)
    subset_support = Counter()
    size_counts = defaultdict(Counter)
    size_role_counts = defaultdict(Counter)
    single_counts = Counter()
    global_counts = Counter()
    global_role_counts = Counter()
    user_item_counts = defaultdict(Counter)
    user_role_counts = defaultdict(Counter)
    meal_item_counts = defaultdict(Counter)
    meal_role_counts = defaultdict(Counter)
    segment_role_counts = defaultdict(Counter)
    visible_role_to_hidden_role = defaultdict(Counter)
    baskets = []

    for row in train_df.itertuples(index=False):
        basket = tuple(x for x in unique_context(parse_items(row.item_ids)) if x in valid_items)
        if not basket:
            continue
        bset = set(basket)
        bsize = len(bset)
        user = str(row.user_id)
        meal = str(row.meal_period)
        baskets.append(basket)

        for item in bset:
            role = role_map.get(item, "dessert_addon")
            global_counts[item] += 1
            global_role_counts[role] += 1
            size_counts[bsize][item] += 1
            size_role_counts[bsize][role] += 1
            user_item_counts[user][item] += 1
            user_role_counts[user][role] += 1
            meal_item_counts[meal][item] += 1
            meal_role_counts[meal][role] += 1
        if bsize == 1:
            single_counts[next(iter(bset))] += 1

        for r in range(0, bsize + 1):
            for sub in combinations(basket, r):
                subset_support[sub] += 1

        for target in bset:
            context = tuple(sorted(bset - {target}))
            target_role = role_map.get(target, "dessert_addon")
            segment = segment_for_context(context, role_map)
            visible_roles = tuple(sorted(role_map.get(x, "dessert_addon") for x in context))
            ctx_counts[context][target] += 1
            ctx_role_counts[context][target_role] += 1
            segment_role_counts[segment][target_role] += 1
            visible_role_to_hidden_role[visible_roles][target_role] += 1

    return {
        "ctx_counts": ctx_counts,
        "ctx_totals": {ctx: sum(cnt.values()) for ctx, cnt in ctx_counts.items()},
        "ctx_role_counts": ctx_role_counts,
        "ctx_role_totals": {ctx: sum(cnt.values()) for ctx, cnt in ctx_role_counts.items()},
        "subset_support": subset_support,
        "size_counts": size_counts,
        "size_totals": {sz: sum(cnt.values()) for sz, cnt in size_counts.items()},
        "size_role_counts": size_role_counts,
        "single_counts": single_counts,
        "single_total": sum(single_counts.values()),
        "global_counts": global_counts,
        "global_total": sum(global_counts.values()),
        "global_role_counts": global_role_counts,
        "global_role_probs": normalize_counter(global_role_counts),
        "global_order": ordered(global_counts, global_counts),
        "single_order": ordered(single_counts, global_counts),
        "user_item_counts": user_item_counts,
        "user_role_counts": user_role_counts,
        "meal_item_counts": meal_item_counts,
        "meal_role_counts": meal_role_counts,
        "segment_role_counts": segment_role_counts,
        "visible_role_to_hidden_role": visible_role_to_hidden_role,
        "n_baskets": len(baskets),
    }


def rule_metrics(subset, item, stats):
    subset_support = stats["subset_support"]
    n_baskets = stats["n_baskets"]
    denom = subset_support.get(subset, 0)
    if denom <= 0:
        return 0.0, 0.0, 0.0, 0

    joint = tuple(sorted(set(subset) | {item}))
    joint_count = subset_support.get(joint, 0)
    if joint_count <= 0:
        return 0.0, 0.0, 0.0, 0

    conf = joint_count / denom
    p_joint = joint_count / n_baskets
    p_subset = denom / n_baskets
    p_item = subset_support.get((item,), 0) / n_baskets
    if p_item <= 0 or p_subset <= 0:
        return conf, 0.0, 0.0, joint_count

    lift = conf / p_item
    pmi = math.log(p_joint / (p_subset * p_item)) if p_joint > 0 else 0.0
    npmi = pmi / (-math.log(p_joint)) if p_joint > 0 and p_joint < 1 else 0.0
    return conf, lift, npmi, joint_count


def weighted_subset_signal(context, item, stats):
    if not context:
        return 0.0, 0.0, 0.0
    total_weight = 0.0
    conf_sum = 0.0
    lift_sum = 0.0
    npmi_sum = 0.0
    full_len = len(context)
    for subset in powerset_nonempty(context):
        if item in subset:
            continue
        support = stats["subset_support"].get(subset, 0)
        if support <= 0:
            continue
        conf, lift, npmi, joint_count = rule_metrics(subset, item, stats)
        if joint_count < MIN_ASSOC_JOINT or conf < MIN_ASSOC_CONF:
            continue
        rel_size = len(subset) / full_len
        reliability = math.log1p(joint_count)
        weight = (rel_size ** 2.0) * reliability
        total_weight += weight
        conf_sum += weight * conf
        lift_sum += weight * min(MAX_LIFT_BONUS, max(0.0, math.log1p(max(0.0, lift - 1.0))))
        npmi_sum += weight * min(MAX_NPMI_BONUS, max(0.0, npmi))
    if total_weight <= 0:
        return 0.0, 0.0, 0.0
    return conf_sum / total_weight, lift_sum / total_weight, npmi_sum / total_weight


def subset_role_probs(context, stats, role_map):
    if not context:
        return {}
    role_scores = Counter()
    total_weight = 0.0
    full_len = len(context)
    for subset in powerset_nonempty(context):
        denom = stats["subset_support"].get(subset, 0)
        if denom <= 0:
            continue
        rel_size = len(subset) / full_len
        weight = rel_size ** 2.0 * math.log1p(denom)
        if weight <= 0:
            continue
        for item in stats["global_counts"]:
            if item in context:
                continue
            joint = tuple(sorted(set(subset) | {item}))
            joint_count = stats["subset_support"].get(joint, 0)
            if joint_count < MIN_ASSOC_JOINT:
                continue
            conf = joint_count / denom
            if conf < MIN_ASSOC_CONF:
                continue
            role_scores[role_map.get(item, "dessert_addon")] += weight * conf
        total_weight += weight
    if total_weight <= 0:
        return {}
    total = sum(role_scores.values())
    if total <= 0:
        return {}
    return {role: val / total for role, val in role_scores.items()}


def exact_context_recs(context, stats, valid_items):
    current = set(context)
    full_size = len(current) + 1
    if len(current) == 0:
        sources = [stats["single_order"], stats["global_order"]]
    else:
        sources = [
            ordered(stats["ctx_counts"].get(tuple(sorted(context)), Counter()), stats["global_counts"]),
            ordered(stats["size_counts"].get(full_size, Counter()), stats["global_counts"]),
            stats["global_order"],
        ]
    out = []
    seen = set()
    for source in sources:
        for item in source:
            if item in seen or item in current or item not in valid_items:
                continue
            seen.add(item)
            out.append(item)
            if len(out) == 10:
                return out
    return out


def item_score_assoc(context, item, stats, base_rank=None, weights=None):
    if weights is None:
        weights = ASSOC_WEIGHTS
    current = set(context)
    full_size = len(current) + 1
    ctx = tuple(sorted(context))
    ctx_counter = stats["ctx_counts"].get(ctx, Counter())
    ctx_total = stats["ctx_totals"].get(ctx, 0)
    size_counter = stats["size_counts"].get(full_size, Counter())
    size_total = stats["size_totals"].get(full_size, 0)
    exact_conf = (ctx_counter.get(item, 0) / ctx_total) if ctx_total else 0.0
    subset_conf, lift_bonus, npmi_bonus = weighted_subset_signal(context, item, stats)
    size_prior = (size_counter.get(item, 0) / size_total) if size_total else 0.0
    base_prior = 1.0 / base_rank[item] if base_rank and item in base_rank else 0.0
    if not current:
        subset_conf = 0.0
        lift_bonus = 0.0
        npmi_bonus = 0.0
    return (
        weights["exact_conf"] * exact_conf
        + weights["subset_conf"] * subset_conf
        + weights["lift"] * lift_bonus
        + weights["npmi"] * npmi_bonus
        + weights["size_prior"] * size_prior
        + weights.get("base_prior", 0.0) * base_prior
    )


def assoc_recs(context, stats, valid_items, base_items=None):
    base_rank = {item: pos for pos, item in enumerate(base_items or [], start=1)}
    current = set(context)
    ctx = tuple(sorted(context))
    ctx_counter = stats["ctx_counts"].get(ctx, Counter())
    ctx_total = stats["ctx_totals"].get(ctx, 0)

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
        scored.append((item, item_score_assoc(context, item, stats, base_rank, ASSOC_WEIGHTS)))
    return [
        item
        for item, _ in sorted(
            scored,
            key=lambda kv: (
                -exact_bucket(kv[0]),
                -kv[1],
                -ctx_counter.get(kv[0], 0),
                -stats["global_counts"].get(kv[0], 0),
                kv[0],
            ),
        )[:10]
    ]


def role_distribution(context, user_id, meal_period, stats, role_map):
    ctx = tuple(sorted(context))
    segment = segment_for_context(context, role_map)
    visible_roles = tuple(sorted(role_map.get(x, "dessert_addon") for x in context))
    pieces = []

    def add_piece(weight, probs):
        if probs:
            pieces.append((weight, probs))

    add_piece(ROLE_SOURCES["exact"], normalize_counter(stats["ctx_role_counts"].get(ctx, Counter())))
    add_piece(ROLE_SOURCES["subset"], subset_role_probs(context, stats, role_map))
    add_piece(ROLE_SOURCES["segment"], normalize_counter(stats["segment_role_counts"].get(segment, Counter())))
    add_piece(ROLE_SOURCES["segment"], normalize_counter(stats["visible_role_to_hidden_role"].get(visible_roles, Counter())))
    add_piece(ROLE_SOURCES["user"], normalize_counter(stats["user_role_counts"].get(str(user_id), Counter())))
    add_piece(ROLE_SOURCES["meal"], normalize_counter(stats["meal_role_counts"].get(str(meal_period), Counter())))
    add_piece(ROLE_SOURCES["global"], stats["global_role_probs"])

    out = Counter()
    total_weight = 0.0
    for weight, probs in pieces:
        total_weight += weight
        for role, prob in probs.items():
            out[role] += weight * prob
    if total_weight <= 0:
        return dict(stats["global_role_probs"])
    return {role: out[role] / total_weight for role in stats["global_role_probs"]}


def level3_recs(context, user_id, meal_period, stats, valid_items, role_map, role_weight, base_items=None):
    current = set(context)
    base_rank = {item: pos for pos, item in enumerate(base_items or [], start=1)}
    role_probs = role_distribution(context, user_id, meal_period, stats, role_map)
    predicted_role = max(role_probs.items(), key=lambda kv: (kv[1], kv[0]))[0]
    full_size = len(current) + 1
    ctx = tuple(sorted(context))
    ctx_counter = stats["ctx_counts"].get(ctx, Counter())
    ctx_total = stats["ctx_totals"].get(ctx, 0)
    size_counter = stats["size_counts"].get(full_size, Counter())
    size_total = stats["size_totals"].get(full_size, 0)
    user_counter = stats["user_item_counts"].get(str(user_id), Counter())
    user_total = sum(user_counter.values())
    meal_counter = stats["meal_item_counts"].get(str(meal_period), Counter())
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
        exact_conf = (ctx_counter.get(item, 0) / ctx_total) if ctx_total else 0.0
        subset_conf, lift_bonus, npmi_bonus = weighted_subset_signal(context, item, stats)
        size_prior = (size_counter.get(item, 0) / size_total) if size_total else 0.0
        user_item = (user_counter.get(item, 0) / user_total) if user_total else 0.0
        meal_item = (meal_counter.get(item, 0) / meal_total) if meal_total else 0.0
        base_prior = 1.0 / base_rank[item] if item in base_rank else 0.0
        if not context:
            subset_conf = 0.0
            lift_bonus = 0.0
            npmi_bonus = 0.0
        role = role_map.get(item, "dessert_addon")
        role_prob = role_probs.get(role, 0.0)
        score = (
            LEVEL3_BASE_WEIGHTS["exact_conf"] * exact_conf
            + LEVEL3_BASE_WEIGHTS["subset_conf"] * subset_conf
            + LEVEL3_BASE_WEIGHTS["lift"] * lift_bonus
            + LEVEL3_BASE_WEIGHTS["npmi"] * npmi_bonus
            + LEVEL3_BASE_WEIGHTS["size_prior"] * size_prior
            + LEVEL3_BASE_WEIGHTS["user_item"] * user_item
            + LEVEL3_BASE_WEIGHTS["meal_item"] * meal_item
            + LEVEL3_BASE_WEIGHTS["base_prior"] * base_prior
            + role_weight * role_prob
        )
        scored.append((item, score, role_prob))

    recs = [
        item
        for item, _, _ in sorted(
            scored,
            key=lambda kv: (
                -exact_bucket(kv[0]),
                -kv[1],
                -kv[2],
                -ctx_counter.get(kv[0], 0),
                -stats["global_counts"].get(kv[0], 0),
                kv[0],
            ),
        )[:10]
    ]
    return recs, predicted_role


def evaluate_rows(rows, predictor, role_map):
    ndcgs = []
    hits = {1: [], 3: [], 5: [], 10: []}
    rank1 = []
    role_ok = []
    by_segment = defaultdict(list)
    by_role = defaultdict(list)
    by_category = defaultdict(list)
    seg_counts = Counter()
    role_counts = Counter()
    category_counts = Counter()

    for row in rows:
        recs, pred_role = predictor(row)
        target = row["hidden"]
        ndcg = ndcg_at10(recs, target)
        ndcgs.append(ndcg)
        for k in hits:
            hits[k].append(hit_at(recs, target, k))
        rank1.append(int(recs and recs[0] == target))
        role_ok.append(int(pred_role == row["hidden_role"]))
        by_segment[row["segment"]].append(ndcg)
        by_role[row["hidden_role"]].append(ndcg)
        by_category[row["hidden_category"]].append(ndcg)
        seg_counts[row["segment"]] += 1
        role_counts[row["hidden_role"]] += 1
        category_counts[row["hidden_category"]] += 1

    def mean(vals):
        return float(sum(vals) / len(vals)) if vals else 0.0

    return {
        "n": len(rows),
        "overall_ndcg": mean(ndcgs),
        "hit_at": {str(k): mean(v) for k, v in hits.items()},
        "rank1_accuracy": mean(rank1),
        "role_prediction_accuracy": mean(role_ok),
        "ndcg_by_segment": {k: mean(by_segment[k]) for k in sorted(by_segment)},
        "ndcg_by_role": {k: mean(by_role[k]) for k in sorted(by_role)},
        "ndcg_by_category": {k: mean(by_category[k]) for k in sorted(by_category)},
        "counts_by_segment": dict(seg_counts),
        "counts_by_role": dict(role_counts),
        "counts_by_category": dict(category_counts),
    }


def generate_candidate(label, base_file, out_file, stats, test_df, menu, role_map, valid_items, role_weight):
    base_path = os.path.join(BASE, base_file)
    base_df = pd.read_csv(base_path)
    if list(base_df.columns) != EXPECTED_COLS:
        raise ValueError(f"{base_file} has unexpected columns: {list(base_df.columns)}")

    out_df = base_df.copy()
    changed = 0
    basket_viol = 0
    invalid = 0
    dup_rows = 0
    for idx, row in base_df.iterrows():
        test_row = test_df.iloc[idx]
        iid = str(row["instance_id"])
        if iid != str(test_row["instance_id"]):
            test_row = test_df[test_df["instance_id"].astype(str) == iid].iloc[0]
        context = unique_context(parse_items(test_row["current_basket_items"]))
        base_items = [str(row[col]) for col in REC_COLS]
        recs, _ = level3_recs(
            context,
            str(test_row["user_id"]),
            str(test_row["meal_period"]),
            stats,
            valid_items,
            role_map,
            role_weight,
            base_items=base_items,
        )
        if len(recs) != 10:
            raise RuntimeError(f"Could not fill 10 recommendations for {iid}: {recs}")
        if recs != base_items:
            changed += 1
        if len(set(recs)) < len(recs):
            dup_rows += 1
        invalid += sum(1 for item in recs if item not in valid_items)
        if set(recs) & set(context):
            basket_viol += 1
        for col, item in zip(REC_COLS, recs):
            out_df.at[idx, col] = item

    out_df.to_csv(os.path.join(BASE, out_file), index=False)
    return {
        "label": label,
        "base_file": base_file,
        "output_file": out_file,
        "changed_rows": int(changed),
        "duplicate_recommendation_rows": int(dup_rows),
        "invalid_item_ids": int(invalid),
        "current_basket_violations": int(basket_viol),
    }


def fmt_metric(x):
    return f"{x:.6f}"


def table_from_dict(title, data, counts=None):
    lines = [f"## {title}", "", "| slice | NDCG@10 | count |", "|---|---:|---:|"]
    for key in sorted(data):
        count = counts.get(key, "") if counts else ""
        lines.append(f"| {key} | {fmt_metric(data[key])} | {count} |")
    lines.append("")
    return lines


def write_report(payload):
    best = payload["best_level3"]
    assoc = payload["baselines"]["basket_completion_assoc_oldbase_proxy"]
    exact = payload["baselines"]["exact_context_blend_oldbase_proxy"]
    lines = [
        "# Level 3 Role Basket Completion Diagnosis",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "## Decision",
        "",
        f"**{payload['decision']}**",
        "",
        payload["decision_reason"],
        "",
        "## Validation Setup",
        "",
        f"- Train split rows: {payload['split']['train_rows']}",
        f"- Validation baskets: {payload['split']['validation_rows']}",
        f"- Split time max train: {payload['split']['train_max_timestamp']}",
        f"- Split time min validation: {payload['split']['validation_min_timestamp']}",
        "- Validation target: remove exactly one deterministic item from each validation basket.",
        "",
        "## Overall Comparison",
        "",
        "| method | NDCG@10 | Hit@1 | Hit@3 | Hit@5 | Hit@10 | role acc | rank_1 acc |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for name, metrics in [
        ("exact_context_blend_oldbase_proxy", exact),
        ("basket_completion_assoc_oldbase_proxy", assoc),
        ("level3_role_best", best),
    ]:
        lines.append(
            "| "
            + name
            + " | "
            + " | ".join(
                [
                    fmt_metric(metrics["overall_ndcg"]),
                    fmt_metric(metrics["hit_at"]["1"]),
                    fmt_metric(metrics["hit_at"]["3"]),
                    fmt_metric(metrics["hit_at"]["5"]),
                    fmt_metric(metrics["hit_at"]["10"]),
                    fmt_metric(metrics["role_prediction_accuracy"]),
                    fmt_metric(metrics["rank1_accuracy"]),
                ]
            )
            + " |"
        )

    lines += [
        "",
        f"Best role weight: `{payload['best_role_weight']}`",
        f"Delta vs exact proxy: `{fmt_metric(best['overall_ndcg'] - exact['overall_ndcg'])}`",
        f"Delta vs assoc proxy: `{fmt_metric(best['overall_ndcg'] - assoc['overall_ndcg'])}`",
        f"Rank_1 delta vs assoc proxy: `{fmt_metric(best['rank1_accuracy'] - assoc['rank1_accuracy'])}`",
        "",
    ]

    lines += table_from_dict("Level 3 NDCG by Segment", best["ndcg_by_segment"], best["counts_by_segment"])
    lines += table_from_dict("Level 3 NDCG by Role", best["ndcg_by_role"], best["counts_by_role"])
    lines += table_from_dict("Level 3 NDCG by Category", best["ndcg_by_category"], best["counts_by_category"])

    lines += [
        "## Role Weight Sweep",
        "",
        "| role_weight | NDCG@10 | Hit@1 | role acc | rank_1 acc | delta vs assoc |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["role_weight_sweep"]:
        lines.append(
            f"| {row['role_weight']} | {fmt_metric(row['overall_ndcg'])} | "
            f"{fmt_metric(row['hit_at']['1'])} | {fmt_metric(row['role_prediction_accuracy'])} | "
            f"{fmt_metric(row['rank1_accuracy'])} | {fmt_metric(row['overall_ndcg'] - assoc['overall_ndcg'])} |"
        )

    lines += ["", "## Candidate CSV Generation", ""]
    if payload["generated_candidates"]:
        lines.append("Generated because validation gates passed:")
        lines.append("")
        for item in payload["generated_candidates"]:
            lines.append(
                f"- `{item['output_file']}` from `{item['base_file']}`: changed_rows={item['changed_rows']}, "
                f"dups={item['duplicate_recommendation_rows']}, invalid={item['invalid_item_ids']}, "
                f"basket_violations={item['current_basket_violations']}"
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
    category_map, role_map = make_menu_maps(menu)

    print("Building temporal validation split...")
    train_part, val_part = split_train_validation(train)
    val_rows = build_validation_rows(val_part, valid_items, role_map, category_map)
    print(f"  train rows: {len(train_part):,}")
    print(f"  validation rows: {len(val_rows):,}")

    print("Building train-only stats...")
    stats = build_stats(train_part, valid_items, role_map)

    print("Evaluating exact-context proxy...")
    exact_metrics = evaluate_rows(
        val_rows,
        lambda r: (
            exact_context_recs(r["context"], stats, valid_items),
            max(stats["global_role_probs"].items(), key=lambda kv: (kv[1], kv[0]))[0],
        ),
        role_map,
    )

    print("Evaluating association proxy...")
    assoc_metrics = evaluate_rows(
        val_rows,
        lambda r: (
            assoc_recs(r["context"], stats, valid_items),
            max(role_distribution(r["context"], r["user_id"], r["meal_period"], stats, role_map).items(), key=lambda kv: (kv[1], kv[0]))[0],
        ),
        role_map,
    )

    print("Sweeping Level 3 role weights...")
    sweep = []
    best_weight = None
    best_metrics = None
    for role_weight in ROLE_WEIGHT_GRID:
        metrics = evaluate_rows(
            val_rows,
            lambda r, rw=role_weight: level3_recs(
                r["context"],
                r["user_id"],
                r["meal_period"],
                stats,
                valid_items,
                role_map,
                rw,
            ),
            role_map,
        )
        metrics["role_weight"] = role_weight
        sweep.append(metrics)
        print(
            f"  role_weight={role_weight:.2f} ndcg={metrics['overall_ndcg']:.6f} "
            f"hit1={metrics['hit_at']['1']:.6f} role_acc={metrics['role_prediction_accuracy']:.6f}"
        )
        if best_metrics is None or metrics["overall_ndcg"] > best_metrics["overall_ndcg"]:
            best_weight = role_weight
            best_metrics = metrics

    delta_assoc = best_metrics["overall_ndcg"] - assoc_metrics["overall_ndcg"]
    rank1_delta_assoc = best_metrics["rank1_accuracy"] - assoc_metrics["rank1_accuracy"]
    role_improved_over_global = best_metrics["role_prediction_accuracy"] > exact_metrics["role_prediction_accuracy"]
    gates_pass = (
        best_weight > 0
        and delta_assoc >= 0.0005
        and rank1_delta_assoc >= -0.0001
        and role_improved_over_global
    )

    decision = "GO" if gates_pass else "NO-GO"
    if gates_pass:
        reason = (
            "Level 3 improved validation versus the association proxy, selected a positive role weight, "
            "and did not materially regress rank_1 accuracy."
        )
    else:
        reason = (
            "Role/category prediction did not clear the validation gates. Per instruction, no candidate CSV "
            "should be generated unless Level 3 improves validation."
        )

    generated = []
    if gates_pass:
        print("Validation gates passed. Building full-train candidate CSVs...")
        full_stats = build_stats(train, valid_items, role_map)
        generated.append(generate_candidate("oldbase", OLD_BASE, OLD_OUT, full_stats, test, menu, role_map, valid_items, best_weight))
        generated.append(generate_candidate("v7base", V7_BASE, V7_OUT, full_stats, test, menu, role_map, valid_items, best_weight))
    else:
        print("Validation gates failed. No candidate CSVs will be generated.")

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": time.time() - t0,
        "split": {
            "train_rows": int(len(train_part)),
            "validation_rows": int(len(val_rows)),
            "train_max_timestamp": str(train_part["timestamp"].max()),
            "validation_min_timestamp": str(val_part["timestamp"].min()),
        },
        "baselines": {
            "exact_context_blend_oldbase_proxy": exact_metrics,
            "basket_completion_assoc_oldbase_proxy": assoc_metrics,
        },
        "best_role_weight": best_weight,
        "best_level3": best_metrics,
        "role_weight_sweep": sweep,
        "decision": decision,
        "decision_reason": reason,
        "generated_candidates": generated,
    }

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    write_report(payload)

    print(f"\nDecision: {decision}")
    print(reason)
    print(f"Report: {REPORT_PATH}")
    print(f"JSON: {JSON_PATH}")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
