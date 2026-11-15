import json
import math
import os
import time
from collections import Counter, defaultdict

import pandas as pd

import build_level3_role_basket_completion as lvl3
import build_weak_slice_assoc_reranker as weak


BASE = os.path.dirname(os.path.abspath(__file__))
REC_COLS = [f"rank_{i}" for i in range(1, 11)]
EXPECTED_COLS = ["instance_id"] + REC_COLS

OUT_CSV = "candidate_ordered_context_final_oldbase.csv"
JSON_REPORT = "ordered_context_investigation_report.json"
MD_REPORT = "ordered_context_investigation_report.md"

CURRENT_BEST_VALIDATION = 0.767120188826832
BEST_WEAK_PARAMS = {"name": "graph_cat_small", "graph": 0.10, "cat": 0.10, "user": 0.0, "meal": 0.0, "subset": 0.0}

WEAK_ROLES = {"beverage", "dessert_addon"}
WEAK_CATEGORIES = {"Beverages", "Plates", "Desserts", "Appetizers"}
STRONG_CATEGORIES = {"Sauces", "Wraps"}

PARAM_GRID = [
    {"name": "ordered_exact_tiny", "exact": 0.25, "first": 0.0, "last": 0.0, "pair": 0.0, "role": 0.0, "min_sup": 8, "min_gain": 0.08},
    {"name": "ordered_pair_tiny", "exact": 0.0, "first": 0.04, "last": 0.06, "pair": 0.16, "role": 0.04, "min_sup": 10, "min_gain": 0.06},
    {"name": "ordered_combo_tiny", "exact": 0.18, "first": 0.03, "last": 0.05, "pair": 0.12, "role": 0.05, "min_sup": 10, "min_gain": 0.06},
    {"name": "ordered_combo_small", "exact": 0.30, "first": 0.05, "last": 0.08, "pair": 0.18, "role": 0.07, "min_sup": 12, "min_gain": 0.07},
    {"name": "ordered_role_small", "exact": 0.12, "first": 0.02, "last": 0.04, "pair": 0.10, "role": 0.16, "min_sup": 12, "min_gain": 0.07},
]


def parse_items(value):
    return lvl3.parse_items(value)


def role_for_item(item, role_map):
    return role_map.get(item, "dessert_addon")


def category_for_item(item, category_map):
    return category_map.get(item, "unknown")


def ndcg_at10(recs, target):
    if target not in recs[:10]:
        return 0.0
    rank = recs.index(target) + 1
    return 1.0 / math.log2(rank + 1)


def metric_summary(rows, rec_fn):
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
        recs = rec_fn(row)
        target = row["hidden"]
        ndcg = ndcg_at10(recs, target)
        ndcgs.append(ndcg)
        for k in hits:
            hits[k].append(int(target in recs[:k]))
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


def make_ordered_validation_rows(val_part, valid_items, role_map, category_map):
    rows = []
    for row in val_part.itertuples(index=False):
        raw_items = [x for x in parse_items(row.item_ids) if x in valid_items]
        seen = set()
        ordered_items = []
        for item in raw_items:
            if item not in seen:
                ordered_items.append(item)
                seen.add(item)
        if not ordered_items:
            continue
        hidden = lvl3.deterministic_hidden_item(row.transaction_id, ordered_items)
        if hidden is None:
            continue
        ordered_context = tuple(x for x in ordered_items if x != hidden)
        sorted_context = tuple(sorted(ordered_context))
        rows.append({
            "instance_id": str(row.transaction_id),
            "user_id": str(row.user_id),
            "timestamp": row.timestamp,
            "context": sorted_context,
            "ordered_context": ordered_context,
            "hidden": hidden,
            "hidden_role": role_for_item(hidden, role_map),
            "hidden_category": category_for_item(hidden, category_map),
            "segment": lvl3.segment_for_context(sorted_context, role_map),
            "meal_period": str(row.meal_period),
            "hour": int(row.hour) if not pd.isna(row.hour) else -1,
        })
    return rows


def build_ordered_stats(train_part, valid_items, role_map, category_map):
    ordered_ctx_counts = defaultdict(Counter)
    ordered_ctx_totals = Counter()
    first_counts = defaultdict(Counter)
    last_counts = defaultdict(Counter)
    adj_pair_counts = defaultdict(Counter)
    role_pattern_role_counts = defaultdict(Counter)
    role_pattern_cat_counts = defaultdict(Counter)
    order_examples = Counter()

    for row in train_part.itertuples(index=False):
        raw_items = [x for x in parse_items(row.item_ids) if x in valid_items]
        seen = set()
        basket = []
        for item in raw_items:
            if item not in seen:
                basket.append(item)
                seen.add(item)
        if not basket:
            continue
        order_examples[tuple(basket)] += 1
        bset = set(basket)
        for target in bset:
            ordered_ctx = tuple(x for x in basket if x != target)
            ordered_ctx_counts[ordered_ctx][target] += 1
            ordered_ctx_totals[ordered_ctx] += 1
            if ordered_ctx:
                first_counts[ordered_ctx[0]][target] += 1
                last_counts[ordered_ctx[-1]][target] += 1
                role_pattern = tuple(role_for_item(x, role_map) for x in ordered_ctx)
                role_pattern_role_counts[role_pattern][role_for_item(target, role_map)] += 1
                role_pattern_cat_counts[role_pattern][category_for_item(target, category_map)] += 1
                for a, b in zip(ordered_ctx, ordered_ctx[1:]):
                    adj_pair_counts[(a, b)][target] += 1

    return {
        "ordered_ctx_counts": ordered_ctx_counts,
        "ordered_ctx_totals": ordered_ctx_totals,
        "first_counts": first_counts,
        "last_counts": last_counts,
        "adj_pair_counts": adj_pair_counts,
        "role_pattern_role_counts": role_pattern_role_counts,
        "role_pattern_cat_counts": role_pattern_cat_counts,
        "order_examples": order_examples,
    }


def normalize_count(counter, item):
    total = sum(counter.values())
    if total <= 0:
        return 0.0, 0
    return counter.get(item, 0) / total, counter.get(item, 0)


def ordered_score(row, item, ordered_stats, role_map, category_map, params):
    ctx = row["ordered_context"]
    if not ctx:
        return 0.0, 0

    exact_counter = ordered_stats["ordered_ctx_counts"].get(ctx, Counter())
    exact_conf, exact_sup = normalize_count(exact_counter, item)
    first_conf, first_sup = normalize_count(ordered_stats["first_counts"].get(ctx[0], Counter()), item)
    last_conf, last_sup = normalize_count(ordered_stats["last_counts"].get(ctx[-1], Counter()), item)
    pair_conf = 0.0
    pair_sup = 0
    if len(ctx) >= 2:
        vals = []
        for pair in zip(ctx, ctx[1:]):
            conf, sup = normalize_count(ordered_stats["adj_pair_counts"].get(pair, Counter()), item)
            if sup:
                vals.append((conf, sup))
        if vals:
            pair_conf = max(v[0] for v in vals)
            pair_sup = max(v[1] for v in vals)

    role_pattern = tuple(role_for_item(x, role_map) for x in ctx)
    role_counter = ordered_stats["role_pattern_role_counts"].get(role_pattern, Counter())
    cat_counter = ordered_stats["role_pattern_cat_counts"].get(role_pattern, Counter())
    role_conf, role_sup = normalize_count(role_counter, role_for_item(item, role_map))
    cat_conf, cat_sup = normalize_count(cat_counter, category_for_item(item, category_map))
    support = max(exact_sup, first_sup, last_sup, pair_sup, role_sup, cat_sup)

    score = (
        params["exact"] * exact_conf
        + params["first"] * first_conf
        + params["last"] * last_conf
        + params["pair"] * pair_conf
        + params["role"] * max(role_conf, cat_conf)
    )
    return score, support


def unordered_recs(row, stats, valid_items):
    return lvl3.assoc_recs(row["context"], stats, valid_items)


def ordered_exact_recs(row, ordered_stats, stats, valid_items):
    current = set(row["ordered_context"])
    counter = ordered_stats["ordered_ctx_counts"].get(row["ordered_context"], Counter())
    sources = [
        [item for item, _ in counter.most_common()],
        lvl3.assoc_recs(row["context"], stats, valid_items),
        stats["global_order"],
    ]
    out, seen = [], set()
    for source in sources:
        for item in source:
            if item in current or item in seen or item not in valid_items:
                continue
            out.append(item)
            seen.add(item)
            if len(out) == 10:
                return out
    return out


def ordered_pair_recs(row, ordered_stats, stats, valid_items, role_map, category_map):
    current = set(row["ordered_context"])
    base = lvl3.assoc_recs(row["context"], stats, valid_items)
    scored = []
    params = {"exact": 0.0, "first": 0.08, "last": 0.10, "pair": 0.25, "role": 0.08, "min_sup": 1, "min_gain": 0.0}
    for item in valid_items:
        if item in current:
            continue
        oscore, sup = ordered_score(row, item, ordered_stats, role_map, category_map, params)
        base_bonus = 1.0 / (base.index(item) + 1) if item in base else 0.0
        scored.append((item, oscore + 0.02 * base_bonus, sup))
    return [item for item, _, _ in sorted(scored, key=lambda x: (-x[1], -x[2], x[0]))[:10]]


def weak_slice_base_recs(row, stats, cat_stats, valid_items, category_map, role_map):
    return weak.weak_slice_recs(row, stats, cat_stats, valid_items, category_map, role_map, BEST_WEAK_PARAMS)


def ordered_selective_recs(row, base_recs, ordered_stats, valid_items, role_map, category_map, params):
    if len(row["ordered_context"]) < 2:
        return base_recs
    if row["hidden_category"] in STRONG_CATEGORIES:
        # Validation-only guard: ordered experiments should not chase strong slices.
        pass
    current = set(row["ordered_context"])
    base_rank = {item: i for i, item in enumerate(base_recs, start=1)}
    scored = []
    for item in valid_items:
        if item in current:
            continue
        oscore, sup = ordered_score(row, item, ordered_stats, role_map, category_map, params)
        base_score = 1.0 / base_rank[item] if item in base_rank else 0.0
        targetable = role_for_item(item, role_map) in WEAK_ROLES or category_for_item(item, category_map) in WEAK_CATEGORIES
        bonus = oscore if targetable and sup >= params["min_sup"] else 0.0
        scored.append((item, base_score + bonus, oscore, sup, targetable))
    ordered = [item for item, *_ in sorted(scored, key=lambda x: (-x[1], -x[2], -x[3], x[0]))[:10]]
    if ordered == base_recs:
        return base_recs

    # Conservative change gate: only accept if a targetable item gains clear ordered evidence.
    old_top = base_recs[0] if base_recs else None
    new_top = ordered[0] if ordered else None
    if old_top != new_top:
        new_score, new_sup = ordered_score(row, new_top, ordered_stats, role_map, category_map, params)
        old_score, old_sup = ordered_score(row, old_top, ordered_stats, role_map, category_map, params) if old_top else (0.0, 0)
        if new_score - old_score < params["min_gain"] or new_sup < params["min_sup"]:
            return base_recs
        if category_for_item(old_top, category_map) in STRONG_CATEGORIES and category_for_item(new_top, category_map) not in STRONG_CATEGORIES:
            if new_score - old_score < params["min_gain"] * 1.8:
                return base_recs
    return ordered


def order_forensics(train, trans, test, menu, role_map):
    menu_order = {item: i for i, item in enumerate(menu["item_id"].astype(str))}
    category_order = {cat: i for i, cat in enumerate(menu["category"].drop_duplicates().astype(str))}
    item_category = dict(zip(menu["item_id"].astype(str), menu["category"].astype(str)))

    def rates(series):
        total = nonempty = alpha = menu_sorted = cat_sorted = role_main_first = 0
        role_pos = defaultdict(Counter)
        reverse_pairs = Counter()
        pair_hidden = defaultdict(Counter)
        for raw in series:
            items = parse_items(raw)
            if not items:
                continue
            total += 1
            if len(items) >= 2:
                nonempty += 1
                if items == sorted(items):
                    alpha += 1
                if items == sorted(items, key=lambda x: menu_order.get(x, 999)):
                    menu_sorted += 1
                if items == sorted(items, key=lambda x: category_order.get(item_category.get(x, ""), 999)):
                    cat_sorted += 1
                roles = [role_for_item(x, role_map) for x in items]
                for pos, role in enumerate(roles):
                    role_pos[pos][role] += 1
                if roles and roles[0] == "main":
                    role_main_first += 1
                if len(items) == 2:
                    reverse_pairs[(items[1], items[0])] += 1
        return {
            "rows_with_items": total,
            "rows_len_ge_2": nonempty,
            "alphabetical_rate_len_ge_2": alpha / nonempty if nonempty else 0.0,
            "menu_order_rate_len_ge_2": menu_sorted / nonempty if nonempty else 0.0,
            "category_order_rate_len_ge_2": cat_sorted / nonempty if nonempty else 0.0,
            "main_first_rate_len_ge_2": role_main_first / nonempty if nonempty else 0.0,
            "role_by_position_top": {str(k): v.most_common(5) for k, v in role_pos.items()},
        }

    grouped = trans.groupby("transaction_id")["item_id"].apply(list)
    basket_map = dict(zip(train["transaction_id"].astype(str), train["item_ids"].map(parse_items)))
    comparable = same_file_order = 0
    for tid, items in grouped.items():
        bitems = basket_map.get(str(tid))
        if bitems is None:
            continue
        comparable += 1
        if list(items) == bitems:
            same_file_order += 1

    return {
        "train_item_ids_order": rates(train["item_ids"]),
        "test_current_basket_order": rates(test["current_basket_items"]),
        "train_baskets_match_train_transactions_file_order": {
            "comparable_transactions": comparable,
            "same_order": same_file_order,
            "same_order_rate": same_file_order / comparable if comparable else 0.0,
        },
    }


def compare_reversal_distributions(train_part, valid_items):
    pair_targets = defaultdict(Counter)
    for row in train_part.itertuples(index=False):
        basket = []
        seen = set()
        for item in parse_items(row.item_ids):
            if item in valid_items and item not in seen:
                basket.append(item)
                seen.add(item)
        if len(basket) < 3:
            continue
        bset = set(basket)
        for target in bset:
            ctx = [x for x in basket if x != target]
            if len(ctx) == 2:
                pair_targets[tuple(ctx)][target] += 1
    diffs = []
    for pair, cnt in pair_targets.items():
        rev = (pair[1], pair[0])
        if rev not in pair_targets or pair >= rev:
            continue
        a = cnt
        b = pair_targets[rev]
        total_a = sum(a.values())
        total_b = sum(b.values())
        if total_a < 10 or total_b < 10:
            continue
        top_a = a.most_common(1)[0]
        top_b = b.most_common(1)[0]
        diffs.append({
            "pair": list(pair),
            "reverse": list(rev),
            "support": [total_a, total_b],
            "top": [top_a, top_b],
            "same_top": top_a[0] == top_b[0],
        })
    return {
        "reverse_pair_cases_support_ge_10": len(diffs),
        "different_top_hidden": sum(1 for d in diffs if not d["same_top"]),
        "examples": diffs[:20],
    }


def gate_decision(base_metrics, cand_metrics):
    deltas = {
        "overall": cand_metrics["overall_ndcg"] - CURRENT_BEST_VALIDATION,
        "sauces": cand_metrics["ndcg_by_category"].get("Sauces", 0.0) - base_metrics["ndcg_by_category"].get("Sauces", 0.0),
        "wraps": cand_metrics["ndcg_by_category"].get("Wraps", 0.0) - base_metrics["ndcg_by_category"].get("Wraps", 0.0),
        "beverage": cand_metrics["ndcg_by_role"].get("beverage", 0.0) - base_metrics["ndcg_by_role"].get("beverage", 0.0),
        "plates": cand_metrics["ndcg_by_category"].get("Plates", 0.0) - base_metrics["ndcg_by_category"].get("Plates", 0.0),
        "dessert_addon": cand_metrics["ndcg_by_role"].get("dessert_addon", 0.0) - base_metrics["ndcg_by_role"].get("dessert_addon", 0.0),
        "multi_with_main": cand_metrics["ndcg_by_segment"].get("multi-with-main", 0.0) - base_metrics["ndcg_by_segment"].get("multi-with-main", 0.0),
    }
    checks = {
        "overall_improves_at_least_0.0005": deltas["overall"] >= 0.0005,
        "sauce_regression_guard": deltas["sauces"] >= -0.002,
        "wrap_regression_guard": deltas["wraps"] >= -0.002,
        "beverage_neutral_or_better": deltas["beverage"] >= 0.0,
        "plates_neutral_or_better": deltas["plates"] >= 0.0,
        "dessert_addon_neutral_or_better": deltas["dessert_addon"] >= 0.0,
        "multi_with_main_neutral_or_better": deltas["multi_with_main"] >= 0.0,
    }
    return all(checks.values()), {"checks": checks, "deltas": deltas}


def audit_submission(df, test, valid_items):
    basket = {str(r.instance_id): set(parse_items(r.current_basket_items)) for r in test.itertuples(index=False)}
    dup = invalid = viol = 0
    for row in df.itertuples(index=False):
        recs = [str(getattr(row, c)) for c in REC_COLS]
        if len(set(recs)) < 10:
            dup += 1
        invalid += sum(1 for x in recs if x not in valid_items)
        if set(recs) & basket.get(str(row.instance_id), set()):
            viol += 1
    return {
        "shape_ok": list(df.shape) == [14147, 11],
        "columns_exact": list(df.columns) == EXPECTED_COLS,
        "missing": int(df.isnull().sum().sum()),
        "duplicate_instance_id": int(len(df) - df["instance_id"].nunique()),
        "duplicate_rec_rows": dup,
        "invalid_items": invalid,
        "basket_violations": viol,
    }


def generate_candidate(test, base_file, out_file, stats, cat_stats, ordered_stats, valid_items, category_map, role_map, params):
    base = pd.read_csv(os.path.join(BASE, base_file))
    out = base.copy()
    idx_map = {str(iid): idx for idx, iid in out["instance_id"].astype(str).items()}
    changed = rank1_changes = 0
    for row in test.itertuples(index=False):
        iid = str(row.instance_id)
        ordered_ctx = tuple(x for x in parse_items(row.current_basket_items) if x in valid_items)
        sorted_ctx = tuple(sorted(set(ordered_ctx)))
        pseudo = {
            "instance_id": iid,
            "user_id": str(row.user_id),
            "context": sorted_ctx,
            "ordered_context": ordered_ctx,
            "meal_period": str(row.meal_period),
            "segment": lvl3.segment_for_context(sorted_ctx, role_map),
            "hidden": None,
            "hidden_role": None,
            "hidden_category": None,
        }
        base_items = [str(out.at[idx_map[iid], c]) for c in REC_COLS]
        weak_recs = weak.weak_slice_recs(pseudo, stats, cat_stats, valid_items, category_map, role_map, BEST_WEAK_PARAMS, base_items)
        recs = ordered_selective_recs(pseudo, weak_recs, ordered_stats, valid_items, role_map, category_map, params)
        if recs != base_items:
            changed += 1
        if recs and base_items and recs[0] != base_items[0]:
            rank1_changes += 1
        for c, item in zip(REC_COLS, recs):
            out.at[idx_map[iid], c] = item
    out.to_csv(os.path.join(BASE, out_file), index=False)
    return out, {"changed_rows": changed, "rank1_changes_vs_base": rank1_changes}


def main():
    t0 = time.time()
    train = pd.read_csv(os.path.join(BASE, "train_baskets.csv"))
    trans = pd.read_csv(os.path.join(BASE, "train_transactions.csv"))
    test = pd.read_csv(os.path.join(BASE, "test_instances.csv"))
    menu = pd.read_csv(os.path.join(BASE, "menu_items.csv"))
    valid_items = set(menu["item_id"].astype(str))
    category_map, role_map = lvl3.make_menu_maps(menu)

    train_part, val_part = lvl3.split_train_validation(train)
    stats = lvl3.build_stats(train_part, valid_items, role_map)
    cat_stats = weak.category_stats(train_part, valid_items, category_map, role_map)
    ordered_stats = build_ordered_stats(train_part, valid_items, role_map, category_map)
    val_rows = make_ordered_validation_rows(val_part, valid_items, role_map, category_map)

    forensic = order_forensics(train, trans, test, menu, role_map)
    reversal = compare_reversal_distributions(train_part, valid_items)

    base_metrics = metric_summary(
        val_rows,
        lambda r: weak_slice_base_recs(r, stats, cat_stats, valid_items, category_map, role_map),
    )
    unordered_metrics = metric_summary(val_rows, lambda r: unordered_recs(r, stats, valid_items))
    ordered_exact_metrics = metric_summary(val_rows, lambda r: ordered_exact_recs(r, ordered_stats, stats, valid_items))
    ordered_pair_metrics = metric_summary(val_rows, lambda r: ordered_pair_recs(r, ordered_stats, stats, valid_items, role_map, category_map))

    sweep = []
    for params in PARAM_GRID:
        metrics = metric_summary(
            val_rows,
            lambda r, p=params: ordered_selective_recs(
                r,
                weak_slice_base_recs(r, stats, cat_stats, valid_items, category_map, role_map),
                ordered_stats,
                valid_items,
                role_map,
                category_map,
                p,
            ),
        )
        passes, gate = gate_decision(base_metrics, metrics)
        sweep.append({"params": params, "metrics": metrics, "passes": passes, "gate": gate})
        print(f"{params['name']}: ndcg={metrics['overall_ndcg']:.6f} pass={passes}")

    best = max(sweep, key=lambda x: x["metrics"]["overall_ndcg"])
    generated = None
    decision = "NO-GO"
    reason = "Ordered context did not pass hard validation gates."
    if best["passes"]:
        full_stats = lvl3.build_stats(train, valid_items, role_map)
        full_cat_stats = weak.category_stats(train, valid_items, category_map, role_map)
        full_ordered_stats = build_ordered_stats(train, valid_items, role_map, category_map)
        df, gen_info = generate_candidate(
            test,
            "candidate_weak_slice_assoc_oldbase.csv",
            OUT_CSV,
            full_stats,
            full_cat_stats,
            full_ordered_stats,
            valid_items,
            category_map,
            role_map,
            best["params"],
        )
        audit = audit_submission(df, test, valid_items)
        audit_ok = all([
            audit["shape_ok"], audit["columns_exact"], audit["missing"] == 0,
            audit["duplicate_instance_id"] == 0, audit["duplicate_rec_rows"] == 0,
            audit["invalid_items"] == 0, audit["basket_violations"] == 0,
        ])
        generated = {"file": OUT_CSV, "info": gen_info, "audit": audit}
        if audit_ok:
            decision = "GO"
            reason = "Ordered selective method passed validation gates and final audit."
        else:
            decision = "NO-GO"
            reason = "Validation gates passed, but generated CSV audit failed."

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": time.time() - t0,
        "decision": decision,
        "reason": reason,
        "split": {
            "train_rows": int(len(train_part)),
            "validation_rows": int(len(val_part)),
            "train_max_timestamp": str(pd.to_datetime(train_part["timestamp"]).max()),
            "validation_min_timestamp": str(pd.to_datetime(val_part["timestamp"]).min()),
        },
        "forensics": forensic,
        "reversal_distribution": reversal,
        "metrics": {
            "current_best_weak_slice": base_metrics,
            "unordered_assoc": unordered_metrics,
            "ordered_exact": ordered_exact_metrics,
            "ordered_pair": ordered_pair_metrics,
        },
        "best": best,
        "sweep": sweep,
        "generated": generated,
    }
    with open(os.path.join(BASE, JSON_REPORT), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=lambda x: int(x) if hasattr(x, "__int__") else str(x))

    lines = [
        "# Ordered Context Investigation",
        "",
        f"Decision: **{decision}**",
        "",
        reason,
        "",
        "## Validation",
        f"- Train rows: {len(train_part)}",
        f"- Validation rows: {len(val_part)}",
        f"- Current weak-slice NDCG: {base_metrics['overall_ndcg']:.6f}",
        f"- Ordered exact NDCG: {ordered_exact_metrics['overall_ndcg']:.6f}",
        f"- Ordered pair NDCG: {ordered_pair_metrics['overall_ndcg']:.6f}",
        f"- Best ordered selective: {best['params']['name']} = {best['metrics']['overall_ndcg']:.6f}",
        "",
        "## Gates",
    ]
    for k, v in best["gate"]["checks"].items():
        lines.append(f"- {k}: {v} ({best['gate']['deltas'].get(k.replace('_neutral_or_better', '').replace('_regression_guard', ''), '')})")
    lines.extend([
        "",
        "## Forensics",
        f"- Train alphabetical rate len>=2: {forensic['train_item_ids_order']['alphabetical_rate_len_ge_2']:.4f}",
        f"- Test alphabetical rate len>=2: {forensic['test_current_basket_order']['alphabetical_rate_len_ge_2']:.4f}",
        f"- Train basket/order transaction same-order rate: {forensic['train_baskets_match_train_transactions_file_order']['same_order_rate']:.4f}",
        f"- Reverse pair cases support>=10: {reversal['reverse_pair_cases_support_ge_10']}",
        f"- Reverse pair cases with different top hidden: {reversal['different_top_hidden']}",
    ])
    if generated:
        lines.extend(["", "## Generated", f"- {generated}"])
    with open(os.path.join(BASE, MD_REPORT), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(json.dumps({
        "decision": decision,
        "reason": reason,
        "current_best": base_metrics["overall_ndcg"],
        "ordered_exact": ordered_exact_metrics["overall_ndcg"],
        "ordered_pair": ordered_pair_metrics["overall_ndcg"],
        "best_ordered_selective": best["metrics"]["overall_ndcg"],
        "best_params": best["params"],
        "generated": generated,
        "report": os.path.join(BASE, JSON_REPORT),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
