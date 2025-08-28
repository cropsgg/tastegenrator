import argparse
import json
import os
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from validate_beverage import load_constraints, validate_record


def find_latest_embedding(output_dir: str) -> str:
    cand = []
    for fn in os.listdir(output_dir):
        if fn.startswith("FlavorGraph+CSL-embedding_") and fn.endswith(".pickle"):
            cand.append((os.path.getmtime(os.path.join(output_dir, fn)), fn))
    if not cand:
        raise FileNotFoundError("No embedding found")
    cand.sort(reverse=True)
    return os.path.join(output_dir, cand[0][1])


def load_embeddings(path: str):
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def load_nodes(nodes_csv: str):
    df = pd.read_csv(nodes_csv)
    id_to_name = {str(r.node_id): r.name for _, r in df.iterrows()}
    id_to_type = {str(r.node_id): r.node_type for _, r in df.iterrows()}
    # case-insensitive map, prefer last occurrence
    name_to_id = {}
    for _, r in df.iterrows():
        name_to_id[str(r.name).strip().lower()] = str(r.node_id)
    return id_to_name, id_to_type, name_to_id


def unit_norm(v):
    n = np.linalg.norm(v) + 1e-12
    return v / n


def pair_features(e1, e2):
    e1 = unit_norm(e1)
    e2 = unit_norm(e2)
    return np.abs(e1 - e2)


def score_set(ings, embeddings, clf):
    # score = average pairwise compatibility
    if len(ings) < 2:
        return 0.0
    scores = []
    for i in range(len(ings)):
        for j in range(i + 1, len(ings)):
            a, b = ings[i], ings[j]
            f = pair_features(embeddings[a], embeddings[b]).reshape(1, -1)
            s = clf.predict_proba(f)[0, 1]
            scores.append(s)
    return float(np.mean(scores)) if scores else 0.0


def propose_base_pool(id_to_type, name_to_id):
    # limit to a reasonable beverage pool
    base_names = [
        "water", "sugar", "citric_acid", "lemon", "lime", "black_salt", "cumin", "ginger",
        "stevia"
    ]
    pool = [name_to_id[n.lower()] for n in base_names if n.lower() in name_to_id]
    return [nid for nid in pool if id_to_type.get(nid) == "ingredient"]


def amounts_for_profile(names):
    # simple heuristics for demo
    total_mL = 1000.0
    has_sugar = "sugar" in names
    has_stevia = "stevia" in names
    if has_sugar:
        brix = 10.0
        sugar_g = brix * 10.0  # approx for demo
    else:
        brix = 1.0
        sugar_g = 0.0
    acid_g = 1.8 if "citric_acid" in names else 0.0
    flavor_mL = 6.0 if ("lemon" in names or "lime" in names) else 0.0
    water_mL = total_mL - flavor_mL
    return {
        "brix_percent": brix,
        "pH": 3.2,
        "ta_g_L_as_citric": 3.5,
        "co2_volumes": 3.0
    }, {
        "water": (water_mL, "mL"),
        "sugar": (sugar_g, "g"),
        "citric_acid": (acid_g, "g"),
        "lemon": (flavor_mL, "mL"),
        "lime": (0.0, "mL"),
        "black_salt": (0.3 if "black_salt" in names else 0.0, "g"),
        "cumin": (0.2 if "cumin" in names else 0.0, "g"),
        "stevia": (0.08 if has_stevia else 0.0, "g")
    }


def resolve_node_id(preferred_names, name_to_id, id_to_name):
    # try exact (case-insensitive), then contains
    for n in preferred_names:
        key = n.strip().lower()
        if key in name_to_id:
            return name_to_id[key], n
    # contains search fallback
    # build reverse: lower name -> id already exists in name_to_id; but we need list of names
    # We'll iterate id_to_name
    for nid, nm in id_to_name.items():
        low = str(nm).strip().lower()
        if any(p in low for p in [p.strip().lower() for p in preferred_names]):
            return nid, nm
    return None, preferred_names[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", default="input/nodes_191120.csv")
    parser.add_argument("--emb", default=None)
    parser.add_argument("--model", default="models/compat_beverage_IN.pkl")
    parser.add_argument("--constraints", default="config/constraints/fssai_carbonated_beverage_constraints.json")
    args = parser.parse_args()

    emb_path = args.emb or find_latest_embedding("output")
    embeddings = load_embeddings(emb_path)
    id_to_name, id_to_type, name_to_id = load_nodes(args.nodes)
    clf = joblib.load(args.model)["model"]
    rules = load_constraints(args.constraints)

    pool = propose_base_pool(id_to_type, name_to_id)
    # Start with water + acid + citrus; greedily add others if score improves and validation passes
    water_id, water_name = resolve_node_id(["water", "carbonated_water", "bottled_water", "distilled_water"], name_to_id, id_to_name)
    acid_id, acid_name = resolve_node_id(["citric_acid"], name_to_id, id_to_name)
    citrus_id, citrus_name = resolve_node_id(["lemon", "lime"], name_to_id, id_to_name)
    start = [nid for nid in [water_id, acid_id, citrus_id] if nid]
    current = [nid for nid in start if nid in embeddings]
    best_score = score_set(current, embeddings, clf)

    improved = True
    while improved:
        improved = False
        for cand in pool:
            if cand in current:
                continue
            trial = current + [cand]
            s = score_set(trial, embeddings, clf)
            if s > best_score:
                # build a rec for validation
                names = [id_to_name[nid] for nid in trial]
                targets, amt = amounts_for_profile([id_to_name[nid] for nid in trial])
                rec = {
                    "id": "GEN-CSD-001",
                    "name": "Generated Carbonated Beverage",
                    "country": "IN",
                    "subcategory": "carbonated",
                    "dietary_mode": "liquid",
                    "veg_flag": True,
                    "jain_flag": True,
                    "targets": targets,
                    "ingredients": [],
                    "process": {"template": "standard_csd", "steps": [{"step": s, "params": {}} for s in [
                        "prepare_syrup_mix", "filter", "in_line_blend_to_target_brix_and_acid", "chill_to_cold_fill_temperature", "carbonation_to_target_volumes", "fill_and_seal", "date_code_and_pack"
                    ]]},
                    "labels": {"veg_symbol": True, "claims": [], "warnings": []}
                }
                for k, v in amt.items():
                    if v[0] <= 0:
                        continue
                    nid, actual_name = resolve_node_id([k], name_to_id, id_to_name)
                    if nid is None:
                        continue
                    rec["ingredients"].append({
                        "name": actual_name,
                        "node_id": nid,
                        "quantity": float(v[0]),
                        "unit": v[1],
                        "class": ("water" if k=="water" else ("acid" if k=="citric_acid" else ("sweetener" if k in ["sugar","stevia"] else "flavor")))
                    })
                v = validate_record(rec, rules)
                if not v:
                    current = trial
                    best_score = s
                    improved = True

    # Finalize recipe
    names = [id_to_name[nid] for nid in current]
    targets, amt = amounts_for_profile(names)
    recipe = {
        "id": "GEN-CSD-001",
        "name": "Generated Carbonated Beverage",
        "country": "IN",
        "subcategory": "carbonated",
        "dietary_mode": "liquid",
        "veg_flag": True,
        "jain_flag": True,
        "targets": targets,
        "ingredients": [],
        "process": {"template": "standard_csd", "steps": [{"step": s, "params": {}} for s in [
            "prepare_syrup_mix", "filter", "in_line_blend_to_target_brix_and_acid", "chill_to_cold_fill_temperature", "carbonation_to_target_volumes", "fill_and_seal", "date_code_and_pack"
        ]]},
        "labels": {"veg_symbol": True, "claims": [], "warnings": []}
    }
    for k, v in amt.items():
        if v[0] <= 0:
            continue
        nid, actual_name = resolve_node_id([k], name_to_id, id_to_name)
        if nid is None:
            continue
        recipe["ingredients"].append({
            "name": actual_name,
            "node_id": nid,
            "quantity": float(v[0]),
            "unit": v[1],
            "class": ("water" if k=="water" else ("acid" if k=="citric_acid" else ("sweetener" if k in ["sugar","stevia"] else "flavor")))
        })

    print(json.dumps(recipe, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


