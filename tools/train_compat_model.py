import argparse
import json
import os
import pickle
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def find_latest_embedding(output_dir: str) -> str:
    cand = []
    for fn in os.listdir(output_dir):
        if fn.startswith("FlavorGraph+CSL-embedding_") and fn.endswith(".pickle"):
            cand.append((os.path.getmtime(os.path.join(output_dir, fn)), fn))
    if not cand:
        raise FileNotFoundError("No embedding file found in output/")
    cand.sort(reverse=True)
    return os.path.join(output_dir, cand[0][1])


def load_embeddings(path: str):
    with open(path, "rb") as f:
        emb = pickle.load(f)
    # keys are node_id strings; values are numpy arrays
    return emb


def load_nodes(nodes_csv: str):
    df = pd.read_csv(nodes_csv)
    # Columns: node_id,name,id,node_type,is_hub
    id_to_name = {str(r.node_id): r.name for _, r in df.iterrows()}
    id_to_type = {str(r.node_id): r.node_type for _, r in df.iterrows()}
    name_to_id = {r.name: str(r.node_id) for _, r in df.iterrows()}
    return id_to_name, id_to_type, name_to_id


def unit_norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v) + 1e-12
    return v / n


def pair_features(e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    e1 = unit_norm(e1)
    e2 = unit_norm(e2)
    # Use absolute difference as feature; simple and effective for metric learning proxies
    return np.abs(e1 - e2)


def build_dataset(seed_path: str, embeddings: dict, id_to_type: dict, max_neg_ratio: float = 1.0):
    # Collect ingredients present in seed
    seed = [json.loads(l) for l in open(seed_path)]
    present_ids = set()
    for rec in seed:
        for ing in rec.get("ingredients", []):
            nid = str(ing.get("node_id", ""))
            if nid and nid in embeddings and id_to_type.get(nid) == "ingredient":
                present_ids.add(nid)

    # Positives: all pairs within each recipe
    X, y = [], []
    pos_pairs = set()
    for rec in seed:
        ids = [str(i.get("node_id")) for i in rec.get("ingredients", []) if str(i.get("node_id")) in embeddings]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                if a in embeddings and b in embeddings:
                    f = pair_features(embeddings[a], embeddings[b])
                    X.append(f)
                    y.append(1)
                    pos_pairs.add(tuple(sorted((a, b))))

    # Negatives: sample random ingredient pairs not in positives
    ingredient_pool = [nid for nid in embeddings.keys() if id_to_type.get(nid) == "ingredient"]
    random.shuffle(ingredient_pool)
    num_negs = int(len(X) * max_neg_ratio)
    tries = 0
    while len(y) - len(pos_pairs) < num_negs and tries < num_negs * 20:
        a, b = random.sample(ingredient_pool, 2)
        if tuple(sorted((a, b))) in pos_pairs:
            tries += 1
            continue
        f = pair_features(embeddings[a], embeddings[b])
        X.append(f)
        y.append(0)
        tries += 1

    X = np.stack(X)
    y = np.array(y)
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default="data/beverage_seed_carbonated_IN.jsonl")
    parser.add_argument("--nodes", default="input/nodes_191120.csv")
    parser.add_argument("--emb", default=None)
    parser.add_argument("--out", default="models/compat_beverage_IN.pkl")
    args = parser.parse_args()

    emb_path = args.emb or find_latest_embedding("output")
    os.makedirs(Path(args.out).parent, exist_ok=True)

    embeddings = load_embeddings(emb_path)
    id_to_name, id_to_type, name_to_id = load_nodes(args.nodes)

    X, y = build_dataset(args.seed, embeddings, id_to_type)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    clf = LogisticRegression(max_iter=200, class_weight="balanced")
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    try:
        auc = roc_auc_score(y_test, proba)
    except Exception:
        auc = float("nan")

    meta = {
        "emb_path": emb_path,
        "feature": "abs_diff_l2norm",
        "auc": auc,
        "num_train": int(len(X_train)),
        "num_test": int(len(X_test))
    }
    joblib.dump({"model": clf, "meta": meta}, args.out)
    print(f"Saved compatibility model → {args.out} | AUC={auc:.3f}")


if __name__ == "__main__":
    main()


