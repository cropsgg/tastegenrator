import json
import csv
from pathlib import Path

SCHEMA_PATH = "config/schema/beverage_training_schema.json"
NODES_CSV = "input/nodes_191120.csv"
OUTPUT_JSONL = "data/beverage_seed_carbonated_IN.jsonl"


def load_nodes(nodes_csv):
    id_to_name = {}
    with open(nodes_csv, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            node_id = str(r["node_id"]) if "node_id" in r else str(r["node_id"])  # consistent string
            id_to_name[node_id] = r["name"]
    return id_to_name


def invert_nodes(nodes_csv):
    name_to_id = {}
    with open(nodes_csv, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            name_to_id[r["name"]] = str(r["node_id"])
    return name_to_id


def ensure_dirs(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def main():
    name_to_id = invert_nodes(NODES_CSV)

    # Choose common India carbonated profiles (nimbu soda, masala soda, lime soda, diet lemon, ginger-lime)
    recipes = [
        {
            "id": "IN-CSD-0001",
            "name": "Nimbu Soda (Sweetened)",
            "country": "IN",
            "subcategory": "carbonated",
            "dietary_mode": "liquid",
            "veg_flag": True,
            "jain_flag": True,
            "targets": {"brix_percent": 10.5, "pH": 3.1, "ta_g_L_as_citric": 4.0, "co2_volumes": 3.0},
            "ingredients": [
                {"name": "water", "node_id": name_to_id.get("water", ""), "quantity": 920.0, "unit": "mL", "class": "water"},
                {"name": "sugar", "node_id": name_to_id.get("sugar", ""), "quantity": 100.0, "unit": "g", "class": "sweetener"},
                {"name": "citric_acid", "node_id": name_to_id.get("citric_acid", ""), "quantity": 2.0, "unit": "g", "class": "acid"},
                {"name": "lemon", "node_id": name_to_id.get("lemon", ""), "quantity": 5.0, "unit": "mL", "class": "flavor"},
                {"name": "black_salt", "node_id": name_to_id.get("black_salt", ""), "quantity": 0.3, "unit": "g", "class": "flavor"}
            ],
            "process": {"template": "standard_csd", "steps": [{"step": s, "params": {}} for s in [
                "prepare_syrup_mix", "filter", "in_line_blend_to_target_brix_and_acid", "chill_to_cold_fill_temperature", "carbonation_to_target_volumes", "fill_and_seal", "date_code_and_pack"
            ]]},
            "labels": {"veg_symbol": True, "claims": [], "warnings": []}
        },
        {
            "id": "IN-CSD-0002",
            "name": "Masala Soda",
            "country": "IN",
            "subcategory": "carbonated",
            "dietary_mode": "liquid",
            "veg_flag": True,
            "jain_flag": False,
            "targets": {"brix_percent": 9.5, "pH": 3.2, "ta_g_L_as_citric": 3.5, "co2_volumes": 3.0},
            "ingredients": [
                {"name": "water", "node_id": name_to_id.get("water", ""), "quantity": 930.0, "unit": "mL", "class": "water"},
                {"name": "sugar", "node_id": name_to_id.get("sugar", ""), "quantity": 85.0, "unit": "g", "class": "sweetener"},
                {"name": "citric_acid", "node_id": name_to_id.get("citric_acid", ""), "quantity": 1.8, "unit": "g", "class": "acid"},
                {"name": "lime", "node_id": name_to_id.get("lime", ""), "quantity": 6.0, "unit": "mL", "class": "flavor"},
                {"name": "cumin", "node_id": name_to_id.get("cumin", ""), "quantity": 0.2, "unit": "g", "class": "flavor"},
                {"name": "black_salt", "node_id": name_to_id.get("black_salt", ""), "quantity": 0.3, "unit": "g", "class": "flavor"}
            ],
            "process": {"template": "standard_csd", "steps": [{"step": s, "params": {}} for s in [
                "prepare_syrup_mix", "filter", "in_line_blend_to_target_brix_and_acid", "chill_to_cold_fill_temperature", "carbonation_to_target_volumes", "fill_and_seal", "date_code_and_pack"
            ]]},
            "labels": {"veg_symbol": True, "claims": [], "warnings": []}
        },
        {
            "id": "IN-CSD-0003",
            "name": "Diet Lemon Soda (Non-nutritive sweetener)",
            "country": "IN",
            "subcategory": "carbonated",
            "dietary_mode": "liquid",
            "veg_flag": True,
            "jain_flag": True,
            "targets": {"brix_percent": 1.0, "pH": 3.1, "ta_g_L_as_citric": 4.0, "co2_volumes": 3.0},
            "ingredients": [
                {"name": "water", "node_id": name_to_id.get("water", ""), "quantity": 980.0, "unit": "mL", "class": "water"},
                {"name": "citric_acid", "node_id": name_to_id.get("citric_acid", ""), "quantity": 2.0, "unit": "g", "class": "acid"},
                {"name": "lemon", "node_id": name_to_id.get("lemon", ""), "quantity": 5.0, "unit": "mL", "class": "flavor"},
                {"name": "stevia", "node_id": name_to_id.get("stevia", ""), "quantity": 0.08, "unit": "g", "class": "sweetener", "additive_code": "FSSAI:steviol_glycosides"}
            ],
            "process": {"template": "standard_csd", "steps": [{"step": s, "params": {}} for s in [
                "prepare_syrup_mix", "filter", "in_line_blend_to_target_brix_and_acid", "chill_to_cold_fill_temperature", "carbonation_to_target_volumes", "fill_and_seal", "date_code_and_pack"
            ]]},
            "labels": {"veg_symbol": True, "claims": [], "warnings": ["Contains non-nutritive sweetener"]}
        }
    ]

    ensure_dirs(OUTPUT_JSONL)
    with open(OUTPUT_JSONL, "w") as out:
        for r in recipes:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote seed dataset → {OUTPUT_JSONL} ({len(recipes)} records)")


if __name__ == "__main__":
    main()


