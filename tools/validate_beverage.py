import json
import sys
from pathlib import Path

CONSTRAINTS = "config/constraints/fssai_carbonated_beverage_constraints.json"


def load_constraints(path: str):
    with open(path) as f:
        return json.load(f)


def validate_record(rec: dict, rules: dict) -> list:
    violations = []

    hard = rules.get("hard_constraints", {})
    ops = rules.get("operational_guidelines", {}).get("targets", {})
    labeling = rules.get("labelling", {})

    # dietary/liquid
    if rec.get("dietary_mode") != rules.get("dietary", {}).get("dietary_mode"):
        violations.append({"rule": "dietary_mode", "detail": "Dietary mode must be liquid"})

    # prohibited ingredients
    prohibited = set(hard.get("prohibited_ingredients", []))
    for ing in rec.get("ingredients", []):
        name = (ing.get("name") or "").lower()
        if name in prohibited:
            violations.append({"rule": "prohibited_ingredients", "detail": f"{name} is prohibited"})

    # caffeine placeholder rule
    if any("caffeine" in (i.get("name") or "").lower() for i in rec.get("ingredients", [])):
        caf = rules.get("hard_constraints", {}).get("caffeine", {})
        if caf.get("max") is None:
            violations.append({"rule": "caffeine_limits", "detail": "Caffeine limits undefined; set per FSSAI before use."})

    # targets: pH
    pH = (rec.get("targets", {}) or {}).get("pH")
    if pH is not None:
        lo, hi = ops.get("pH_range", [None, None])
        if lo is not None and (pH < lo or pH > hi):
            violations.append({"rule": "pH_range", "detail": f"pH {pH} out of range [{lo}, {hi}]"})

    # targets: BRIX
    brix = (rec.get("targets", {}) or {}).get("brix_percent")
    if brix is not None:
        # allow any of the profiles to match
        profiles = ops.get("product_profiles", [])
        ok = False
        for prof in profiles:
            lo, hi = prof.get("brix_range_percent", [None, None])
            if lo is not None and hi is not None and lo <= brix <= hi:
                ok = True
                break
        if not ok:
            violations.append({"rule": "brix_profiles", "detail": f"BRIX {brix}% not within any profile ranges"})

    # CO2
    co2 = (rec.get("targets", {}) or {}).get("co2_volumes")
    if co2 is not None:
        lo, hi = ops.get("co2_volumes", [None, None])
        if lo is not None and (co2 < lo or co2 > hi):
            violations.append({"rule": "co2", "detail": f"CO2 {co2} vols out of range [{lo}, {hi}]"})

    # sodium/potassium caps
    ni = rec.get("nutrition_per_100mL", {}) or {}
    if "sodium_mg" in ni:
        cap = ops.get("sodium_mg_per_100mL_max")
        if cap is not None and ni["sodium_mg"] > cap:
            violations.append({"rule": "sodium_cap", "detail": f"Sodium {ni['sodium_mg']}mg/100mL exceeds {cap}"})
    if "potassium_mg" in ni:
        cap = ops.get("potassium_mg_per_100mL_max")
        if cap is not None and ni["potassium_mg"] > cap:
            violations.append({"rule": "potassium_cap", "detail": f"Potassium {ni['potassium_mg']}mg/100mL exceeds {cap}"})

    # labeling minimal checks
    mand = set(labeling.get("mandatory", []))
    # ensure veg symbol for IN
    if not (rec.get("labels", {}) or {}).get("veg_symbol", False):
        violations.append({"rule": "veg_symbol", "detail": "Veg symbol required by default for vegetarian beverages in IN."})

    return violations


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/validate_beverage.py data/beverage_seed_carbonated_IN.jsonl")
        sys.exit(1)

    data_path = Path(sys.argv[1])
    rules = load_constraints(CONSTRAINTS)

    total = 0
    passed = 0
    with open(data_path) as f:
        for line in f:
            total += 1
            rec = json.loads(line)
            v = validate_record(rec, rules)
            if not v:
                passed += 1
                print(f"[PASS] {rec.get('id')} {rec.get('name')}")
            else:
                print(f"[FAIL] {rec.get('id')} {rec.get('name')} → {len(v)} violations")
                for vv in v:
                    print("  -", vv["rule"], ":", vv["detail"])

    print(f"Summary: {passed}/{total} passed")


if __name__ == "__main__":
    main()


