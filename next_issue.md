## Accuracy and Production Readiness Guide (Indianspecific, Carbonated Beverages)

### 1) Current status (what you have now)
- Embeddings: FlavorGraph2Vec trained on a food-chemical graph, good for “what goes with what.”
- Beverage artifacts (India, carbonated):
  - Constraints JSON: `config/constraints/fssai_carbonated_beverage_constraints.json` (draft; needs legal verification and exact permitted-use tables).
  - Training schema: `config/schema/beverage_training_schema.json`.
  - Seed dataset: `data/beverage_seed_carbonated_IN.jsonl` (3 synthetic records; not representative).
  - Validator: `tools/validate_beverage.py` (structure in place).
  - Compatibility model: `models/compat_beverage_IN.pkl` (trained on synthetic data; not indicative of real-world accuracy).
  - Generator: `tools/generate_recipe.py` (greedy, constraints-aware; heuristic amounts/process; node name resolution present but needs canonical aliases).

This is a functional scaffold for the end-to-end loop, but accuracy/trust depend on real data, verified constraints, and stronger evaluation.

---

### 2) What “accurate and trustable” should mean
- Pair/Set accuracy: correct compatibility scoring for Indian carbonated beverage ingredients.
  - Metrics: ROC-AUC, PR-AUC, NDCG@K on held-out pairs/triples.
- Constraint reliability: 0% violations on FSSAI additive limits, caffeine rules, BRIX/pH/CO₂ targets, allergen/veg/Jain constraints.
- Process feasibility: process steps that are manufacturable and safe; validated with domain rules (e.g., cold-fill/carbonation flow).
- Calibration and explainability: confidence scores + reasons (nearest neighbors, metapath trails).
- Human preference alignment: chef/QA feedback increases acceptance rate over time.

---

### 3) Data program (the single biggest driver of accuracy)
- Curate real beverage data (India focus):
  - Carbonated: nimbu/masala soda, citrus variants, ginger-lime, cola-like, orange/lemon-lime profiles.
  - Sources: open recipe corpora, public R&D formulations, supplier app notes, academic publications.
- Normalize to schema:
  - Map each ingredient to a FlavorGraph `node_id` and class (water/sweetener/acid/flavor/preservative).
  - Add targets: BRIX, pH (and TA as citric), CO₂ volumes; optional Na/K.
  - Add process template (`standard_csd`/`preserved_csd`) with parameters.
- Create labels for model training:
  - Positives: co-usage pairs/triples from actual formulas.
  - Negatives: sample random pairs matched by frequency; optionally “hard negatives” (semantically similar but rarely co-used).
- Scale to at least hundreds of beverage formulas initially; aim for thousands for robust generalization.

Checklist:
- [ ] Build a canonical alias map for key ingredients (e.g., water → carbonated_water; sugar → granulated_sugar/cane_sugar; lemon → lemon_juice).
- [ ] Resolve all names to `node_id`s and keep a versioned alias file (`config/aliases/beverage_aliases.json`).

---

### 4) Regulatory/constraints hardening (FSSAI)
- Convert draft constraints to verified tables:
  - Exact permitted sweeteners, colors, preservatives for carbonated beverages with max-use levels and units.
  - Caffeine: fill exact max mg/L and warning thresholds if applicable to your sub-category.
- Encode them as machine-checkable lists:
  - `constraints.additives.*.permitted_items` with `{code, name, max_use, unit, subcategory_scope}`.
- Expand the validator:
  - Strict additive lookup by code and cumulative exposure.
  - Caffeine quantity check from ingredient contributions.
  - Labeling completeness (e.g., sweetener disclaimers).
  - Fail closed: unknown additives or missing limits should block generation.

Checklist:
- [ ] Replace “refer_schedule” with specific lists and numeric maxima.
- [ ] Add caffeine max and labeling thresholds for your subcategory.

---

### 5) Modeling improvements
- Beverage-specific fine-tune:
  - Use existing embeddings; train a compatibility head on real beverage co-usage (pairs, then triples).
  - Features: cosine similarity, abs-diff, elementwise product; or a small MLP over concatenated embeddings.
  - Regularize, calibrate (temperature scaling); report AUC/NDCG with CIs.
- Set optimization:
  - Beam/MIP search optimizing avg pair/triple scores + novelty term; hard constraints via validator.
- Amounts and process:
  - Replace heuristics with a constrained predictor:
    - Option A: Fit small regressors for BRIX and acid contributions (sugars/acid sources) → solve for grams to hit targets.
    - Option B: LLM to propose amounts/process, then deterministically adjust to match BRIX/pH using solver, then re-validate.
- Chemical augmentation (optional but useful):
  - Enable CSP layer by supplying RDKit fingerprints for beverage-relevant compounds to add chemical priors into embeddings.

---

### 6) Evaluation plan (make trust measurable)
- Offline:
  - Pair/Triple prediction ROC-AUC, PR-AUC, NDCG@K on held-out.
  - Constraint violation rate (target 0%).
  - Stability: variance across seeds/metapaths/bootstraps.
  - Calibration: reliability diagrams, Brier score.
- Human sensory/chef evaluation:
  - Win-rate in head-to-head comparisons vs. baselines.
  - Structured feedback captured for preference learning.
- Online:
  - Track validator pass rate, iterations to pass, edits required.

---

### 7) Human-in-the-loop and safety
- Guardrails:
  - All generation must pass validator.
  - Out-of-distribution detector (embedding distance or density) blocks risky suggestions.
- Feedback:
  - Chef/QA ranks suggested sets; store preferences to train a preference model (pairwise ranking loss).
  - Incrementally reweight the compatibility head using feedback.

---

### 8) Deployment and operations
- Versioning:
  - Datasets (JSONL), constraints JSON, models, and alias maps.
- APIs:
  - suggest_flavors, score_pairs, generate_recipe, validate_recipe.
- Monitoring:
  - Drift detection (ingredient distributions), constraint update reminders (FSSAI changes).
- Documentation:
  - Spec sheets: batch sizes, process parameters, labels.

---

### 9) 30/60/90-day plan
- 30 days:
  - Collect 200–400 Indian carbonated formulas; normalize to schema.
  - Finalize FSSAI additive/caffeine tables; harden validator.
  - Train compatibility v1; replace heuristic amounts with target-driven solver (BRIX/pH).
- 60 days:
  - Scale to 800–1,200 formulas; add triples; calibrate and explain.
  - Add preference learning from chef/QA panel; launch MVP UI.
- 90 days:
  - Robust spec generation; batch scaling; cost/availability integration.
  - Pilot with 2–3 product lines; measure acceptance.

---

### 10) Practical commands (local)

Train embeddings (already done in this repo):
```bash
source flavorgraph_env/bin/activate
python3 src/main.py --iterations 2 --num_walks 10 --len_metapath 20 --num_workers 0
```

Generate seed data (replace with real data ASAP):
```bash
python3 tools/generate_seed_dataset.py
```

Validate:
```bash
python3 tools/validate_beverage.py data/beverage_seed_carbonated_IN.jsonl | cat
```

Train compatibility model:
```bash
python3 tools/train_compat_model.py
```

Generate recipe:
```bash
python3 tools/generate_recipe.py | tee output/generated_recipe_IN.json
```

---

### 11) Gaps to close (before real-world use)
- Replace synthetic seed with real, normalized beverage dataset.
- Fill exact FSSAI additive/caffeine tables; enforce with validator.
- Replace heuristic amounts/process with solver/learned models.
- Add alias map to ensure robust node mapping (e.g., carbonated_water, granulated_sugar, lemon_juice).
- Establish human evaluation loop; calibrate and monitor.

Once these are in place, the system can generate India-specific, regulatorily compliant, production-ready carbonated beverage recipes with measurable accuracy and trust.