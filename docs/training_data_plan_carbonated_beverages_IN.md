## Training Data Plan — India, Carbonated Beverages (Liquid dietary mode)

### Objectives
- Train a beverage-specific compatibility model and recipe generator for Indian carbonated soft drinks (CSD), compliant with FSSAI.

### Data sources
1. Beverage recipes/formulas
   - Public recipe corpora (focus: Indian flavors, masala soda, nimbu soda, regional citrus, spice profiles).
   - Open forums/manufacturer manuals (process templates; carbonations; BRIX/pH exemplars).
   - Internal R&D notes (if available) — normalized to schema.
2. Ingredient metadata
   - FlavorGraph nodes mapped to beverage ingredient classes: water, acids, sweeteners, flavors, colors, preservatives, stabilizers.
   - Cost, availability, allergen flags, veg/Jain flags.
3. Regulatory
   - FSSAI additive schedules and permitted use levels for beverages (sweeteners, preservatives, colours).
   - Caffeine rules for caffeinated drinks; labeling elements.

### Data normalization
- Use `config/schema/beverage_training_schema.json`.
- Map each ingredient to a FlavorGraph node_id (string) and class (acid/sweetener/etc.).
- Capture targets: BRIX, pH, TA as citric, CO2 volumes.
- Encode process steps via `standard_csd` or `preserved_csd` templates; parameters as JSON.

### Labeling for training
- Pairwise/triple compatibility labels derived from co-usage in formulas/recipes.
- Negative sampling with similar frequency distributions.
- Optional human preference labels (chef/QA) to power preference learning later.

### Model training
1. Base embeddings
   - Use existing FlavorGraph2Vec (or retrain with beverage-focused metapaths; Node2Vec as baseline).
2. Compatibility head
   - Siamese/bi-encoder scoring f(ingredient_i, ingredient_j[, k]).
   - Loss: margin ranking + calibration (Brier/temperature scaling).
3. Set optimizer
   - Beam/MIP search to assemble sets under constraints from `config/constraints/fssai_carbonated_beverage_constraints.json`.
4. Amounts and process
   - LLM with structured prompts from schema + heuristics (BRIX/pH contributions); post-validate; iterate.

### Validators
- Exact-match tables for permitted additives and max levels (FSSAI) — versioned.
- Targets: BRIX, pH, CO2; sodium/potassium caps; allergens; veg/Jain modes.
- Labeling completeness and warnings.

### Splits and evaluation
- Train/val/test by product family (avoid leakage of near-duplicates).
- Metrics: ROC-AUC/NDCG for pair prediction; constraint violation rate (0% target); calibration; human panel win-rate.

### Artifacts
- Cleaned JSONL dataset conforming to schema.
- Constraints JSON (already added).
- Trained compatibility model, optimizer configs, validator rules, and evaluation reports.

### Timeline (3 weeks)
- Week 1: Data collection/normalization; constraints table verification; baseline evaluations.
- Week 2: Train compatibility head; implement validators; first generator loop.
- Week 3: Human review; preference learning; finalize MVP.


