# Combined Benchmark v1

This benchmark evaluates whether the router and orchestration pipeline can use both the SQL retriever and the Rulebook retriever for a single question, then produce one evidence-grounded answer from both sources.

The current version contains two manually reviewed questions. Both reuse evidence already validated in the Text-to-SQL and Rulebook benchmarks; no additional RAGAS generation is used.

## Combined-01 — Grenades

**Question:** Does an Imperial Space Marine have the Grenades keyword, and if so, how is the Grenade Stratagem used?

**Expected route:** `both`

**Required sources:**

- SQL: `sql_003`
- Rulebook: `rb_001`
- Required Rulebook chunk: `core-rules-grenade-001-001`

**Reference answer:**

Yes. An Imperial Space Marine has the Grenades keyword. The Grenade Stratagem costs 1CP and is used in the Shooting phase. Select one Grenades unit from your army that is not within Engagement Range of any enemy units and has not been selected to shoot that phase. Then select a visible enemy unit within 8 inches that is not within Engagement Range of any units from your army and roll six D6; each 4+ causes 1 mortal wound.

**Review:** approved

## Combined-02 — Objective Control and scoring

**Question:** What is the Objective Control characteristic of Custodian Guard, and how are Victory points scored in Capture and Control?

**Expected route:** `both`

**Required sources:**

- SQL: `sql_008`
- Rulebook: `rb_003`
- Required Rulebook chunk: `core-rules-capture-and-control-001-001`

**Reference answer:**

Custodian Guard has an Objective Control characteristic of 2. In Capture and Control, starting from the second battle round, at the end of each player's Command phase, the player whose turn it is scores 1 Victory point for each objective marker they control, up to 3 VP per turn.

**Review:** approved
