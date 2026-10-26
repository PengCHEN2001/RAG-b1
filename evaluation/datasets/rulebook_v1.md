# Rulebook Benchmark v1

> All samples have been checked against the canonical project chunks.

- Generator model: `gemini-3.5-flash-lite`
- Embedding model: `models/gemini-embedding-001`
- Target distribution: 4 single-hop specific, 2 multi-hop abstract
- Review status: `approved`

| ID | Origin | Type | Question |
|---|---|---|---|
| rb_001 | teacher_required | single_hop_specific | What is the rule for Grenades? |
| rb_002 | teacher_required | single_hop_specific | How can I use a Stratagem? |
| rb_003 | teacher_required | multi_hop_abstract | What are the different ways to score points? |
| rb_004 | ragas_generated | single_hop_specific | When does an Aircraft placed into Strategic Reserves return? |
| rb_005 | ragas_generated | single_hop_specific | Can my unit still make a Normal or Advance move during the Movement phase if the only enemy models within its Engagement Range are Aircraft? |
| rb_006 | ragas_generated | multi_hop_abstract | An Aircraft cannot complete its minimum move and is placed into Strategic Reserves. When will it return, and after it returns, which units can charge it or make melee attacks against it? |

## rb_001 — single_hop_specific

**Question:** What is the rule for Grenades?

**Reference answer:** The Grenade Stratagem costs 1CP. In your Shooting phase, select one unit with the Grenades keyword that is not within Engagement Range of any enemy units and has not been selected to shoot. Select a visible enemy unit within 8 inches that is not within Engagement Range of any units from your army, then roll six D6; each 4+ causes 1 mortal wound.

**Reference chunks:**

- `core-rules-grenade-001-001` — GRENADE, page(s) 42, mapping 100.0%

**Review:** approved — Added the 1CP cost after restoring it to the canonical chunk from PDF page 42.


## rb_002 — single_hop_specific

**Question:** How can I use a Stratagem?

**Reference answer:** Spend the CP listed on the Stratagem. You cannot use it without enough CP. A Stratagem may be reused during a battle, but the same Stratagem cannot be used more than once in the same phase.

**Reference chunks:**

- `core-rules-stratagems-001-001` — STRATAGEMS, page(s) 41, mapping 100.0%

**Review:** approved — Removed the unsupported qualifier 'normally'.


## rb_003 — multi_hop_abstract

**Question:** What are the different ways to score points?

**Reference answer:** Players score Victory points by achieving the objectives defined by the mission. Depending on the mission, these can include recovering relics, capturing strongholds, or eliminating the opposing Warlord. For example, Capture and Control awards 1 VP for each objective marker controlled at the end of the player's Command phase, starting from the second battle round, up to 3 VP per turn.

**Reference chunks:**

- `core-rules-introduction-001-001` — Introduction, page(s) 3, mapping 100.0%
- `core-rules-read-mission-objectives-001-001` — READ MISSION OBJECTIVES, page(s) 57, mapping 100.0%
- `core-rules-capture-and-control-001-001` — CAPTURE AND CONTROL, page(s) 59, mapping 100.0%

**Review:** approved — Distinguishes general mission-dependent scoring from the Capture and Control example.


## rb_004 — single_hop_specific

**Question:** When does an Aircraft placed into Strategic Reserves return?

**Reference answer:** If an Aircraft model is placed into Strategic Reserves, it will always arrive from Strategic Reserves in your next turn.

**Reference chunks:**

- `core-rules-aircraft-in-the-movement-phase-001-001` — AIRCRAFT IN THE MOVEMENT PHASE, page(s) 53, mapping 100.0%

**Review:** approved — RAGAS wording normalized into a complete question.


## rb_005 — single_hop_specific

**Question:** Can my unit still make a Normal or Advance move during the Movement phase if the only enemy models within its Engagement Range are Aircraft?

**Reference answer:** Yes. If the only enemy models within Engagement Range of the selected unit are Aircraft models, that unit can still make a Normal or Advance move.

**Reference chunks:**

- `core-rules-aircraft-and-the-movement-of-other-models-001-001` — AIRCRAFT AND THE MOVEMENT OF OTHER MODELS, page(s) 54, mapping 100.0%

**Review:** approved — Accepted after comparison with the canonical chunk.


## rb_006 — multi_hop_abstract

**Question:** An Aircraft cannot complete its minimum move and is placed into Strategic Reserves. When will it return, and after it returns, which units can charge it or make melee attacks against it?

**Reference answer:** It returns from Strategic Reserves in your next turn. Only units that can Fly can charge it or make melee attacks against it. The Aircraft itself cannot declare a charge; it can fight only while within Engagement Range of an enemy unit that can Fly, and it can make melee attacks only against units that can Fly.

**Reference chunks:**

- `core-rules-aircraft-in-the-movement-phase-001-001` — AIRCRAFT IN THE MOVEMENT PHASE, page(s) 53, mapping 100.0%
- `core-rules-aircraft-in-the-charge-and-fight-phases-001-001` — AIRCRAFT IN THE CHARGE AND FIGHT PHASES, page(s) 54, mapping 100.0%

**Review:** approved — Uses two supporting chunks; corrected the second chunk's page metadata to page 54.
