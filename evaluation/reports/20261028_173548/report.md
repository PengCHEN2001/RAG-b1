# Evaluation Report

- Run ID: `20261028_173548`
- Dataset: `all`
- Agent model: `gemini-3.5-flash-lite`
- RAGAS judge: `gemini-3.5-flash-lite`
- Git commit: `e7eb24884acc1f7bd9e78809bec80aabdaa46bf8`
- Uncommitted changes: `True`

## Results

- Total: 20
- Passed: 16
- Failed: 4

### Deterministic metrics

```json
{
  "rulebook": {
    "case_count": 6,
    "hit_at_k": 1.0,
    "precision_at_k": 0.23333333333333336,
    "recall_at_k": 0.8888888888888888,
    "mrr": 1.0
  },
  "text2sql": {
    "case_count": 12,
    "execution_case_count": 9,
    "execution_accuracy": 0.7777777777777778,
    "status_accuracy": 0.9166666666666666,
    "result_accuracy": 0.8333333333333334,
    "error_rate": 0.0,
    "mean_latency_ms": 55615.22930918727
  },
  "router": {
    "case_count": 20,
    "accuracy": 1.0,
    "macro_f1": 1.0,
    "labels": [
      "both",
      "rulebook",
      "sql",
      "unsupported"
    ],
    "per_route": {
      "both": {
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "support": 2
      },
      "rulebook": {
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "support": 6
      },
      "sql": {
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "support": 11
      },
      "unsupported": {
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "support": 1
      }
    },
    "confusion_matrix": {
      "both": {
        "both": 2,
        "rulebook": 0,
        "sql": 0,
        "unsupported": 0
      },
      "rulebook": {
        "both": 0,
        "rulebook": 6,
        "sql": 0,
        "unsupported": 0
      },
      "sql": {
        "both": 0,
        "rulebook": 0,
        "sql": 11,
        "unsupported": 0
      },
      "unsupported": {
        "both": 0,
        "rulebook": 0,
        "sql": 0,
        "unsupported": 1
      }
    }
  }
}
```

### RAGAS metrics

```json
{
  "context_precision": 0.49999999995,
  "context_recall": 1.0,
  "faithfulness": 1.0,
  "response_relevancy": 0.762739670998154,
  "factual_correctness": null
}
```

## Failures

### `rb_003` — benchmark_mismatch

- Question: What are the different ways to score points?
- Expected route: `rulebook`
- Predicted route: `rulebook`
- Semantic query: What are the different ways to score victory points in the game?
- Keyword query: victory points, scoring, battle round, mission, primary mission, secondary mission

**Actual answer**

> Based on the provided rules, players can score Victory points (VP) progressively by controlling objective markers: starting from the second battle round, at the end of each player's Command phase, the player whose turn it is scores 1 VP for each objective marker they control, up to a maximum of 3 VP per turn [R1].

**Reference answer**

> Players score Victory points by achieving the objectives defined by the mission. Depending on the mission, these can include recovering relics, capturing strongholds, or eliminating the opposing Warlord. For example, Capture and Control awards 1 VP for each objective marker controlled at the end of the player's Command phase, starting from the second battle round, up to 3 VP per turn.

**Retrieved Rulebook chunks**

1. `core-rules-capture-and-control-001-001` — CAPTURE AND CONTROL
2. `core-rules-create-the-battlefield-002-002` — CREATE THE BATTLEFIELD
3. `core-rules-determine-victor-001-001` — DETERMINE VICTOR
4. `core-rules-select-army-faction-001-001` — SELECT ARMY FACTION
5. `core-rules-the-battle-round-001-001` — THE BATTLE ROUND

**Missing reference chunks**

- `core-rules-introduction-001-001`
- `core-rules-read-mission-objectives-001-001`

### `sql_001` — benchmark_mismatch

- Question: What is the Toughness of an Imperial Space Marine?
- Expected route: `sql`
- Predicted route: `sql`

**Actual answer**

> The Imperial Space Marine has a Toughness of 4 [S1].

**Actual SQL result**

```json
{
  "status": "ok",
  "strategy": "generated",
  "columns": [
    "datasheet_name",
    "model_name",
    "toughness"
  ],
  "rows": [
    [
      "Imperial Space Marine",
      "Imperial Space Marine",
      "4"
    ]
  ],
  "error_code": null,
  "candidates": []
}
```

### `sql_003` — benchmark_mismatch

- Question: What keywords does an Imperial Space Marine have?
- Expected route: `sql`
- Predicted route: `sql`

**Actual answer**

> Multiple matches; specify a candidate identity or more context. 1. Imperial Agents 2. Imperial Knights
>
> 1. {"key": "faction", "kind": "faction", "id": "AoI", "name": "Imperial Agents", "matched_text": "Imperial", "match_type": "phrase"}
> 2. {"key": "faction", "kind": "faction", "id": "QI", "name": "Imperial Knights", "matched_text": "Imperial", "match_type": "phrase"}

**Actual SQL result**

```json
{
  "status": "needs_clarification",
  "strategy": "generated",
  "columns": [],
  "rows": [],
  "error_code": null,
  "candidates": [
    {
      "key": "faction",
      "kind": "faction",
      "id": "AoI",
      "name": "Imperial Agents",
      "matched_text": "Imperial",
      "match_type": "phrase"
    },
    {
      "key": "faction",
      "kind": "faction",
      "id": "QI",
      "name": "Imperial Knights",
      "matched_text": "Imperial",
      "match_type": "phrase"
    }
  ]
}
```

### `combined_001` — benchmark_mismatch

- Question: Does an Imperial Space Marine have the Grenades keyword, and if so, how is the Grenade Stratagem used?
- Expected route: `both`
- Predicted route: `both`
- Semantic query: Does an Imperial Space Marine have the Grenades keyword and how is the Grenade Stratagem used
- Keyword query: Imperial Space Marine, Grenades, Grenade Stratagem, keyword

**Actual answer**

> Yes, an Imperial Space Marine has the Grenades keyword [S1]. The Grenade Stratagem costs 1 CP and is used in your Shooting phase [R1]. It targets one Grenades unit from your army that is not within Engagement Range of any enemy units and has not been selected to shoot this phase [R1]. To use it, select one enemy unit that is not within Engagement Range of any units from your army and is within 8" of and visible to your Grenades unit, then roll six D6: for each 4+, that enemy unit suffers 1 mortal wound [R1].

**Reference answer**

> Yes. An Imperial Space Marine has the Grenades keyword. The Grenade Stratagem costs 1CP and is used in the Shooting phase. Select one Grenades unit from your army that is not within Engagement Range of any enemy units and has not been selected to shoot that phase. Then select a visible enemy unit within 8 inches that is not within Engagement Range of any units from your army and roll six D6; each 4+ causes 1 mortal wound.

**Retrieved Rulebook chunks**

1. `core-rules-grenade-001-001` — GRENADE
2. `core-rules-stratagems-001-001` — STRATAGEMS
3. `core-rules-fire-overwatch-001-001` — FIRE OVERWATCH
4. `core-rules-hints-and-tips-fast-dice-rolling-001-003` — Hints and Tips FAST DICE ROLLING
5. `core-rules-stratagem-categories-001-001` — STRATAGEM CATEGORIES

**Actual SQL result**

```json
{
  "status": "ok",
  "strategy": "generated",
  "columns": [
    "DatasheetId",
    "datasheet_name",
    "keyword"
  ],
  "rows": [
    [
      "000000148",
      "Imperial Space Marine",
      "Grenades"
    ]
  ],
  "error_code": null,
  "candidates": []
}
```

