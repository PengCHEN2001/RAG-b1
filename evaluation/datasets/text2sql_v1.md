# Text-to-SQL Benchmark v1

This benchmark evaluates the project's SQL agent against the current `wahadb.sqlite` database snapshot. It contains 12 manually reviewed questions and does not use RAGAS.

`text2sql_v1.jsonl` is the machine-readable benchmark. This document provides a human-readable review of the questions and expected results. The benchmark is frozen independently of the SQL retriever's current implementation status.

| ID | Test question | Category | Expected status | Expected result |
|---|---|---|---|---|
| SQL-01 | What is the Toughness of an Imperial Space Marine? | Unit attribute | `ok` | Imperial Space Marine, Toughness = 4 |
| SQL-02 | What weapons does Custodian Guard have? | Weapon query | `ok` | 5 weapon profiles; see below |
| SQL-03 | What keywords does an Imperial Space Marine have? | Keyword query | `ok` | 6 keywords; see below |
| SQL-04 | What abilities does a Blade Champion have? | Ability query | `ok` | 5 abilities; see below |
| SQL-05 | How many Adeptus Custodes units are in the database? | Faction aggregation | `ok` | 31 |
| SQL-06 | Which Adeptus Custodes models have a Toughness of at least 10? | Relational numeric filter | `ok` | 6 models; see below |
| SQL-07 | Which faction does the Imperial Space Marine belong to? | Unit–faction relation | `ok` | Space Marines |
| SQL-08 | What are the model statistics of Custodian Guard? | Unit attributes | `ok` | Movement, Toughness, Save, Wounds, Leadership, and Objective Control; see below |
| SQL-09 | What are the CP cost and phase of ARCANE GENETIC ALCHEMY for Custodian Guard? | Stratagem relation | `ok` | 1 CP, Any phase |
| SQL-10 | Which Adeptus Custodes models have a Toughness of at least 20? | Empty-result query | `empty` | No matching rows |
| SQL-11 | What are the full weapon profiles of the Terrax-pattern Termite? | Name ambiguity | `needs_clarification` | The agent should ask the user to select a faction; see below |
| SQL-12 | How many Command points do I currently have in this battle? | Out-of-scope query | `unsupported` | The database does not store the current battle's Command point state |

## SQL-02 Expected Result

| Weapon | Type | Range | A | BS/WS | S | AP | D |
|---|---|---:|---:|---:|---:|---:|---:|
| Guardian spear | Melee | Melee | 5 | 2 | 7 | -2 | 2 |
| Guardian spear | Ranged | 24 | 2 | 2 | 4 | -1 | 2 |
| Misericordia | Melee | Melee | 5 | 2 | 5 | -2 | 1 |
| Sentinel blade | Melee | Melee | 5 | 2 | 6 | -2 | 1 |
| Sentinel blade | Ranged | 12 | 2 | 2 | 4 | -1 | 2 |

## SQL-03 Expected Result

- Adeptus Astartes
- Character
- Grenades
- Imperial Space Marine
- Imperium
- Infantry

## SQL-04 Expected Result

- Deep Strike
- Leader
- Martial Inspiration
- Martial Ka’tah
- Swift Onslaught

## SQL-06 Expected Result

- Ares Gunship — 12
- Coronus Grav-carrier — 12
- Orion Assault Dropship — 12
- Venerable Land Raider — 12
- Caladius Grav-tank — 11
- Telemon Heavy Dreadnought — 10

## SQL-08 Expected Result

| Statistic | Value |
|---|---:|
| Movement | 6" |
| Toughness | 6 |
| Save | 2+ |
| Wounds | 3 |
| Leadership | 6+ |
| Objective Control | 2 |

## SQL-11 Accepted Faction Candidates

- Adeptus Mechanicus
- Chaos Space Marines
- Death Guard
- Space Marines
- Thousand Sons
- World Eaters

## Status Definitions

- `ok`: the database can return the expected result.
- `empty`: the question can be queried, but no database row satisfies the conditions.
- `needs_clarification`: multiple database entities match and the user must select one.
- `unsupported`: the requested information is not stored in the database, such as live battle state.
