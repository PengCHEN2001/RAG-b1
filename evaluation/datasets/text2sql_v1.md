# Text-to-SQL Benchmark v1

该测试集用于评估项目的 SQL Agent。它基于当前 `wahadb.sqlite` 数据库快照，共包含 12 个经过人工确认的问题，不使用 RAGAS。

`text2sql_v1.jsonl` 是供评估程序读取的正式数据；本文件用于人工查看。当前只冻结测试问题、参考 SQL 和预期结果，不运行尚待修改的 SQL Retriever。

| ID | 测试问题 | 类型 | 预期状态 | 正确结果 |
|---|---|---|---|---|
| SQL-01 | What is the Toughness of an Imperial Space Marine? | 单位属性 | `ok` | Imperial Space Marine，Toughness = 4 |
| SQL-02 | What weapons does Custodian Guard have? | 武器查询 | `ok` | 5 个武器 profile，见下方 |
| SQL-03 | What keywords does an Imperial Space Marine have? | Keyword 查询 | `ok` | 6 个 keywords，见下方 |
| SQL-04 | What abilities does a Blade Champion have? | Ability 查询 | `ok` | 5 个 abilities，见下方 |
| SQL-05 | How many Adeptus Custodes units are in the database? | 阵营＋聚合 | `ok` | 31 |
| SQL-06 | Which Adeptus Custodes models have a Toughness of at least 10? | 多表＋数值筛选 | `ok` | 6 个 models，见下方 |
| SQL-07 | Which faction does the Imperial Space Marine belong to? | 单位＋阵营 | `ok` | Space Marines |
| SQL-08 | What are the model statistics of Custodian Guard? | 单位属性 | `ok` | Movement、Toughness、Save、Wounds、Leadership 和 Objective Control，见下方 |
| SQL-09 | What are the CP cost and phase of ARCANE GENETIC ALCHEMY for Custodian Guard? | Stratagem 关联 | `ok` | 1 CP，Any phase |
| SQL-10 | Which Adeptus Custodes models have a Toughness of at least 20? | 无结果查询 | `empty` | 无结果 |
| SQL-11 | What are the full weapon profiles of the Terrax-pattern Termite? | 名称歧义 | `needs_clarification` | 应要求选择阵营，见下方 |
| SQL-12 | How many Command points do I currently have in this battle? | 数据库范围外 | `unsupported` | 数据库不保存当前对局中的 Command points |

## SQL-02 正确结果

| Weapon | Type | Range | A | BS/WS | S | AP | D |
|---|---|---:|---:|---:|---:|---:|---:|
| Guardian spear | Melee | Melee | 5 | 2 | 7 | -2 | 2 |
| Guardian spear | Ranged | 24 | 2 | 2 | 4 | -1 | 2 |
| Misericordia | Melee | Melee | 5 | 2 | 5 | -2 | 1 |
| Sentinel blade | Melee | Melee | 5 | 2 | 6 | -2 | 1 |
| Sentinel blade | Ranged | 12 | 2 | 2 | 4 | -1 | 2 |

## SQL-03 正确结果

- Adeptus Astartes
- Character
- Grenades
- Imperial Space Marine
- Imperium
- Infantry

## SQL-04 正确结果

- Deep Strike
- Leader
- Martial Inspiration
- Martial Ka’tah
- Swift Onslaught

## SQL-06 正确结果

- Ares Gunship — 12
- Coronus Grav-carrier — 12
- Orion Assault Dropship — 12
- Venerable Land Raider — 12
- Caladius Grav-tank — 11
- Telemon Heavy Dreadnought — 10

## SQL-08 正确结果

| Statistic | Value |
|---|---:|
| Movement | 6\" |
| Toughness | 6 |
| Save | 2+ |
| Wounds | 3 |
| Leadership | 6+ |
| Objective Control | 2 |

## SQL-11 可接受的候选阵营

- Adeptus Mechanicus
- Chaos Space Marines
- Death Guard
- Space Marines
- Thousand Sons
- World Eaters

## 状态含义

- `ok`：数据库能够正常返回预期结果；
- `empty`：问题可以执行，但数据库中没有符合条件的结果；
- `needs_clarification`：数据库中存在多个候选实体，需要用户补充选择；
- `unsupported`：问题需要数据库中不存在的信息，例如当前对局状态。
