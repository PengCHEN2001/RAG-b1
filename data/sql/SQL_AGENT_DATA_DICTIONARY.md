# SQL Agent Data Dictionary for `wahadb.sqlite`

## Database overview

- Database file: `wahadb.sqlite`
- SQLite database
- Total tables: 21
- Data domain: Warhammer 40,000 rules and reference data
- Schema pattern: most tables are loaded from JSON source files and flattened into SQLite tables with string-valued columns (`TEXT`)
- Important note: the live SQLite schema does not declare explicit primary keys or foreign keys; relationships are inferred by matching IDs and join columns

## Table inventory and row counts

| Table | Rows | Description |
|---|---:|---|
| `abilities` | 91 | Ability catalog |
| `datasheets` | 1680 | Unit cards / datasheets |
| `datasheets_abilities` | 6985 | Datasheet-to-ability links |
| `datasheets_detachment_abilities` | 16257 | Datasheet-to-detachment-ability links |
| `datasheets_enhancements` | 11590 | Datasheet-to-enhancement links |
| `datasheets_keywords` | 16358 | Keywords attached to datasheets |
| `datasheets_leader` | 1637 | Leader/attached unit relationships |
| `datasheets_models` | 1784 | Model stat blocks |
| `datasheets_models_cost` | 6276 | Model-cost rows |
| `datasheets_options` | 2780 | Optional unit upgrades/options |
| `datasheets_stratagems` | 86934 | Datasheet-to-stratagem links |
| `datasheets_unit_composition` | 2131 | Composition text |
| `datasheets_wargear` | 9060 | Weapon and wargear entries |
| `detachment_abilities` | 349 | Detachment-level ability roster |
| `detachments` | 327 | Detachment catalog |
| `detachments_chapter_dp` | 4 | Chapter/DP mapping |
| `enhancements` | 1028 | Enhancement roster |
| `factions` | 25 | Faction catalog |
| `last_update` | 1 | Last refresh timestamp |
| `source` | 66 | Source metadata |
| `stratagems` | 1663 | Stratagem roster |

---

## Global conventions

### ID conventions

- Most tables use string IDs stored in `TEXT` columns.
- Keys are usually named `Id`, `FactionId`, `DatasheetId`, `detachmentId`, or `abilityId`.
- IDs are not guaranteed to be numeric; they may be codes like `TYR`, `000000705`, or other nominal strings.

### Relationship conventions

- Foreign-key-style relations are expressed through matching ID fields instead of declarative SQL constraints.
- Join keys are often named the same way across tables but with case differences, e.g.:
  - `datasheets.Id`
  - `datasheets_abilities.DatasheetId`
  - `datasheets_stratagems.stratagemId`
  - `stratagems.id`
- In practice, agents should join on string equality and may need to normalize case or compare exact values carefully.

### Row semantics

- This is a rules database, not an operational transactional database.
- Many tables are lookup or associative tables used to flatten a JSON source structure into SQL rows.
- Several tables are effectively one-to-many or many-to-many junctions for a parent entity.

---

## Table-by-table schema

### `abilities`
Rows: 91

Columns:
- `Id` — ability identifier
- `name` — ability name
- `legend` — lore or descriptive legend text
- `factionId` — faction owning this ability
- `description` — rules description

Typical use:
- Retrieve all abilities for a faction
- Join `abilities` to `datasheets_abilities` via `abilityId`

---

### `factions`
Rows: 25

Columns:
- `Id` — faction identifier (e.g. `TYR`)
- `name` — faction name
- `link` — faction/reference URL

Typical use:
- Base dimension for units, detachments, stratagems, abilities, and enhancements

---

### `source`
Rows: 66

Columns:
- `Id` — source identifier
- `name` — source name
- `type` — source type
- `edition` — edition label
- `version` — version/revision
- `errataDate` — errata date, if any
- `errataLink` — errata URL

Typical use:
- Trace a datasheet to the publication or rulebook it comes from

---

### `last_update`
Rows: 1

Columns:
- `LastUpdate` — last refresh date/time

Typical use:
- Metadata for freshness or cache invalidation

---

### `detachments`
Rows: 327

Columns:
- `Id` — detachment identifier
- `factionId` — owning faction
- `name` — detachment name
- `legend` — lore/description text
- `type` — detachment category/type
- `dp` — detachment point or designation
- `forceDisposition` — orientation / force-organizational information

Typical use:
- List all detachments by faction
- Join with `detachment_abilities` and `stratagems`

---

### `detachment_abilities`
Rows: 349

Columns:
- `Id` — detachment ability identifier
- `factionId` — owning faction
- `name` — detachment ability name
- `legend` — lore text
- `description` — rules text
- `detachment` — associated detachment label
- `detachmentId` — associated detachment identifier

Typical use:
- Find ability text for a specific detachment
- Join on `detachmentId`

---

### `detachments_chapter_dp`
Rows: 4

Columns:
- `DetachmentId` — detachment identifier
- `keyword` — chapter or keyword label
- `dp` — detachment value/DP

Typical use:
- Map chapter-related keyword to detachment DP metadata

---

### `stratagems`
Rows: 1663

Columns:
- `FactionId` — owning faction
- `name` — stratagem name
- `id` — stratagem identifier
- `type` — type/category
- `cpCost` — command point cost
- `legend` — lore text
- `turn` — timing window
- `phase` — battle phase
- `detachment` — detachment label
- `detachmentId` — associated detachment
- `description` — rule text

Typical use:
- Query stratagems by faction or phase
- Join with `datasheets_stratagems` on `stratagemId` / `id`

---

### `enhancements`
Rows: 1028

Columns:
- `FactionId` — owning faction
- `name` — enhancement name
- `id` — enhancement identifier
- `cost` — cost of enhancement
- `detachment` — detachment label
- `detachmentId` — associated detachment
- `upgrade` — upgrade category
- `legend` — flavor text
- `description` — rules text
- `supportLeader` — support-leader boolean-like flag

Typical use:
- Search enhancements by faction or detachment
- Join with `datasheets_enhancements`

---

### `datasheets`
Rows: 1680

Columns:
- `Id` — datasheet identifier
- `name` — unit name
- `factionId` — owning faction
- `sourceId` — source reference
- `legend` — unit lore text
- `role` — role/category
- `loadout` — weapon or arrangement summary
- `transport` — transport information
- `virtual` — virtual/placeholder flag
- `isSupport` — support unit flag
- `leaderHead` — additional leader header text
- `leaderFooter` — leader footer text
- `damagedW` — damaged-state weapon information
- `damagedDescription` — damaged-state description
- `link` — external link

Typical use:
- Base table for most unit analysis
- Join to all subsidiary datasheet tables via `DatasheetId`

---

### `datasheets_abilities`
Rows: 6985

Columns:
- `DatasheetId` — parent datasheet
- `line` — row/order in the card
- `abilityId` — linked ability identifier
- `model` — model or unit subset to which the ability applies
- `name` — ability name on the card
- `description` — ability text
- `type` — ability category
- `parameter` — any parameter or extra text

Typical use:
- Get all abilities associated with a unit
- Join with `abilities` on `abilityId`

---

### `datasheets_detachment_abilities`
Rows: 16257

Columns:
- `DatasheetId` — parent datasheet
- `detachmentAbilityId` — linked detachment ability ID

Typical use:
- Determine whether a datasheet has detachment abilities
- Join with `detachment_abilities` on `detachmentAbilityId`

---

### `datasheets_enhancements`
Rows: 11590

Columns:
- `DatasheetId` — parent datasheet
- `enhancementId` — linked enhancement ID

Typical use:
- Link units to compatible enhancements
- Join to `enhancements` on `id`

---

### `datasheets_keywords`
Rows: 16358

Columns:
- `DatasheetId` — parent datasheet
- `keyword` — keyword text
- `model` — model context
- `isFactionKeyword` — flag for faction keyword status

Typical use:
- Find all keyword combinations for a datasheet
- Filter units by keyword or faction keyword

---

### `datasheets_leader`
Rows: 1637

Columns:
- `LeaderId` — leader datasheet ID
- `attachedId` — attached datasheet ID

Typical use:
- Model leader/attached-unit relationships
- Join a leader datasheet to the units it leads

---

### `datasheets_models`
Rows: 1784

Columns:
- `DatasheetId` — parent datasheet
- `line` — model line order
- `name` — model name
- `m` — movement
- `t` — toughness
- `sv` — save
- `invSv` — invulnerable save
- `invSvDescr` — invulnerable-save description
- `w` — wounds
- `ld` — leadership
- `oc` — objective control
- `baseSize` — base size
- `baseSizeDescr` — base-size description

Typical use:
- Return model stat lines for a unit or datasheet

---

### `datasheets_models_cost`
Rows: 6276

Columns:
- `DatasheetId` — parent datasheet
- `line` — cost line order
- `description` — cost description
- `cost` — value/cost text

Typical use:
- Get unit-model or unit-upgrade costs

---

### `datasheets_options`
Rows: 2780

Columns:
- `DatasheetId` — parent datasheet
- `line` — order within options
- `button` — option label/button text
- `description` — option description

Typical use:
- Pull optional equipment or upgrade choices for a unit

---

### `datasheets_stratagems`
Rows: 86934

Columns:
- `DatasheetId` — parent datasheet
- `stratagemId` — linked stratagem identifier

Typical use:
- Determine which stratagems are available to a datasheet
- Join with `stratagems` on `stratagemId` = `stratagems.id`

---

### `datasheets_unit_composition`
Rows: 2131

Columns:
- `DatasheetId` — parent datasheet
- `line` — composition order
- `description` — composition text

Typical use:
- Show unit composition details from the datasheet

---

### `datasheets_wargear`
Rows: 9060

Columns:
- `DatasheetId` — parent datasheet
- `line` — display order
- `lineInWargear` — sub-line in wargear section
- `dice` — dice notation
- `name` — weapon or gear name
- `description` — text description
- `range` — weapon range
- `type` — weapon type
- `a` — attacks statistic
- `bsWs` — ballistic-skill/weapon-skill statistic
- `s` — strength
- `ap` — armor penetration
- `d` — damage

Typical use:
- Retrieve the weapon profile for a datasheet or model

---

## Relationship map

This is the practical join graph for SQL generation.

### Faction graph
- `factions.Id` -> `datasheets.factionId`
- `factions.Id` -> `abilities.factionId`
- `factions.Id` -> `detachments.factionId`
- `factions.Id` -> `detachment_abilities.factionId`
- `factions.Id` -> `stratagems.FactionId`
- `factions.Id` -> `enhancements.FactionId`

### Datasheet graph
- `datasheets.Id` -> `datasheets_abilities.DatasheetId`
- `datasheets.Id` -> `datasheets_keywords.DatasheetId`
- `datasheets.Id` -> `datasheets_models.DatasheetId`
- `datasheets.Id` -> `datasheets_models_cost.DatasheetId`
- `datasheets.Id` -> `datasheets_options.DatasheetId`
- `datasheets.Id` -> `datasheets_stratagems.DatasheetId`
- `datasheets.Id` -> `datasheets_unit_composition.DatasheetId`
- `datasheets.Id` -> `datasheets_wargear.DatasheetId`
- `datasheets.Id` -> `datasheets_enhancements.DatasheetId`
- `datasheets.Id` -> `datasheets_detachment_abilities.DatasheetId`
- `datasheets.Id` -> `datasheets_leader.LeaderId`
- `datasheets.Id` -> `datasheets_leader.attachedId`

### Ability / enhancement / stratagem graph
- `abilities.Id` -> `datasheets_abilities.abilityId`
- `detachment_abilities.Id` -> `datasheets_detachment_abilities.detachmentAbilityId`
- `enhancements.id` -> `datasheets_enhancements.enhancementId`
- `stratagems.id` -> `datasheets_stratagems.stratagemId`

### Detachment graph
- `detachments.Id` -> `detachment_abilities.detachmentId`
- `detachments.Id` -> `stratagems.detachmentId`
- `detachments.Id` -> `enhancements.detachmentId`
- `detachments.Id` -> `detachments_chapter_dp.DetachmentId`

### Source graph
- `source.Id` -> `datasheets.sourceId`

---

## Recommended query patterns

### 1. Get a unit and all associated metadata
```sql
SELECT d.Id, d.name, d.factionId, f.name AS faction_name,
       d.role, d.legend, d.link
FROM datasheets d
LEFT JOIN factions f ON f.Id = d.factionId
WHERE d.Id = '...';
```

### 2. Get all abilities for a datasheet
```sql
SELECT d.name AS datasheet_name,
       a.Id AS ability_id,
       a.name AS ability_name,
       a.description AS ability_text
FROM datasheets d
LEFT JOIN datasheets_abilities da ON da.DatasheetId = d.Id
LEFT JOIN abilities a ON a.Id = da.abilityId
WHERE d.Id = '...';
```

### 3. Get all keywords for a datasheet
```sql
SELECT d.name AS datasheet_name,
       dk.keyword,
       dk.model,
       dk.isFactionKeyword
FROM datasheets d
LEFT JOIN datasheets_keywords dk ON dk.DatasheetId = d.Id
WHERE d.Id = '...';
```

### 4. Get all stratagems available to a datasheet
```sql
SELECT d.name AS datasheet_name,
       s.id AS stratagem_id,
       s.name AS stratagem_name,
       s.type,
       s.cpCost,
       s.phase,
       s.description
FROM datasheets d
LEFT JOIN datasheets_stratagems ds ON ds.DatasheetId = d.Id
LEFT JOIN stratagems s ON s.id = ds.stratagemId
WHERE d.Id = '...';
```

### 5. Get all weapons for a datasheet
```sql
SELECT d.name AS datasheet_name,
       dw.name AS weapon_name,
       dw.type,
       dw.range,
       dw.a,
       dw.bsWs,
       dw.s,
       dw.ap,
       dw.d
FROM datasheets d
LEFT JOIN datasheets_wargear dw ON dw.DatasheetId = d.Id
WHERE d.Id = '...';
```

### 6. Get units by faction and role
```sql
SELECT d.Id, d.name, d.role
FROM datasheets d
WHERE d.factionId = 'TYR'
ORDER BY d.name;
```

### 7. Get all detachments and their abilities
```sql
SELECT dt.Id AS detachment_id,
       dt.name AS detachment_name,
       da.Id AS ability_id,
       da.name AS ability_name
FROM detachments dt
LEFT JOIN detachment_abilities da ON da.detachmentId = dt.Id
WHERE dt.factionId = 'TYR';
```

---

## Notes for agent behavior

### When writing SQL

- Prefer `LEFT JOIN` unless you explicitly want only matched rows
- Use table aliases for readability: `d`, `f`, `da`, `a`, `s`, `dt`
- Treat string IDs as exact-string keys; do not assume integers
- Many tables are flattened JSON-derived tables; do not expect normalized foreign keys or unique constraints
- When a field name may be pluralized or case-sensitive, inspect the table definition before writing final SQL

### Good fallback strategy for ambiguous joins

When a key name looks like it should match but does not, check:
- `Id` vs `id`
- `FactionId` vs `factionId`
- `DatasheetId` vs `DatasheetId`
- `DetachmentId` vs `detachmentId`
- `attachedId` vs `LeaderId`

### SQL safety

- Quote identifiers with double quotes when needed, e.g. `"datasheets"`
- Use `SELECT` with explicit columns rather than `SELECT *` in agent-generated queries unless the request is intentionally broad

---

## Quick semantic cheat sheet

- `datasheets` = unit cards
- `datasheets_abilities` = unit -> abilities
- `datasheets_stratagems` = unit -> stratagems
- `datasheets_keywords` = unit -> keywords
- `datasheets_wargear` = unit -> weapons and gear
- `datasheets_models` = unit -> model stat blocks
- `detachments` = detachment definitions
- `detachment_abilities` = detachment -> abilities
- `stratagems` = global stratagem catalog
- `enhancements` = upgrades and enhancements
- `abilities` = standalone ability catalog
- `factions` = top-level faction lookup
- `source` = publication metadata

---

## Final instruction for SQL-generation agents

Use this database as a denormalized rules reference. Prefer joining from the main entity table (`datasheets`, `detachments`, `factions`, `stratagems`, `abilities`) to the associative tables that carry the actual relationship or text payload. Always respect the live naming pattern of the columns and join on exact identifier fields rather than assuming normalized foreign keys.

This schema is optimized for analytical reading and rule lookups, not for transactional relational integrity.
