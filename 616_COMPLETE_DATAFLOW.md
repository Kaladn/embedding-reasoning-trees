# 6-1-6 Complete Data Flow & Architecture

## 🌳 The Tree Mental Model

```
                    CLEARBOX (root/anchor)
                          |
        ┌─────────────────┴─────────────────┐
        |                                   |
    ROOTS (before)                    BRANCHES (after)
        |                                   |
   Position -6 to -1                   Position +1 to +6
        |                                   |
   ┌────┴────┬────┬────┬────┬────┐    ┌────┴────┬────┬────┬────┬────┐
  -6   -5   -4   -3   -2   -1      +1   +2   +3   +4   +5   +6
   |    |    |    |    |    |        |    |    |    |    |    |
 [Top K words with counts at each position]

Example at position +1 (first branch):
  conservation: 15 times
  ecosystem: 12 times
  protect: 8 times
  ...
```

---

## 📋 Complete Data Flow

```
┌────────────────────────────────────────────────────────────────┐
│ PHASE 1: INTAKE                                                │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  User uploads "doc1_clearbox_ecology.txt"                       │
│         ↓                                                       │
│  POST /api/library/citations/create                           │
│    {content: "...", filename: "..."}                          │
│         ↓                                                       │
│  MapManager.create_citation()                                 │
│    - Hash content → cite_id                                   │
│    - Store to SQLite + JSON                                   │
│         ↓                                                       │
│  POST /api/library/maps/create                                │
│    {cite_id: "...", window_size: 6}                          │
│         ↓                                                       │
│  Call Bridge: POST /api/map                                   │
│    {text: "...", source: "..."}                              │
│                                                                 │
└─────────────────────┬──────────────────────────────────────────┘
                      ↓
┌────────────────────────────────────────────────────────────────┐
│ PHASE 2: 6-1-6 MAPPING (Bridge Server - UNCHANGED)            │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  bridge.map_text(text, source)                                │
│    │                                                            │
│    ├→ Split by paragraph (block boundaries)                   │
│    │                                                            │
│    ├→ For each paragraph:                                     │
│    │    ├→ Tokenize → ["clearboxs", "provide", "carbon", ...]  │
│    │    │                                                       │
│    │    ├→ Walk through tokens as anchors:                    │
│    │    │   Position 0: 0-1-6 (0 before, 1 anchor, 6 after)  │
│    │    │   Position 1: 1-1-6 (1 before, 1 anchor, 6 after)  │
│    │    │   Position 2: 2-1-6 (2 before, 1 anchor, 6 after)  │
│    │    │   ...                                                 │
│    │    │   Position 6+: 6-1-6 (full window)                  │
│    │    │   ...                                                 │
│    │    │   End: 6-1-5, 6-1-4, ... (countdown)                │
│    │    │                                                       │
│    │    └→ For each anchor position:                          │
│    │         ├→ Extract 6 words before                        │
│    │         ├→ Extract 6 words after                         │
│    │         ├→ Count co-occurrences by position              │
│    │         └→ Build context structure                       │
│    │                                                            │
│    └→ Aggregate across all paragraphs                         │
│                                                                 │
│  Returns 6-1-6 Map Structure:                                 │
│  {                                                              │
│    "items": {                                                   │
│      "clearbox": {                                               │
│        "before": {                                             │
│          "1": [                                                 │
│            {"token": "carbon", "count": 5, ...},              │
│            {"token": "ecosystem", "count": 3, ...}            │
│          ],                                                     │
│          "2": [...],                                           │
│          ...                                                    │
│          "6": [...]                                            │
│        },                                                       │
│        "after": {                                              │
│          "1": [                                                 │
│            {"token": "conservation", "count": 4, ...},        │
│            {"token": "protect", "count": 2, ...}              │
│          ],                                                     │
│          ...                                                    │
│          "6": [...]                                            │
│        },                                                       │
│        "symbol": "0x...",                                      │
│        "lexicon_payload": {...},                              │
│        "frequency": 0  ← (unused, from lexicon metadata)      │
│      }                                                          │
│    },                                                           │
│    "total_tokens": 87,                                         │
│    "stats": {...}                                              │
│  }                                                              │
│                                                                 │
└─────────────────────┬──────────────────────────────────────────┘
                      ↓
┌────────────────────────────────────────────────────────────────┐
│ PHASE 3: INDEXING (MapManager - MY CODE)                      │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  MapManager.create_map(cite_id, map_data)                     │
│    │                                                            │
│    ├→ Store full map to JSON file                             │
│    │   CHAT_MAPS_DIR/map_abc123.json                          │
│    │                                                            │
│    ├→ Store metadata to SQLite                                │
│    │   maps table: (map_id, cite_id, stats, ...)             │
│    │                                                            │
│    └→ Build inverted anchor index:                            │
│         _index_anchors(map_id, cite_id, anchors)              │
│                                                                 │
│  THE COUNTING ALGORITHM:                                       │
│  ═══════════════════════════════════════                       │
│                                                                 │
│  For each anchor (e.g., "clearbox"):                            │
│                                                                 │
│    total_count = 0                                             │
│                                                                 │
│    # Sum counts from BEFORE windows (roots)                   │
│    for position in ['1', '2', '3', '4', '5', '6']:           │
│      for neighbor in anchor["before"][position]:              │
│        total_count += neighbor["count"]                       │
│                                                                 │
│    # Sum counts from AFTER windows (branches)                 │
│    for position in ['1', '2', '3', '4', '5', '6']:           │
│      for neighbor in anchor["after"][position]:               │
│        total_count += neighbor["count"]                       │
│                                                                 │
│    # Store in anchor_index table                              │
│    INSERT INTO anchor_index                                    │
│      (anchor, cite_id, map_id, count)                         │
│    VALUES                                                       │
│      ('clearbox', cite_id, map_id, total_count)                 │
│                                                                 │
│  RESULT:                                                        │
│  ┌──────────────────────────────────────────────────┐         │
│  │ anchor_index table                                │         │
│  ├──────────┬─────────┬─────────┬────────┤         │
│  │ anchor   │ cite_id │ map_id  │ count  │         │
│  ├──────────┼─────────┼─────────┼────────┤         │
│  │ clearbox   │ cite_6c │ map_797 │ 33     │ ← SUM  │
│  │ carbon   │ cite_6c │ map_797 │ 78     │         │
│  │ conserve │ cite_6c │ map_797 │ 56     │         │
│  └──────────┴─────────┴─────────┴────────┘         │
│                                                                 │
└─────────────────────┬──────────────────────────────────────────┘
                      ↓
┌────────────────────────────────────────────────────────────────┐
│ PHASE 4: QUERY & SEARCH                                       │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  User searches: ["clearbox", "carbon"]                          │
│         ↓                                                       │
│  POST /api/library/query                                      │
│    {keywords: ["clearbox", "carbon"], limit: 10}               │
│         ↓                                                       │
│  MapManager.query_maps(keywords, limit)                       │
│                                                                 │
│  SQL QUERY:                                                    │
│  ═══════════════════════════════════════                       │
│                                                                 │
│  SELECT                                                         │
│    ai.cite_id,                                                 │
│    MAX(ai.map_id) AS map_id,                                  │
│    SUM(ai.count) AS total_mentions,    ← SUM COUNTS          │
│    COUNT(DISTINCT ai.anchor) AS matched_count,                │
│    GROUP_CONCAT(DISTINCT ai.anchor) AS matched_anchors        │
│  FROM anchor_index ai                                          │
│  WHERE ai.anchor IN ('clearbox', 'carbon')  ← KEYWORD MATCH    │
│  GROUP BY ai.cite_id                      ← ONE PER CITATION  │
│  ORDER BY total_mentions DESC                                 │
│  LIMIT 10                                                      │
│                                                                 │
│  SCORING:                                                       │
│  ═══════════════════════════════════════                       │
│                                                                 │
│  For each result:                                              │
│    keyword_coverage = matched_count / len(keywords)           │
│    score = (keyword_coverage * 100) + total_mentions          │
│                                                                 │
│  Example:                                                       │
│    doc1: matches "clearbox" (33) + "carbon" (78)                │
│      → matched_count = 2                                       │
│      → total_mentions = 111                                    │
│      → keyword_coverage = 2/2 = 1.0                           │
│      → score = (1.0 * 100) + 111 = 211                        │
│                                                                 │
│  RESULT:                                                        │
│  [                                                              │
│    {                                                            │
│      "cite_id": "cite_6c345ce7...",                           │
│      "filename": "doc1_clearbox_ecology.txt",                   │
│      "score": 211.0,                                          │
│      "matched_anchors": ["clearbox", "carbon"],                 │
│      "total_mentions": 111                                     │
│    },                                                           │
│    ...                                                          │
│  ]                                                              │
│                                                                 │
└─────────────────────┬──────────────────────────────────────────┘
                      ↓
┌────────────────────────────────────────────────────────────────┐
│ PHASE 5: VISUALIZATION (3D Graph Explorer)                    │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  User Interface:                                               │
│                                                                 │
│  ┌────────────────────────────────────────────────────┐       │
│  │                                                      │       │
│  │          CARBON (78)                                │       │
│  │              ↖                                       │       │
│  │                ↖                                     │       │
│  │         [count: 5 at pos -1]                        │       │
│  │                  ↖                                   │       │
│  │                    ↖                                 │       │
│  │               ╔════════════╗                        │       │
│  │               ║   CLEARBOX   ║  ← Center (clicked)    │       │
│  │               ║    (33)    ║                         │       │
│  │               ╚════════════╝                        │       │
│  │                    ↘                                 │       │
│  │                      ↘                               │       │
│  │              [count: 4 at pos +1]                   │       │
│  │                        ↘                             │       │
│  │                          ↘                           │       │
│  │                    CONSERVATION (56)                 │       │
│  │                                                      │       │
│  ├──────────────────────────────────────────────────┤       │
│  │ Context Inspector:                                  │       │
│  │                                                      │       │
│  │ Before (-6 to -1):                                  │       │
│  │   Pos -1: carbon (5), sustainable (3), ...         │       │
│  │   Pos -2: ecosystem (4), natural (2), ...          │       │
│  │   ...                                                │       │
│  │                                                      │       │
│  │ After (+1 to +6):                                   │       │
│  │   Pos +1: conservation (4), protect (3), ...       │       │
│  │   Pos +2: species (2), habitat (2), ...            │       │
│  │   ...                                                │       │
│  │                                                      │       │
│  │ Total co-occurrence weight: 33                      │       │
│  └──────────────────────────────────────────────────┘       │
│                                                                 │
│  INTERACTIONS:                                                 │
│  • Click any neighbor node → re-center graph on that word     │
│  • Search keywords → highlight matching paths                 │
│  • Filter by document → show/hide nodes                       │
│  • Zoom in/out → explore locally or see full network          │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## 🔢 The Math Verification

### **What We're Counting**

For anchor "clearbox" with this structure:
```json
{
  "before": {
    "1": [{"token": "carbon", "count": 5}, {"token": "sustainable", "count": 3}],
    "2": [{"token": "ecosystem", "count": 4}],
    ...
  },
  "after": {
    "1": [{"token": "conservation", "count": 4}],
    "2": [{"token": "protect", "count": 3}],
    ...
  }
}
```

**My calculation:**
```python
total_count = 5 + 3 + 4 + 4 + 3 + ... = 33
```

This represents: **Total co-occurrence weight** - how strongly "clearbox" is connected to its context words across all 12 positions (6 before + 6 after).

### **Is This Correct for the Tree Model?**

**YES**, because:
- Each position's count represents how many times that relationship appears
- Summing gives the total "strength" of this anchor's connections
- Higher count = more context = stronger node in the graph
- This is exactly the "branches feeding from roots" - the counts ARE the branch weights

### **What This Enables**

1. **Search Relevance**: Documents with higher total_mentions rank higher
2. **Semantic Similarity**: Shared high-count neighbors indicate related docs
3. **Graph Visualization**: Edge thickness = count value
4. **Predictive Text**: Most common next words at each position

---

## 🎨 UI Component Breakdown

### **Left Panel: Document Library**
```
┌─────────────────────────────┐
│ 🔍 Search: [clearbox carbon] │
│                             │
│ 📚 Documents (3)            │
│ ┌─────────────────────────┐ │
│ │ 🟢 Clearbox Ecology       │ │
│ │    68 anchors • 98%     │ │
│ └─────────────────────────┘ │
│ ┌─────────────────────────┐ │
│ │ 🔵 Biodiversity         │ │
│ │    73 anchors • 98%     │ │
│ └─────────────────────────┘ │
│ ┌─────────────────────────┐ │
│ │ 🟡 Climate Policy       │ │
│ │    95 anchors • 96%     │ │
│ └─────────────────────────┘ │
└─────────────────────────────┘
```

### **Center Panel: 3D Graph**
```
Force-directed graph with:
- Nodes = Anchors (size ∝ total_count)
- Edges = Co-occurrences (thickness ∝ count at position)
- Colors = Document source
- Layout = Positions clustered by semantic similarity
```

### **Right Panel: Node Inspector**
```
┌─────────────────────────────┐
│ Selected: CLEARBOX (33)       │
│                             │
│ Before Context:             │
│ [-6] ecosystem (2)          │
│ [-5] natural (1)            │
│ [-4] ...                    │
│ [-3] ...                    │
│ [-2] ...                    │
│ [-1] carbon (5) ⭐         │
│                             │
│ After Context:              │
│ [+1] conservation (4) ⭐    │
│ [+2] protect (3)            │
│ [+3] ...                    │
│ [+4] ...                    │
│ [+5] ...                    │
│ [+6] habitat (2)            │
│                             │
│ Documents: 3                │
│ Total weight: 33            │
└─────────────────────────────┘
```

---

## ✅ Algorithm Correctness

### **Does the Bridge Calculate This Correctly?**

The bridge builds windows and counts co-occurrences at each position. ✅

### **Does MapManager Index This Correctly?**

My `_index_anchors` sums all position counts to get total weight. ✅

**BUT**: Should I also store **per-position counts** for graph visualization?

**ANSWER**: Currently, I only store the SUM in anchor_index. For the 3D graph, I need the FULL map data (which is stored in the JSON file). The anchor_index is just for SEARCH, not visualization.

### **Proposed Enhancement**

For graph visualization, I need to:
1. **Query**: Use anchor_index to find relevant documents (FAST)
2. **Retrieve**: Load full map JSON to get per-position counts (DETAILED)
3. **Visualize**: Show position-specific edges with counts

**This requires NO changes** - the full map data is already stored in JSON!

---

## 📊 Backend Stack Summary

| Layer | Component | Purpose | Status |
|-------|-----------|---------|--------|
| **Storage** | SQLite + JSON | Metadata + full maps | ✅ Working |
| **Mapping** | Bridge Server | 6-1-6 algorithm | ✅ UNCHANGED |
| **Indexing** | MapManager | Anchor index for search | ✅ Working |
| **API** | FastAPI endpoints | REST interface | ✅ Working |
| **Frontend** | Graph Explorer | 3D visualization | 🔨 Demo ready |

---

## 🎯 Next Steps

1. **Verify counting is correct** ✅ (Math checks out)
2. **Test query results** ✅ (Returns 2 unique docs as expected)
3. **Build working graph visualization** (Next: D3.js implementation)
4. **Load full map for clicked node** (Read JSON file on demand)

---

**Ready to proceed with D3.js graph implementation?**
