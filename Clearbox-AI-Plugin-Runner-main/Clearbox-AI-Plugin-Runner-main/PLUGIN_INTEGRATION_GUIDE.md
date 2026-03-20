# Clearbox AI — Deterministic Plugin Integration Guide

**Version**: 1.0
**Date**: 2026-02-16
**Derived from**: LakeSpeak v0 (the only shipping plugin)
**Rule**: Every statement below maps to real code that runs in production. No theory.

---

## Overview

A Clearbox AI plugin is a Python package that:

1. Lives in its own top-level directory (e.g. `lakespeak/`, `myplugin/`)
2. Exposes a FastAPI `APIRouter` at `{plugin}/api/router.py`
3. Owns dedicated WriteZones for disk I/O
4. Is mounted by the bridge server via a **non-fatal** try/import
5. Degrades gracefully when absent — the system boots without it

There are exactly **7 integration surfaces**. Miss one and the plugin won't work.
Add extras and you've coupled too tightly.

---

## The 7 Integration Surfaces

```
                  ┌─────────────────────────────────────────────┐
                  │               Bridge Server (:5050)         │
                  │  clearbox_bridge_server.py                    │
                  │                                             │
                  │  1. try: import router → include_router()   │
                  │  2. Mode handler in /api/chat/send          │
                  └──────┬──────────────────────┬───────────────┘
                         │                      │
                    3. API Router          4. Config
                    {plugin}/api/           clearbox.config.json
                    router.py              "{plugin}": { ... }
                         │
                    ┌────┴────┐
                    │ Engine  │  ← Your core logic
                    └────┬────┘
                         │
              ┌──────────┼──────────┐
              │          │          │
         5. Data      6. Security   7. UI
         Paths        Gateway       Integration
         data_paths   gateway.py    JS mode +
         .py                        status dot
```

---

## Surface 1: Router Mount (bridge server)

**File**: `bridges/clearbox_bridge_server.py`
**Pattern**: Non-fatal try/import

```python
# ── Mount {PluginName} plugin (optional, non-fatal) ────────
try:
    from {plugin}.api.router import router as {plugin}_router
    app.include_router({plugin}_router)
    LOGGER.info("{PluginName} plugin mounted at /api/{plugin}")
except ImportError:
    LOGGER.info("{PluginName} plugin not available (optional)")
```

**Rules:**
- The `try/except ImportError` is **mandatory**. The bridge must boot without your plugin
- Place this in the startup section alongside other plugin mounts (~line 1493+)
- Your router defines its own prefix: `APIRouter(prefix="/api/{plugin}")`
- The bridge server never imports anything else from your plugin directly

**LakeSpeak reference**: `clearbox_bridge_server.py:1493-1499`

---

## Surface 2: Chat Mode Handler (bridge server)

**File**: `bridges/clearbox_bridge_server.py`
**Location**: Inside the `/api/chat/send` endpoint

If your plugin provides a chat mode (like LakeSpeak's "grounded"), add a mode branch:

```python
if req.mode == "{your_mode}":
    try:
        from {plugin}.api.router import get_engine
        engine = get_engine()

        result = await asyncio.to_thread(
            engine.query,
            query=msg,
            mode="{your_mode}",
            topk=req.topk,
            session_id=req.session_id,
        )

        # ... process result, build response ...

        return ChatSendResponse(
            mode="{your_mode}",
            source="{plugin}",
            response=result.answer_text,
            citations=result.citations,
            # ...
        )
    except ImportError:
        return ChatSendResponse(
            mode="{your_mode}",
            source="{plugin}",
            response="{PluginName} plugin not installed.",
            error={"type": "not_available", "message": "{PluginName} not installed"}
        )
    except Exception as e:
        return ChatSendResponse(
            mode="{your_mode}",
            source="{plugin}",
            error={"type": "{plugin}_error", "message": str(e)}
        )
```

**Rules:**
- Lazy-import your engine inside the handler (not at module top)
- Always handle `ImportError` (plugin not installed) and `Exception` (plugin crashed)
- Return `ChatSendResponse` in all cases — never raise into the bridge
- Use `asyncio.to_thread()` for blocking operations

**LakeSpeak reference**: `clearbox_bridge_server.py:475-651`

---

## Surface 3: API Router (your plugin)

**File**: `{plugin}/api/router.py`

This is the only file the bridge imports from your plugin.

### Required structure

```python
"""Plugin API Router — FastAPI endpoints."""

from __future__ import annotations
import asyncio, logging
from fastapi import APIRouter
from {plugin}.api.models import (  # Your request/response models
    QueryRequest, QueryResponse, StatusResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/{plugin}", tags=["{plugin}"])

# ── Singleton Engine ─────────────────────────────────────────

_engine = None

def get_engine():
    """Get or create the engine singleton. Lazy-loaded."""
    global _engine
    if _engine is None:
        from {plugin}.core.engine import PluginEngine
        _engine = PluginEngine()
    return _engine

# ── Endpoints (minimum viable set) ──────────────────────────

@router.post("/query", response_model=QueryResponse)
async def plugin_query(req: QueryRequest):
    """Primary query endpoint."""
    try:
        engine = get_engine()
        result = await asyncio.to_thread(engine.query, query=req.query)
        return QueryResponse(source="{plugin}", response=result.text, ...)
    except Exception as e:
        logger.error("{PluginName} query error: %s", e, exc_info=True)
        return QueryResponse(error={"type": "{plugin}_error", "message": str(e)})

@router.get("/status", response_model=StatusResponse)
async def plugin_status():
    """Health/status endpoint. UI polls this."""
    try:
        engine = get_engine()
        return StatusResponse(version=VERSION, enabled=True, ...)
    except Exception as e:
        return StatusResponse(error={"type": "status_error", "message": str(e)})
```

### Required endpoints

| Endpoint | Method | Purpose | Required? |
|----------|--------|---------|-----------|
| `/api/{plugin}/status` | GET | Health + readiness. UI polls this | **Yes** |
| `/api/{plugin}/query` | POST | Primary operation | If plugin answers queries |
| `/api/{plugin}/ingest` | POST | Data intake | If plugin ingests data |
| `/api/{plugin}/reindex` | POST | Rebuild indexes | If plugin has indexes |

### Request/Response models

**File**: `{plugin}/api/models.py`

```python
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

class QueryRequest(BaseModel):
    query: str
    mode: str = "default"
    topk: int = 5
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    mode: str = ""
    source: str = "{plugin}"
    response: str = ""
    error: Optional[Dict[str, Any]] = None
    # ... plugin-specific fields with defaults ...

class StatusResponse(BaseModel):
    version: str = ""
    enabled: bool = False
    error: Optional[Dict[str, Any]] = None
    # ... plugin-specific metrics ...
```

**Rules:**
- All response fields must have defaults (never crash on serialization)
- Error is always `Optional[Dict[str, Any]]` — same shape across all plugins
- The `source` field always equals your plugin name

**LakeSpeak reference**: `lakespeak/api/router.py`, `lakespeak/api/models.py`

---

## Surface 4: Configuration

**File**: `clearbox.config.json` (workspace root)
**Your config module**: `{plugin}/config.py`

### Config pattern

```python
"""Plugin configuration — reads from clearbox.config.json."""

import json, logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    # ... all your tuning knobs with sensible defaults ...
}

def load_config(config_path: Path = None) -> Dict[str, Any]:
    """Load config. Returns merged: file values override defaults."""
    if config_path is None:
        try:
            from security.data_paths import SOURCE_ROOT
            config_path = SOURCE_ROOT / "clearbox.config.json"
        except ImportError:
            config_path = Path("clearbox.config.json")

    config = dict(DEFAULTS)
    try:
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                full = json.load(f)
            section = full.get("{plugin}", {})
            config.update(section)
    except Exception as e:
        logger.warning("Failed to load {PluginName} config: %s", e)
    return config
```

### In clearbox.config.json

```json
{
    "existing_keys": "...",
    "{plugin}": {
        "enabled": true,
        "custom_param": 42
    }
}
```

**Rules:**
- Your key in clearbox.config.json is your plugin directory name
- Every parameter has a default — config section can be entirely absent
- Use `try/except ImportError` for `security.data_paths` (plugin may run standalone)
- Never modify clearbox.config.json programmatically

**LakeSpeak reference**: `lakespeak/config.py`

---

## Surface 5: Data Paths

**File**: `security/data_paths.py`

Every plugin gets a directory tree under `%LOCALAPPDATA%\ClearboxAI\`.

### Add your paths

```python
# ── {PluginName} ──────────────────────────────────────────────
{PLUGIN}_DIR         = CLEARBOX_DATA_ROOT / "{plugin}"
{PLUGIN}_INDEX_DIR   = {PLUGIN}_DIR / "index"       # if you have indexes
{PLUGIN}_EVENTS_DIR  = {PLUGIN}_DIR / "events"      # if you log events
{PLUGIN}_EVAL_DIR    = {PLUGIN}_DIR / "eval"         # if you run evals
```

### Register for auto-creation

Add to the `for _d in [...]` loop that creates directories on import:

```python
for _d in [
    # ... existing dirs ...
    {PLUGIN}_INDEX_DIR, {PLUGIN}_EVENTS_DIR, {PLUGIN}_EVAL_DIR,
]:
    _d.mkdir(parents=True, exist_ok=True)
```

**Rules:**
- Your top-level directory is `CLEARBOX_DATA_ROOT / "{plugin}"`
- Sub-directories follow the pattern: `index/`, `events/`, `eval/`
- All paths are constants (not computed at runtime)
- All directories are auto-created on module import

**LakeSpeak reference**: `security/data_paths.py:59-65, 82-83`

---

## Surface 6: Security Gateway

**File**: `security/gateway.py`

### Step A: Add WriteZones

```python
class WriteZone(enum.Enum):
    # ... existing zones ...

    # ── {PluginName} zones ────────────────────────────────────
    {PLUGIN}_INDEX   = "{plugin}_index"
    {PLUGIN}_EVENTS  = "{plugin}_events"
    {PLUGIN}_EVAL    = "{plugin}_eval"
```

### Step B: Grant caller permissions

Add your zones to the `"system"` and `"human"` permission sets:

```python
_CALLER_PERMISSIONS = {
    "system": {
        # ... existing ...
        WriteZone.{PLUGIN}_INDEX,
        WriteZone.{PLUGIN}_EVENTS,
        WriteZone.{PLUGIN}_EVAL,
    },
    "human": {
        # ... existing ...
        WriteZone.{PLUGIN}_INDEX,
        WriteZone.{PLUGIN}_EVENTS,
        WriteZone.{PLUGIN}_EVAL,
    },
}
```

**Do NOT add plugin zones to `"ai"`.** AI caller is restricted to ARTIFACTS only.

### Step C: Add zone path resolution

```python
def _zone_root(zone: WriteZone) -> Path:
    _map = {
        # ... existing ...
        WriteZone.{PLUGIN}_INDEX:   CLEARBOX_DATA_ROOT / "{plugin}" / "index",
        WriteZone.{PLUGIN}_EVENTS:  CLEARBOX_DATA_ROOT / "{plugin}" / "events",
        WriteZone.{PLUGIN}_EVAL:    CLEARBOX_DATA_ROOT / "{plugin}" / "eval",
    }
    return _map[zone]
```

### Step D: Use gateway in your plugin code

```python
def _write_data(self, filename: str, payload: str) -> None:
    """Write through gateway (DPAPI + audit)."""
    try:
        from security.gateway import gateway, WriteZone
        result = gateway.write(
            caller="system",
            zone=WriteZone.{PLUGIN}_INDEX,
            name=filename,
            content=payload,
        )
        if not result.success:
            raise OSError(f"Gateway write failed: {result.error}")
    except ImportError:
        # Fallback: direct write (no DPAPI, no audit)
        from security.secure_storage import secure_json_dump
        secure_json_dump(self._path / filename, json.loads(payload))
```

**Rules:**
- ALL disk writes go through `gateway.write()` or `gateway.append()`
- Caller is always `"system"` for plugin internal operations
- Every write gets DPAPI encryption + audit trail automatically
- Always handle `ImportError` for standalone operation
- Never write outside your zone's root directory (gateway enforces this)

**LakeSpeak reference**: `security/gateway.py:64-66, 96-98, 143-145`; `lakespeak/events/training_logger.py:44-80`

---

## Surface 7: UI Integration

**File**: `ui/clearbox_ai_production.js`

### Step A: Add chat mode option

In the mode dropdown HTML:

```html
<option value="{your_mode}">{PluginName}</option>
```

### Step B: Add mode handler in sendGpt()

```javascript
if (mode === '{your_mode}') {
    const bridgeUrl = 'http://127.0.0.1:5050/api/chat/send';
    const res = await fetch(bridgeUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
            message: prompt,
            mode: '{your_mode}',
            session_id: 'ui',
            inject_context: false,
            topk: 8
        })
    });
    const data = await res.json();
    // Render response...
}
```

### Step C: Add status indicator

```javascript
// Status dot
async function update{PluginName}Status() {
    try {
        const r = await fetch('http://127.0.0.1:5050/api/{plugin}/status');
        const data = await r.json();
        const dot = document.getElementById('{plugin}-dot');
        dot.className = data.enabled ? 'status-dot green' : 'status-dot red';
    } catch {
        // Plugin not available — show gray dot
    }
}
```

### Step D: Add module toggle

```javascript
{ id: 'module-N', key: 'module_{plugin}_enabled', name: '{PluginName}' }
```

**LakeSpeak reference**: `ui/clearbox_ai_production.js:852-880, 4155-4200, 4215`

---

## Plugin Directory Structure (Template)

```
{plugin}/
├── __init__.py              # VERSION = "0.1.0", nothing else
├── config.py                # DEFAULTS + load_config()
├── schemas.py               # All dataclasses, version-pinned
├── api/
│   ├── __init__.py
│   ├── router.py            # APIRouter + get_engine() + endpoints
│   └── models.py            # Pydantic request/response models
├── core/                    # (or retrieval/, index/, etc.)
│   ├── __init__.py
│   └── engine.py            # Main logic, singleton
├── events/                  # Optional: logging/training
│   ├── __init__.py
│   └── logger.py
└── text/                    # Optional: shared utilities
    ├── __init__.py
    └── normalize.py
```

### `__init__.py` (plugin root)

```python
"""{PluginName} — one-line description."""
VERSION = "0.1.0"
```

Nothing else. No heavy imports. The bridge server imports `{plugin}.api.router` — if your `__init__.py` imports heavy deps, the bridge will fail to boot when those deps are missing.

### `schemas.py`

```python
"""Pinned schema definitions for {PluginName}.

All schemas are pure Python dataclasses — no external deps.
Schema versions are immutable: breaking change = new version string.
"""

from dataclasses import dataclass, field

QUERY_VERSION = "{plugin}_query@1"
ANSWER_VERSION = "{plugin}_answer@1"

@dataclass
class YourResult:
    schema_version: str = ANSWER_VERSION
    # ... fields with defaults ...
```

**Schema versioning rule**: The `@N` suffix is immutable. If you change the shape, create `@2` and keep `@1` parseable.

---

## Dependency Rules

### What your plugin CAN import from Clearbox AI

| Module | How | Why |
|--------|-----|-----|
| `security.data_paths` | Lazy (`try/except`) | Get directory constants |
| `security.gateway` | Lazy (`try/except`) | Governed writes |
| `security.secure_storage` | Lazy (`try/except`) | Fallback I/O |
| `bridges.clearbox_bridge` | Lazy (`try/except`) | Lexicon bridge access |
| `Conversations.citations.citation_intake` | Direct | If your plugin creates citations |

### What your plugin MUST NOT import

| Module | Why |
|--------|-----|
| `local_llm_server` | Circular: server imports plugin |
| `bridges.clearbox_bridge_server` | Circular: bridge imports plugin |
| `Conversations.threads` | Not your domain; use citation_intake if needed |
| `ui/*` | Plugins don't touch the UI directly |

### What imports your plugin

Only two places, both with `try/except`:

1. `bridges/clearbox_bridge_server.py` — mounts your router
2. `bridges/clearbox_bridge_server.py` — calls `get_engine()` in mode handler

That's it. If anything else imports your plugin, you've coupled too tightly.

---

## Citation Integration

If your plugin produces citations (like LakeSpeak does), funnel them through the existing intake gate:

```python
from Conversations.citations.citation_intake import (
    create_citation_record,
    SourceType,  # Literal["ui", "import", "linker", "system", "lakespeak"]
)
```

If your plugin needs a new source type, add it to the `SourceType` literal in `citation_intake.py`:

```python
SourceType = Literal["ui", "import", "linker", "system", "lakespeak", "{plugin}"]
```

Then add `"{plugin}"` to the `valid_sources` set in `create_citation_record()`.

**LakeSpeak reference**: `citation_intake.py:164, 192`

---

## Checklist: Adding a New Plugin

Use this as a sequential checklist. Each step depends on the previous.

### Phase 1: Plugin package (no integration yet)

- [ ] Create `{plugin}/` directory with `__init__.py` (VERSION only)
- [ ] Create `{plugin}/schemas.py` with pinned dataclasses
- [ ] Create `{plugin}/config.py` with DEFAULTS + load_config()
- [ ] Create `{plugin}/core/engine.py` with main logic
- [ ] Create `{plugin}/api/models.py` with Pydantic request/response
- [ ] Create `{plugin}/api/router.py` with APIRouter + get_engine() + endpoints
- [ ] Verify: `python -c "from {plugin}.api.router import router"` works

### Phase 2: Security layer

- [ ] Add data path constants to `security/data_paths.py`
- [ ] Add directories to the auto-creation loop in `data_paths.py`
- [ ] Add WriteZone entries to `security/gateway.py`
- [ ] Add zones to `"system"` and `"human"` permission sets
- [ ] Add zone→path mappings to `_zone_root()`
- [ ] Verify: `python -c "from security.gateway import WriteZone; print(WriteZone.{PLUGIN}_INDEX)"`

### Phase 3: Bridge integration

- [ ] Add router mount try/import to `bridges/clearbox_bridge_server.py`
- [ ] Add mode handler in `/api/chat/send` (if plugin answers queries)
- [ ] Verify: start bridge server, hit `GET /api/{plugin}/status`
- [ ] Verify: remove plugin directory, bridge still boots cleanly

### Phase 4: UI integration

- [ ] Add mode option to HTML dropdown
- [ ] Add mode handler in `sendGpt()` JavaScript
- [ ] Add status indicator dot + polling function
- [ ] Add module toggle entry
- [ ] Verify: UI shows mode, status dot updates, toggle works

### Phase 5: Citation integration (if applicable)

- [ ] Add source type to `citation_intake.py` SourceType literal
- [ ] Add to `valid_sources` set in `create_citation_record()`
- [ ] Verify: `create_citation_record(coord, source="{plugin}")` works

### Phase 6: Config integration (optional)

- [ ] Add `"{plugin}": { ... }` section to clearbox.config.json
- [ ] Verify: `load_config()` reads overrides correctly
- [ ] Verify: missing section uses all defaults

---

## Invariants (Things That Must Always Be True)

1. **Bridge boots without plugin**: `ImportError` is always caught
2. **All writes go through gateway**: No direct `open()` for persistent data
3. **All responses have defaults**: Pydantic models never fail to serialize
4. **Engine is lazy-singleton**: Heavy deps loaded on first call, not on import
5. **Config has full defaults**: Plugin works with zero config entries
6. **Schemas are version-pinned**: Breaking change = new `@N` version
7. **Plugin never imports server/bridge**: Only the reverse direction
8. **Status endpoint always responds**: Even when engine is broken (return error dict)
9. **Citations go through intake gate**: One path, one validation, one format
10. **Directories auto-create on import**: `data_paths.py` creates them eagerly

---

*This guide is derived from the LakeSpeak v0 integration. Every pattern described above is running in production. When in doubt, read the LakeSpeak code — it IS the reference implementation.*
