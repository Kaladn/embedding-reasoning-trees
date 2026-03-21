# Cascade 6-1-6 — Clearbox Integration Steps

Self-contained plugin. No external dependencies beyond stdlib + FastAPI/Pydantic.

## Plugin contents

```
cascade_616/
├── __init__.py              # VERSION = "1.0.0"
├── config.py                # DEFAULTS + load_config()
├── manifest.json            # Plugin metadata
├── CLEARBOX_INTEGRATION.md  # This file
├── api/
│   ├── __init__.py
│   ├── models.py            # Pydantic request/response
│   ├── router.py            # FastAPI router + /panel endpoint
│   └── panel.html           # Interactive 6-1-6 explorer UI
└── core/
    ├── __init__.py
    ├── engine.py             # Prediction engine (self-contained)
    └── cell_reader.py        # Binary cell store reader (stdlib only)
```

## What you need

1. This `cascade_616/` directory → copy into Clearbox's `plugins/` folder
2. An `evidence.bin` + `evidence.bin.idx` file (the binary cell store)
3. A config entry pointing to the store path

---

## Step-by-step integration

### 1. Copy plugin

```
Copy:  cascade_616/
To:    {clearbox_workspace}/plugins/cascade_616/
```

No other files needed. The plugin bundles its own binary cell reader.

### 2. Add config entry

In `clearbox.config.json`:

```json
{
    "cascade_616": {
        "enabled": true,
        "store_path": "D:/Clearbox AI Data Root/cascade_616/evidence.bin"
    }
}
```

The store_path must point to an evidence.bin file with its .idx sidecar alongside it.

### 3. Mount router in bridge server

In `bridges/clearbox_bridge_server.py`, add with the other plugin mounts:

```python
# ── Mount Cascade 6-1-6 plugin (optional, non-fatal) ────────
try:
    from cascade_616.api.router import router as cascade_616_router
    app.include_router(cascade_616_router)
    LOGGER.info("Cascade 6-1-6 plugin mounted at /api/cascade-616")
except ImportError:
    LOGGER.info("Cascade 6-1-6 plugin not available (optional)")
```

### 4. Add chat mode handler (optional)

If you want cascade predictions available in the chat UI, add a mode branch
in `/api/chat/send`:

```python
if req.mode == "cascade":
    try:
        from cascade_616.api.router import get_engine
        engine = get_engine()

        result = await asyncio.to_thread(
            engine.full_context,
            anchor=msg.strip().lower(),
            k=req.topk or 5,
        )

        if "error" in result:
            response_text = f"No data for '{msg}'"
        else:
            # Build readable summary from positional neighbors
            lines = [f"**{result['display']}** ({result['symbol']})"]
            for pos_label, items in sorted(result.get("positions", {}).items()):
                tokens = ", ".join(f"{it['token']}({it['count']})" for it in items[:5])
                lines.append(f"  {pos_label}: {tokens}")
            response_text = "\n".join(lines)

        return ChatSendResponse(
            mode="cascade",
            source="cascade_616",
            response=response_text,
        )
    except ImportError:
        return ChatSendResponse(
            mode="cascade",
            source="cascade_616",
            response="Cascade 6-1-6 plugin not installed.",
            error={"type": "not_available"}
        )
```

### 5. Add UI mode option (optional)

In `ui/clearbox_ai_production.js`, add to the mode dropdown:

```html
<option value="cascade">Cascade 6-1-6</option>
```

### 6. Add status indicator (optional)

```javascript
async function updateCascadeStatus() {
    try {
        const r = await fetch('http://127.0.0.1:5050/api/cascade-616/status');
        const data = await r.json();
        const dot = document.getElementById('cascade-dot');
        dot.className = data.enabled ? 'status-dot green' : 'status-dot red';
    } catch {
        // Plugin not available
    }
}
```

---

## API endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/cascade-616/status` | GET | Health + cell count |
| `/api/cascade-616/panel` | GET | Interactive explorer UI |
| `/api/cascade-616/predict` | POST | Top-K next tokens at +1 |
| `/api/cascade-616/predict/previous` | POST | Top-K previous at -1 |
| `/api/cascade-616/chain/forward` | POST | Forward chain traversal |
| `/api/cascade-616/chain/backward` | POST | Backward chain traversal |
| `/api/cascade-616/context` | POST | Full 6-1-6 (12 positions) |

All POST endpoints accept:
```json
{"anchor": "word", "k": 5, "steps": 6}
```

---

## What this plugin does NOT do

- Does not write to disk (read-only evidence store)
- Does not import from other plugins
- Does not call the LLM
- Does not modify the bridge, UI, or security layers
- Does not require the lexicon at runtime (symbols are baked into the evidence store)

## What this plugin needs from Clearbox

- FastAPI app to mount its router
- Config file for store_path
- Nothing else
