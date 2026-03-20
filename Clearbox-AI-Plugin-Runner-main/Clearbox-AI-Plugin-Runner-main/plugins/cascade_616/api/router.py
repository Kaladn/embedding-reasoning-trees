"""Cascade 6-1-6 Plugin — FastAPI router.

Self-contained. No imports from other plugins.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse
from cascade_616.api.models import (
    PredictRequest,
    PredictResponse,
    ChainResponse,
    ContextResponse,
    StatusResponse,
)
from cascade_616 import VERSION

_PANEL_HTML = Path(__file__).parent / "panel.html"

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/cascade-616", tags=["cascade_616"])


@router.get("/panel", include_in_schema=False)
async def serve_panel():
    """Serve the explorer UI."""
    return FileResponse(_PANEL_HTML, media_type="text/html")


# ── Singleton Engine ────────────────────────────────────────

_engine = None


def get_engine():
    """Get or create the engine singleton. Lazy-loaded."""
    global _engine
    if _engine is None:
        from cascade_616.core.engine import CascadeEngine
        from cascade_616.config import load_config

        cfg = load_config()
        _engine = CascadeEngine(
            store_path=cfg.get("store_path", ""),
            lexicon_path=cfg.get("lexicon_path", ""),
        )
    return _engine


# ── Endpoints ───────────────────────────────────────────────

@router.get("/status", response_model=StatusResponse)
async def cascade_status():
    """Health and readiness."""
    try:
        engine = get_engine()
        s = await asyncio.to_thread(engine.stats)
        return StatusResponse(
            version=VERSION,
            enabled=True,
            cells=s.get("cells", 0),
            store_path=s.get("store_path", ""),
        )
    except Exception as e:
        logger.error("cascade_616 status error: %s", e, exc_info=True)
        return StatusResponse(
            version=VERSION,
            error={"type": "status_error", "message": str(e)},
        )


@router.post("/predict", response_model=PredictResponse)
async def cascade_predict(req: PredictRequest):
    """Predict next tokens for an anchor."""
    try:
        engine = get_engine()
        preds = await asyncio.to_thread(engine.predict_next, req.anchor, req.k)
        return PredictResponse(anchor=req.anchor, predictions=preds)
    except Exception as e:
        logger.error("cascade_616 predict error: %s", e, exc_info=True)
        return PredictResponse(
            anchor=req.anchor,
            error={"type": "predict_error", "message": str(e)},
        )


@router.post("/predict/previous", response_model=PredictResponse)
async def cascade_predict_previous(req: PredictRequest):
    """Predict previous tokens for an anchor."""
    try:
        engine = get_engine()
        preds = await asyncio.to_thread(engine.predict_previous, req.anchor, req.k)
        return PredictResponse(anchor=req.anchor, predictions=preds)
    except Exception as e:
        return PredictResponse(
            anchor=req.anchor,
            error={"type": "predict_error", "message": str(e)},
        )


@router.post("/chain/forward", response_model=ChainResponse)
async def cascade_chain_forward(req: PredictRequest):
    """Forward chain: anchor -> top+1 -> top+1 -> ..."""
    try:
        engine = get_engine()
        chain = await asyncio.to_thread(engine.forward_chain, req.anchor, req.steps)
        return ChainResponse(anchor=req.anchor, direction="forward", chain=chain)
    except Exception as e:
        return ChainResponse(
            anchor=req.anchor, direction="forward",
            error={"type": "chain_error", "message": str(e)},
        )


@router.post("/chain/backward", response_model=ChainResponse)
async def cascade_chain_backward(req: PredictRequest):
    """Backward chain: ... -> top-1 -> anchor."""
    try:
        engine = get_engine()
        chain = await asyncio.to_thread(engine.backward_chain, req.anchor, req.steps)
        return ChainResponse(anchor=req.anchor, direction="backward", chain=chain)
    except Exception as e:
        return ChainResponse(
            anchor=req.anchor, direction="backward",
            error={"type": "chain_error", "message": str(e)},
        )


@router.post("/context", response_model=ContextResponse)
async def cascade_full_context(req: PredictRequest):
    """Full 6-1-6 context view for an anchor."""
    try:
        engine = get_engine()
        ctx = await asyncio.to_thread(engine.full_context, req.anchor, req.k)
        if "error" in ctx:
            return ContextResponse(
                anchor=req.anchor,
                error={"type": "not_found", "message": ctx["error"]},
            )
        return ContextResponse(
            anchor=ctx["anchor"],
            symbol=ctx.get("symbol", ""),
            display=ctx.get("display", ""),
            category=ctx.get("category", ""),
            total_count=ctx.get("total_count", 0),
            positions=ctx.get("positions", {}),
        )
    except Exception as e:
        return ContextResponse(
            anchor=req.anchor,
            error={"type": "context_error", "message": str(e)},
        )
