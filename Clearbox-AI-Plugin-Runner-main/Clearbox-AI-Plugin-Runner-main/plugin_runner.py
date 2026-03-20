"""
Clearbox Plugin Runner — Standalone plugin harness.
Discovers, mounts, tests, and inspects all Clearbox AI Studio plugins
without the bridge, LLM server, or UI server.

Usage:
    python plugin_runner.py [--port 9090] [--repo E:\\ForestAI-ROCm-7.1]
    python plugin_runner.py --fixtures          # inject test data
    python plugin_runner.py --check wolf_engine  # promotion checklist

Opens:  http://localhost:9090       (UI)
        http://localhost:9090/docs  (Swagger — all mounted routes)
"""
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportUnknownLambdaType=false, reportUnknownReturnType=false

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
import time
import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(message)s",
)
LOG = logging.getLogger("plugin_runner")


# ═══════════════════════════════════════════════════════════════
# 1. PLUGIN MANIFEST CONTRACT
# ═══════════════════════════════════════════════════════════════

@dataclass
class PluginManifest:
    """Canonical manifest — what every plugin must declare."""
    id: str = ""
    version: str = "?"
    display_name: str = ""
    description: str = ""
    mount_prefix: str = ""          # e.g. /api/wolf
    router_module: str = ""         # e.g. wolf_engine.api.router
    type: str = "router"            # router | service | embedded
    has_status: bool = False
    has_hooks: bool = False
    requires: list[str] = field(default_factory=list)   # other plugin ids
    config_section: str = ""        # key in forest.config.json
    data_dirs: list[str] = field(default_factory=list)
    # Discovered at runtime
    mounted: bool = False
    mount_path: str = ""
    error: str | None = None
    health: str = "unknown"         # ok | down | unknown | unmounted
    endpoints: list[dict] = field(default_factory=list)
    # Promotion checklist
    promotion: dict = field(default_factory=dict)


def read_manifest_file(plugin_dir: Path) -> dict:
    """Read manifest.json if present."""
    mf = plugin_dir / "manifest.json"
    if mf.exists():
        try:
            return json.loads(mf.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def build_manifest(pid: str, plugin_dir: Path, repo_root: Path) -> PluginManifest:
    """Build manifest from filesystem + optional manifest.json."""
    mf_data = read_manifest_file(plugin_dir)

    # Read VERSION from __init__.py
    version = "?"
    try:
        mod = importlib.import_module(pid)
        version = getattr(mod, "VERSION", "?")
    except Exception:
        pass

    # Detect router module
    router_module = ""
    if (plugin_dir / "api" / "router.py").exists():
        router_module = f"{pid}.api.router"
    elif (plugin_dir / "router.py").exists():
        router_module = f"{pid}.router"

    # Detect features
    has_hooks = False
    for hook_loc in [plugin_dir / "hooks.py", plugin_dir / "api" / "hooks.py"]:
        if hook_loc.exists():
            has_hooks = True
            break

    return PluginManifest(
        id=pid,
        version=version,
        display_name=mf_data.get("display_name", pid.replace("_", " ").title()),
        description=mf_data.get("description", ""),
        mount_prefix=mf_data.get("mount_prefix", ""),
        router_module=router_module,
        type=mf_data.get("type", "router"),
        has_status=True,  # assumed, verified at health check
        has_hooks=has_hooks,
        requires=mf_data.get("requires", []),
        config_section=mf_data.get("config_section", ""),
        data_dirs=mf_data.get("data_dirs", []),
    )


# ═══════════════════════════════════════════════════════════════
# 2. DISCOVERY + MOUNTING
# ═══════════════════════════════════════════════════════════════

def discover_plugins(plugins_dir: Path, repo_root: Path) -> list[PluginManifest]:
    results = []
    if not plugins_dir.is_dir():
        return results
    for entry in sorted(plugins_dir.iterdir()):
        if not entry.is_dir() or entry.name.startswith(("_", ".")):
            continue
        if not (entry / "__init__.py").exists():
            continue
        results.append(build_manifest(entry.name, entry, repo_root))
    return results


# Router import paths to try
_ROUTER_PATHS = [
    "{pid}.api.router",
    "{pid}.router",
]


def try_mount(app: FastAPI, m: PluginManifest) -> None:
    pid = m.id
    # Try explicit router_module first, then fallback patterns
    candidates = []
    if m.router_module:
        candidates.append(m.router_module)
    for tpl in _ROUTER_PATHS:
        c = tpl.format(pid=pid)
        if c not in candidates:
            candidates.append(c)

    for module_path in candidates:
        try:
            mod = importlib.import_module(module_path)
            router = getattr(mod, "router", None)
            if router is None:
                continue
            app.include_router(router)
            prefix = getattr(router, "prefix", "") or f"/api/{pid}"
            m.mounted = True
            m.mount_path = prefix
            LOG.info("  OK  %-24s  ->  %s", pid, prefix)
            return
        except Exception as exc:
            m.error = f"{module_path}: {exc}"
            continue

    if not m.mounted:
        m.health = "unmounted"
        LOG.warning("  --  %-24s  (no router found)", pid)


# ═══════════════════════════════════════════════════════════════
# 3. RUNTIME STATUS + HEALTH
# ═══════════════════════════════════════════════════════════════

_plugins: list[PluginManifest] = []
_boot_time = time.time()
_repo_root: Path = Path(".")
_scan_dirs: list[Path] = []


async def check_plugin_health(m: PluginManifest, base_url: str = "") -> str:
    """Check a plugin's health via its status/health endpoint."""
    if not m.mounted:
        return "unmounted"
    import httpx
    for suffix in ["/status", "/health"]:
        url = f"http://127.0.0.1:{_port}{m.mount_path}{suffix}"
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                r = await client.get(url)
                if r.status_code < 400:
                    return "ok"
        except Exception:
            continue
    return "down"


# ═══════════════════════════════════════════════════════════════
# 4. FIXTURE / TEST DATA INJECTION
# ═══════════════════════════════════════════════════════════════

FIXTURES = {
    "clearbox_ai_social": {
        "profile": {
            "method": "POST",
            "path": "/api/clearbox-social/profile",
            "body": {
                "display_name": "Test User",
                "bio": "Fixture-injected test profile for dev preview.",
                "interests": ["NLP", "local-first", "plugin dev"],
                "top_3_needs": ["peer sync", "signed identity", "marketplace"],
                "featured_plugins": ["wolf_engine", "lakespeak"],
            },
        },
        "listing_wolf": {
            "method": "POST",
            "path": "/api/clearbox-social/marketplace/listings",
            "body": {
                "plugin_id": "wolf_engine",
                "name": "Wolf Engine",
                "description": "Symbol-first cognitive architecture with governance",
                "plugin_version": "1.0.0",
                "category": "reasoning",
                "price": 0,
            },
        },
        "listing_lakespeak": {
            "method": "POST",
            "path": "/api/clearbox-social/marketplace/listings",
            "body": {
                "plugin_id": "lakespeak",
                "name": "GroveSpeak",
                "description": "Retrieval-augmented grounding (BM25 + dense)",
                "plugin_version": "0.1.0",
                "category": "retrieval",
                "price": 0,
            },
        },
        "listing_genesis": {
            "method": "POST",
            "path": "/api/clearbox-social/marketplace/listings",
            "body": {
                "plugin_id": "genesis_cite",
                "name": "Genesis Citation Tool",
                "description": "Read-only training corpus citation lookup",
                "plugin_version": "0.1.0",
                "category": "knowledge",
                "price": 0,
            },
        },
    },
    "genesis_cite": {
        "reload": {
            "method": "POST",
            "path": "/api/genesis/reload",
            "body": {},
        },
    },
    "help_system": {
        "status": {
            "method": "GET",
            "path": "/api/help/status",
            "body": None,
        },
    },
}


async def inject_fixtures(port: int, plugin_id: str = ""):
    """Inject test data into mounted plugins."""
    import httpx
    base = f"http://127.0.0.1:{port}"
    targets = {plugin_id: FIXTURES[plugin_id]} if plugin_id and plugin_id in FIXTURES else FIXTURES
    results = []

    async with httpx.AsyncClient(timeout=5.0) as client:
        for pid, fixtures in targets.items():
            for name, fx in fixtures.items():
                try:
                    if fx["method"] == "POST":
                        r = await client.post(base + fx["path"], json=fx.get("body", {}))
                    else:
                        r = await client.get(base + fx["path"])
                    ok = r.status_code < 400
                    results.append({"plugin": pid, "fixture": name, "ok": ok, "status": r.status_code})
                    LOG.info("  fixture %-20s %-20s -> %d", pid, name, r.status_code)
                except Exception as e:
                    results.append({"plugin": pid, "fixture": name, "ok": False, "error": str(e)})
                    LOG.warning("  fixture %-20s %-20s -> FAILED: %s", pid, name, e)
    return results


# ═══════════════════════════════════════════════════════════════
# 5. MOCK HOST SERVICES
# ═══════════════════════════════════════════════════════════════

def mount_mock_services(app: FastAPI) -> None:
    """
    Mount lightweight stubs for bridge services that plugins may call.
    These allow plugins to boot without the real bridge.
    """

    @app.get("/api/stats", tags=["mock_host"])
    async def mock_stats():
        return {
            "status": "mock",
            "bridge": "plugin_runner (standalone)",
            "services": {"bridge": True, "ui": False, "llm": False},
            "uptime": round(time.time() - _boot_time, 1),
        }

    @app.get("/api/config", tags=["mock_host"])
    async def mock_config():
        config_path = _repo_root / "forest.config.json"
        if config_path.exists():
            try:
                return json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"plugins": {"connected": [], "pipeline_order": []}}

    @app.post("/api/config", tags=["mock_host"])
    async def mock_config_save():
        return {"ok": False, "message": "read-only in standalone mode"}

    @app.get("/api/models", tags=["mock_host"])
    async def mock_models():
        return {"models": [], "default": "mock-standalone"}

    LOG.info("  Mock host services mounted: /api/stats, /api/config, /api/models")


# ═══════════════════════════════════════════════════════════════
# 6. PROMOTION CHECKLIST: standalone → host-mounted
# ═══════════════════════════════════════════════════════════════

def run_promotion_check(m: PluginManifest, plugins_dir: Path, repo_root: Path) -> dict:
    """Check if a plugin is ready to be mounted in the real bridge."""
    checks = []
    pid = m.id
    pdir = plugins_dir / pid

    # 1. __init__.py with VERSION
    init_file = pdir / "__init__.py"
    has_version = False
    if init_file.exists():
        content = init_file.read_text(encoding="utf-8")
        has_version = "VERSION" in content
    checks.append({"check": "__init__.py with VERSION", "pass": has_version})

    # 2. Router exists
    checks.append({"check": "Router module found", "pass": bool(m.router_module)})

    # 3. /status endpoint
    checks.append({"check": "/status endpoint", "pass": m.health == "ok"})

    # 4. Pydantic models
    has_models = (pdir / "api" / "models.py").exists()
    checks.append({"check": "api/models.py exists", "pass": has_models})

    # 5. No hardcoded ports
    hardcoded_ports = False
    for pyf in pdir.rglob("*.py"):
        try:
            text = pyf.read_text(encoding="utf-8", errors="ignore")
            if "localhost:5050" in text or "localhost:11435" in text or "localhost:8080" in text:
                hardcoded_ports = True
                break
        except Exception:
            pass
    checks.append({"check": "No hardcoded host ports", "pass": not hardcoded_ports})

    # 6. Bridge mount block exists
    bridge_file = repo_root / "bridges" / "forest_bridge_server.py"
    in_bridge = False
    if bridge_file.exists():
        try:
            text = bridge_file.read_text(encoding="utf-8", errors="ignore")
            in_bridge = pid in text
        except Exception:
            pass
    checks.append({"check": "Mount block in bridge server", "pass": in_bridge})

    # 7. _PLUGIN_META entry
    in_meta = False
    if bridge_file.exists():
        try:
            text = bridge_file.read_text(encoding="utf-8", errors="ignore")
            in_meta = f'"{pid}"' in text and "_PLUGIN_META" in text
        except Exception:
            pass
    checks.append({"check": "Entry in _PLUGIN_META", "pass": in_meta})

    # 8. Config section (optional but nice)
    config_path = repo_root / "forest.config.json"
    in_config = False
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            in_config = pid in cfg or pid in cfg.get("plugins", {}).get("connected", [])
        except Exception:
            pass
    checks.append({"check": "Config section in forest.config.json", "pass": in_config})

    # 9. manifest.json (new contract)
    has_manifest = (pdir / "manifest.json").exists()
    checks.append({"check": "manifest.json present", "pass": has_manifest})

    passed = sum(1 for c in checks if c["pass"])
    total = len(checks)
    ready = passed >= 6  # first 6 are critical

    return {
        "plugin_id": pid,
        "checks": checks,
        "passed": passed,
        "total": total,
        "ready_for_bridge": ready,
        "summary": f"{passed}/{total} checks passed" + (" — READY" if ready else " — NOT READY"),
    }


# ═══════════════════════════════════════════════════════════════
# 7. HOT RELOAD
# ═══════════════════════════════════════════════════════════════

_file_hashes: dict[str, str] = {}


def compute_plugin_hash(plugin_dir: Path) -> str:
    """Hash all .py files in a plugin directory."""
    h = hashlib.md5()
    for pyf in sorted(plugin_dir.rglob("*.py")):
        try:
            h.update(pyf.read_bytes())
        except Exception:
            pass
    return h.hexdigest()


def detect_changes(plugins_dir: Path) -> list[str]:
    """Return list of plugin IDs whose files changed since last check."""
    changed = []
    for entry in sorted(plugins_dir.iterdir()):
        if not entry.is_dir() or entry.name.startswith(("_", ".")):
            continue
        if not (entry / "__init__.py").exists():
            continue
        pid = entry.name
        current = compute_plugin_hash(entry)
        if pid in _file_hashes and _file_hashes[pid] != current:
            changed.append(pid)
        _file_hashes[pid] = current
    return changed


# ═══════════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════════

_port = 9090

app = FastAPI(
    title="Clearbox Plugin Runner",
    version="2.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Static UI ──

@app.get("/", include_in_schema=False)
async def serve_ui():
    return FileResponse(Path(__file__).parent / "plugin_runner_ui.html", media_type="text/html")


# ── Runner API ──

@app.get("/api/runner/plugins", tags=["runner"])
async def api_list_plugins():
    """List all discovered plugins with manifest + health."""
    out = []
    for m in _plugins:
        d = asdict(m)
        d.pop("promotion", None)
        d.pop("endpoints", None)
        out.append(d)
    return {
        "plugins": out,
        "total": len(_plugins),
        "mounted": sum(1 for p in _plugins if p.mounted),
        "uptime": round(time.time() - _boot_time, 1),
    }


@app.get("/api/runner/status", tags=["runner"])
async def api_runner_status():
    """Full runner runtime status."""
    return {
        "ok": True,
        "version": "2.0.0",
        "repo_root": str(_repo_root),
        "scan_dirs": [str(sd) for sd in _scan_dirs],
        "total_plugins": len(_plugins),
        "mounted_plugins": sum(1 for p in _plugins if p.mounted),
        "uptime_sec": round(time.time() - _boot_time, 1),
        "boot_time": _boot_time,
        "features": {
            "manifest_contract": True,
            "runtime_status": True,
            "hot_reload_detect": True,
            "fixture_injection": True,
            "mock_host_services": True,
            "promotion_checklist": True,
        },
    }


@app.get("/api/runner/health", tags=["runner"])
async def api_health_all():
    """Health-check every mounted plugin."""
    results = {}
    for m in _plugins:
        if m.mounted:
            m.health = await check_plugin_health(m)
        results[m.id] = m.health
    return {"health": results}


@app.get("/api/runner/manifest/{plugin_id}", tags=["runner"])
async def api_manifest(plugin_id: str):
    """Get full manifest for a plugin."""
    m = next((p for p in _plugins if p.id == plugin_id), None)
    if not m:
        return JSONResponse({"error": "not found"}, 404)
    return asdict(m)


@app.post("/api/runner/fixtures", tags=["runner"])
async def api_inject_fixtures(plugin_id: str = ""):
    """Inject test fixtures into mounted plugins."""
    results = await inject_fixtures(_port, plugin_id)
    return {"ok": True, "results": results}


@app.get("/api/runner/changes", tags=["runner"])
async def api_detect_changes():
    """Detect which plugins have changed files since last check."""
    changed = []
    for sd in _scan_dirs:
        changed.extend(detect_changes(sd))
    return {
        "changed": changed,
        "message": f"{len(changed)} plugin(s) changed" if changed else "No changes detected",
        "hint": "Restart the runner to pick up changes" if changed else "",
    }


@app.get("/api/runner/promote/{plugin_id}", tags=["runner"])
async def api_promote_check(plugin_id: str):
    """Run promotion checklist for a plugin."""
    m = next((p for p in _plugins if p.id == plugin_id), None)
    if not m:
        return JSONResponse({"error": "not found"}, 404)
    # Find which scan dir contains this plugin
    pcheck_dir = _repo_root / "plugins"
    for sd in _scan_dirs:
        if (sd / m.id).is_dir():
            pcheck_dir = sd
            break
    result = run_promotion_check(m, pcheck_dir, _repo_root)
    m.promotion = result
    return result


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    global _plugins, _repo_root, _port, _scan_dirs

    parser = argparse.ArgumentParser(description="Clearbox Plugin Runner")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--repo", type=str, default=None,
                        help="Path to repo root (auto-detected if run from scripts/)")
    parser.add_argument("--scan", type=str, nargs="+", default=None,
                        help="Extra plugin directories to scan (in addition to {repo}/plugins/)")
    parser.add_argument("--fixtures", action="store_true",
                        help="Inject test fixtures after boot")
    parser.add_argument("--check", type=str, default="",
                        help="Run promotion checklist for a plugin and exit")
    args = parser.parse_args()
    _port = args.port

    # Resolve repo root
    if args.repo:
        _repo_root = Path(args.repo).resolve()
    else:
        # Try: script parent, then look for plugins/ dir
        for candidate in [Path(__file__).resolve().parent,
                          Path(__file__).resolve().parents[1]]:
            if (candidate / "plugins").is_dir():
                _repo_root = candidate
                break
        else:
            _repo_root = Path.cwd()

    plugins_dir = _repo_root / "plugins"

    # Build list of all scan directories
    _scan_dirs = [plugins_dir]
    scan_dirs = _scan_dirs
    if args.scan:
        for sd in args.scan:
            p = Path(sd).resolve()
            if p.is_dir() and p not in scan_dirs:
                scan_dirs.append(p)

    # sys.path setup — add every scan dir + repo root + bridges
    sys.path.insert(0, str(_repo_root))
    sys.path.insert(0, str(_repo_root / "bridges"))
    for sd in scan_dirs:
        sys.path.insert(0, str(sd))
        # Also add the parent so `plugins.x.api.router` works
        if str(sd.parent) not in sys.path:
            sys.path.insert(0, str(sd.parent))

    LOG.info("Repo root: %s", _repo_root)
    for sd in scan_dirs:
        LOG.info("Scan dir:  %s", sd)

    # Discover from all scan directories
    _plugins = []
    for sd in scan_dirs:
        found = discover_plugins(sd, _repo_root)
        # Avoid duplicates by id
        existing_ids = {m.id for m in _plugins}
        for m in found:
            if m.id not in existing_ids:
                _plugins.append(m)
                existing_ids.add(m.id)
    LOG.info("Found %d plugin(s), mounting...\n", len(_plugins))

    # Mount mock host services first (plugins may call them during init)
    mount_mock_services(app)

    for m in _plugins:
        try_mount(app, m)

    # Snapshot file hashes for change detection — check all scan dirs
    for m in _plugins:
        for sd in scan_dirs:
            pdir = sd / m.id
            if pdir.is_dir():
                _file_hashes[m.id] = compute_plugin_hash(pdir)
                break

    mounted = sum(1 for p in _plugins if p.mounted)
    LOG.info("\n  %d/%d plugins mounted.", mounted, len(_plugins))

    # --check mode: run checklist and exit
    if args.check:
        m = next((p for p in _plugins if p.id == args.check), None)
        if not m:
            LOG.error("Plugin '%s' not found.", args.check)
            sys.exit(1)
        # Find the scan dir that actually contains this plugin
        check_dir = plugins_dir
        for sd in scan_dirs:
            if (sd / m.id).is_dir():
                check_dir = sd
                break
        result = run_promotion_check(m, check_dir, _repo_root)
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["ready_for_bridge"] else 1)

    LOG.info("  UI:      http://%s:%d", args.host, args.port)
    LOG.info("  Swagger: http://%s:%d/docs", args.host, args.port)
    LOG.info("  Status:  http://%s:%d/api/runner/status\n", args.host, args.port)

    # Wire up --fixtures to inject test data on startup
    if args.fixtures:
        async def _auto_inject_fixtures():
            LOG.info("--fixtures: injecting test data...")
            await inject_fixtures(_port)
        app.add_event_handler("startup", _auto_inject_fixtures)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
