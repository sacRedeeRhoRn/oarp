from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from oarp.models import BootstrapResult, PreflightReport, RunConfig
from oarp.runtime import ensure_run_layout, now_iso

_REQUIRED_MODULES = [
    "yaml",
    "requests",
    "pandas",
    "pyarrow",
    "pypdf",
    "pdfplumber",
    "trafilatura",
    "torch",
    "torch_geometric",
]


def _repo_root() -> Path:
    return Path(__file__).expanduser().resolve().parents[2]


def _container_name(cfg: RunConfig) -> str:
    token = cfg.as_path().name.replace("_", "-").replace(".", "-")
    return f"oarp-tgi-{token}"[:80]


def _host_arch() -> str:
    return str(platform.machine() or "").strip().lower()


def _normalize_http_path(value: str, default: str) -> str:
    clean = str(value or "").strip() or str(default or "/")
    if not clean.startswith("/"):
        clean = f"/{clean}"
    return clean


def _resolve_tgi_endpoints(cfg: RunConfig) -> dict[str, str]:
    health_path = _normalize_http_path(cfg.tgi_health_path, "/health")
    generate_path = _normalize_http_path(cfg.tgi_generate_path, "/generate")
    raw = str(cfg.tgi_endpoint or "").strip()

    if not raw:
        base = f"http://127.0.0.1:{int(cfg.tgi_port)}"
        generate = base.rstrip("/") + generate_path
        return {"base": base, "health_url": base.rstrip("/") + health_path, "generate_url": generate}

    parsed = urlparse(raw)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        base = f"{parsed.scheme}://{parsed.netloc}"
        if parsed.path and parsed.path not in {"", "/"}:
            generate = raw
        else:
            generate = base.rstrip("/") + generate_path
        return {"base": base, "health_url": base.rstrip("/") + health_path, "generate_url": generate}

    base = raw.rstrip("/")
    generate = base + generate_path
    return {"base": base, "health_url": base + health_path, "generate_url": generate}


def _run_cmd(cmd: list[str], cwd: Path | None = None, timeout_sec: int = 900) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=max(10, int(timeout_sec)),
    )
    return int(proc.returncode), str(proc.stdout or ""), str(proc.stderr or "")


def _python311_path() -> str:
    if sys.version_info >= (3, 11):
        return sys.executable
    candidate = shutil.which("python3.11")
    if candidate:
        return candidate
    return ""


def _docker_manifest_supports(image: str, platform_token: str) -> bool:
    os_name, arch = (platform_token.split("/", 1) + [""])[:2]
    if not os_name or not arch:
        return False
    rc, out, _err = _run_cmd(["docker", "manifest", "inspect", str(image)], timeout_sec=180)
    if rc != 0 or not out.strip():
        return False
    try:
        payload = json.loads(out)
    except Exception:
        payload = None
    if isinstance(payload, dict):
        manifests = payload.get("manifests")
        if isinstance(manifests, list):
            for item in manifests:
                if not isinstance(item, dict):
                    continue
                platform_node = item.get("platform")
                if not isinstance(platform_node, dict):
                    continue
                if (
                    str(platform_node.get("os") or "").strip().lower() == os_name
                    and str(platform_node.get("architecture") or "").strip().lower() == arch
                ):
                    return True
        if (
            str(payload.get("os") or "").strip().lower() == os_name
            and str(payload.get("architecture") or "").strip().lower() == arch
        ):
            return True

    out_l = out.lower()
    return f'"os":"{os_name}"' in out_l and f'"architecture":"{arch}"' in out_l


def _resolve_tgi_platform(cfg: RunConfig) -> tuple[str, bool, str]:
    mode = str(cfg.tgi_mode or "docker").strip().lower()
    requested = str(cfg.tgi_platform or "auto").strip().lower()
    host_arch = _host_arch()
    if mode != "docker":
        return "", False, "external_mode"

    if requested in {"linux/amd64", "linux/arm64"}:
        emulated = requested == "linux/amd64" and host_arch in {"arm64", "aarch64"}
        return requested, emulated, "explicit"

    if requested != "auto":
        return "linux/amd64", host_arch in {"arm64", "aarch64"}, "invalid_requested_fallback"

    if host_arch in {"arm64", "aarch64"}:
        arm_supported = _docker_manifest_supports(str(cfg.tgi_docker_image), "linux/arm64")
        if arm_supported:
            return "linux/arm64", False, "arm64_manifest_available"
        return "linux/amd64", True, "arm64_manifest_missing_fallback_amd64"

    return "linux/amd64", False, "host_amd64"


def _target_python(cfg: RunConfig) -> str:
    explicit = str(cfg.python_exec or "").strip()
    if explicit:
        return explicit
    candidate = _python311_path()
    if candidate:
        return candidate
    return sys.executable


def _port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(0.2)
        return sock.connect_ex((host, int(port))) == 0
    except Exception:
        return False
    finally:
        sock.close()


def _parse_port_range(raw: str, fallback: int) -> list[int]:
    text = str(raw or "").strip()
    if "-" in text:
        left, right = text.split("-", 1)
        try:
            start = int(left.strip())
            end = int(right.strip())
            if start > end:
                start, end = end, start
            return [port for port in range(max(1, start), min(65535, end) + 1)]
        except Exception:
            pass
    if "," in text:
        out: list[int] = []
        for token in text.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                value = int(token)
            except Exception:
                continue
            if 1 <= value <= 65535:
                out.append(value)
        if out:
            return out
    return [int(fallback)]


def _select_tgi_port(cfg: RunConfig) -> tuple[int, str]:
    desired = int(cfg.tgi_port)
    policy = str(getattr(cfg, "tgi_port_policy", "reuse_or_allocate") or "reuse_or_allocate").strip().lower()
    if not _port_in_use(desired):
        return desired, "desired_free"
    if policy != "reuse_or_allocate":
        return desired, "desired_in_use"
    candidates = _parse_port_range(str(getattr(cfg, "tgi_port_range", "8080-8090") or "8080-8090"), desired)
    for port in candidates:
        if not _port_in_use(port):
            return int(port), "allocated_from_range"
    return desired, "range_exhausted"


def _python_version_from_exec(python_exec: str) -> tuple[bool, str]:
    script = "import sys; print(sys.version.split()[0])"
    rc, out, err = _run_cmd([python_exec, "-c", script], timeout_sec=30)
    version = str(out or "").strip()
    if rc != 0 or not version:
        detail = str(err or out or "unable_to_probe_python").strip()
        return False, detail
    parts = version.split(".")
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
    except Exception:
        return False, f"invalid_version:{version}"
    ok = (major, minor) >= (3, 11)
    return ok, version


def _module_probe(python_exec: str, modules: list[str]) -> tuple[bool, list[str], str]:
    script = (
        "import importlib, json\n"
        f"mods={json.dumps(modules, sort_keys=True)}\n"
        "missing=[]\n"
        "for m in mods:\n"
        "  try:\n"
        "    importlib.import_module(m)\n"
        "  except Exception:\n"
        "    missing.append(m)\n"
        "print(json.dumps({'missing': missing}, sort_keys=True))\n"
    )
    rc, out, err = _run_cmd([python_exec, "-c", script], timeout_sec=120)
    if rc != 0:
        return False, modules, str(err or out or "").strip()
    try:
        payload = json.loads(str(out or "").strip() or "{}")
    except Exception:
        payload = {}
    missing = payload.get("missing") if isinstance(payload, dict) else []
    if not isinstance(missing, list):
        missing = modules
    missing_clean = [str(item).strip() for item in missing if str(item).strip()]
    return len(missing_clean) == 0, missing_clean, ""


def _local_repo_file_count(path: Path, recursive: bool, pattern: str) -> int:
    try:
        iterator = path.rglob(pattern) if recursive else path.glob(pattern)
        return int(sum(1 for item in iterator if item.is_file()))
    except Exception:
        return 0


def _cpu_env_probe_payload(python_exec: str) -> dict[str, Any]:
    script = (
        "import json, os, platform, sys\n"
        "keys=['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_MAX_THREADS','CUDA_VISIBLE_DEVICES']\n"
        "payload={\n"
        "  'python_executable': sys.executable,\n"
        "  'python_version': sys.version.split()[0],\n"
        "  'platform': platform.platform(),\n"
        "  'machine': platform.machine(),\n"
        "  'thread_env': {k: os.environ.get(k,'') for k in keys},\n"
        "}\n"
        "print(json.dumps(payload, sort_keys=True))\n"
    )
    rc, out, err = _run_cmd([python_exec, "-c", script], timeout_sec=30)
    if rc != 0:
        return {
            "python_executable": python_exec,
            "probe_error": str(err or out or "").strip(),
        }
    try:
        payload = json.loads(str(out or "").strip() or "{}")
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    payload.setdefault("python_executable", python_exec)
    return payload


def tgi_status(cfg: RunConfig, *, check_generate: bool = False) -> dict[str, Any]:
    endpoints = _resolve_tgi_endpoints(cfg)
    selected_platform, emulation_used, platform_reason = _resolve_tgi_platform(cfg)
    payload: dict[str, Any] = {
        "tgi_mode": str(cfg.tgi_mode or "docker").strip().lower(),
        "docker_available": bool(shutil.which("docker")),
        "endpoint": endpoints["base"],
        "health_url": endpoints["health_url"],
        "generate_url": endpoints["generate_url"],
        "container_name": _container_name(cfg),
        "container_running": False,
        "healthy": False,
        "http_status": 0,
        "generate_ok": False,
        "generate_http_status": 0,
        "selected_tgi_platform": selected_platform,
        "tgi_emulation_used": emulation_used,
        "tgi_platform_reason": platform_reason,
    }

    if payload["tgi_mode"] == "docker" and payload["docker_available"]:
        rc, out, _err = _run_cmd(["docker", "ps", "--format", "{{.Names}}"])
        if rc == 0:
            names = [line.strip() for line in out.splitlines() if line.strip()]
            payload["container_running"] = _container_name(cfg) in names

    try:
        resp = requests.get(endpoints["health_url"], timeout=5)
        payload["http_status"] = int(resp.status_code)
        payload["healthy"] = 200 <= int(resp.status_code) < 300
    except Exception as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"

    if check_generate:
        try:
            response = requests.post(
                endpoints["generate_url"],
                json={
                    "inputs": "ping",
                    "parameters": {"max_new_tokens": 1, "temperature": 0.1, "return_full_text": False},
                },
                timeout=8,
            )
            payload["generate_http_status"] = int(response.status_code)
            payload["generate_ok"] = 200 <= int(response.status_code) < 300
        except Exception as exc:
            payload["generate_error"] = f"{type(exc).__name__}: {exc}"
    return payload


def tgi_start(cfg: RunConfig) -> dict[str, Any]:
    mode = str(cfg.tgi_mode or "docker").strip().lower()
    name = _container_name(cfg)
    status = tgi_status(cfg, check_generate=(mode == "external" or bool(getattr(cfg, "tgi_reuse_existing", True))))
    if mode == "external":
        ok = bool(status.get("healthy")) and bool(status.get("generate_ok"))
        return {
            "ok": ok,
            "action": "external_verified" if ok else "external_unhealthy",
            **status,
        }

    if status.get("healthy") and status.get("generate_ok") and (
        status.get("container_running") or bool(getattr(cfg, "tgi_reuse_existing", True))
    ):
        return {
            "ok": True,
            "action": "already_healthy" if status.get("container_running") else "reuse_existing_healthy_endpoint",
            **status,
        }
    if not shutil.which("docker"):
        return {
            "ok": False,
            "action": "docker_missing",
            **status,
        }

    selected_port, port_reason = _select_tgi_port(cfg)
    if selected_port != int(cfg.tgi_port):
        cfg.tgi_port = int(selected_port)
    if _port_in_use(int(cfg.tgi_port)):
        return {
            "ok": False,
            "action": "port_in_use",
            "port_reason": port_reason,
            **tgi_status(cfg),
        }

    _run_cmd(["docker", "rm", "-f", name], timeout_sec=120)

    selected_platform, emulation_used, platform_reason = _resolve_tgi_platform(cfg)
    cache_dir = cfg.as_path() / "artifacts" / "tgi_data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        name,
        "-p",
        f"{int(cfg.tgi_port)}:80",
        "-v",
        f"{cache_dir}:/data",
        "--platform",
        selected_platform,
        str(cfg.tgi_docker_image),
        "--model-id",
        str(cfg.tgi_model_id),
        "--max-concurrent-requests",
        "32",
        "--max-input-tokens",
        "768",
        "--disable-custom-kernels",
    ]
    rc, out, err = _run_cmd(cmd, timeout_sec=1800)
    if rc != 0:
        return {
            "ok": False,
            "action": "docker_run_failed",
            "stdout": out,
            "stderr": err,
            "selected_tgi_platform": selected_platform,
            "tgi_emulation_used": emulation_used,
            "tgi_platform_reason": platform_reason,
            "port_reason": port_reason,
            **tgi_status(cfg),
        }

    deadline = time.time() + 900
    last = tgi_status(cfg, check_generate=True)
    while time.time() < deadline:
        last = tgi_status(cfg, check_generate=True)
        if last.get("healthy") and last.get("generate_ok"):
            break
        if not last.get("container_running"):
            break
        time.sleep(2)

    return {
        "ok": bool(last.get("healthy")) and bool(last.get("generate_ok")),
        "action": "started" if (last.get("healthy") and last.get("generate_ok")) else "start_timeout",
        "selected_tgi_platform": selected_platform,
        "tgi_emulation_used": emulation_used,
        "tgi_platform_reason": platform_reason,
        "port_reason": port_reason,
        **last,
    }


def tgi_stop(cfg: RunConfig) -> dict[str, Any]:
    mode = str(cfg.tgi_mode or "docker").strip().lower()
    if mode != "docker":
        return {
            "ok": True,
            "action": "external_noop",
            **tgi_status(cfg),
        }
    name = _container_name(cfg)
    if not shutil.which("docker"):
        return {
            "ok": False,
            "action": "docker_missing",
            **tgi_status(cfg),
        }
    rc, out, err = _run_cmd(["docker", "rm", "-f", name], timeout_sec=120)
    return {
        "ok": bool(rc == 0),
        "action": "stopped" if rc == 0 else "not_stopped",
        "stdout": out,
        "stderr": err,
        **tgi_status(cfg),
    }


def bootstrap_runtime(cfg: RunConfig) -> BootstrapResult:
    layout = ensure_run_layout(cfg.as_path())
    artifacts = layout["artifacts"]
    steps: list[dict[str, Any]] = []

    python_exec = _target_python(cfg)
    venv_path = cfg.as_path() / ".venv"

    def _add_step(name: str, ok: bool, detail: str, stdout: str = "", stderr: str = "") -> None:
        steps.append(
            {
                "name": name,
                "ok": bool(ok),
                "detail": detail,
                "stdout": stdout,
                "stderr": stderr,
                "created_at": now_iso(),
            }
        )

    if not python_exec:
        _add_step(
            "python311",
            False,
            "python3.11 not found on PATH; cannot create strict runtime venv",
        )
        tgi_info = tgi_status(cfg, check_generate=True)
        payload = {
            "ok": False,
            "python_executable": "",
            "venv_path": str(venv_path),
            "steps": steps,
            "tgi_status": tgi_info,
            "target_python_executable": "",
            "selected_tgi_platform": str(tgi_info.get("selected_tgi_platform") or ""),
            "tgi_emulation_used": bool(tgi_info.get("tgi_emulation_used")),
            "reexec_required": False,
        }
        path = artifacts / "bootstrap_report.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return BootstrapResult(
            ok=False,
            python_executable="",
            venv_path=venv_path,
            tgi_status=tgi_info,
            steps=steps,
            report_path=path,
            target_python_executable="",
            selected_tgi_platform=str(tgi_info.get("selected_tgi_platform") or ""),
            tgi_emulation_used=bool(tgi_info.get("tgi_emulation_used")),
            reexec_required=False,
        )

    if not venv_path.exists():
        rc, out, err = _run_cmd([python_exec, "-m", "venv", str(venv_path)], timeout_sec=300)
        _add_step("create_venv", rc == 0, "created" if rc == 0 else "failed", out, err)
    else:
        _add_step("create_venv", True, "already_exists")

    venv_python = venv_path / "bin" / "python"
    if venv_python.exists():
        rc, out, err = _run_cmd([str(venv_python), "-m", "pip", "install", "-U", "pip"], timeout_sec=600)
        _add_step("pip_upgrade", rc == 0, "ok" if rc == 0 else "failed", out, err)

        rc, out, err = _run_cmd(
            [str(venv_python), "-m", "pip", "install", "-e", ".[ml,pdf,html]"],
            cwd=_repo_root(),
            timeout_sec=1800,
        )
        _add_step("install_oarp_strict", rc == 0, "ok" if rc == 0 else "failed", out, err)

        ok_mods, missing, mod_err = _module_probe(str(venv_python), _REQUIRED_MODULES)
        _add_step(
            "import_check",
            bool(ok_mods),
            "ok" if ok_mods else f"missing:{','.join(missing)}",
            "",
            mod_err,
        )

        rc, out, err = _run_cmd([str(venv_python), "-m", "pip", "freeze"], timeout_sec=120)
        lock_path = artifacts / "dependency_lock_snapshot.txt"
        if rc == 0:
            lock_path.write_text(str(out or ""), encoding="utf-8")
        _add_step("dependency_snapshot", rc == 0, "ok" if rc == 0 else "failed", "", err)
    else:
        _add_step("venv_python", False, f"missing interpreter at {venv_python}")
        lock_path = artifacts / "dependency_lock_snapshot.txt"

    cpu_probe_path = artifacts / "cpu_env_probe.json"
    cpu_payload = _cpu_env_probe_payload(str(venv_python if venv_python.exists() else python_exec))
    cpu_probe_path.write_text(json.dumps(cpu_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _add_step("cpu_env_probe", True, str(cpu_probe_path))

    if not str(cfg.tgi_endpoint or "").strip():
        cfg.tgi_endpoint = _resolve_tgi_endpoints(cfg)["generate_url"]
    cfg.python_exec = str(venv_python if venv_python.exists() else python_exec)

    tgi_info = tgi_start(cfg)
    _add_step(
        "tgi_start",
        bool(tgi_info.get("ok")),
        str(tgi_info.get("action") or ""),
        json.dumps(tgi_info, sort_keys=True),
        "",
    )

    ok = all(bool(item.get("ok")) for item in steps)
    reexec_required = bool(venv_python.exists()) and Path(sys.executable).resolve() != venv_python.resolve()
    payload = {
        "ok": ok,
        "python_executable": str(venv_python if venv_python.exists() else python_exec),
        "target_python_executable": str(venv_python if venv_python.exists() else python_exec),
        "venv_path": str(venv_path),
        "steps": steps,
        "tgi_status": tgi_info,
        "selected_tgi_platform": str(tgi_info.get("selected_tgi_platform") or ""),
        "tgi_emulation_used": bool(tgi_info.get("tgi_emulation_used")),
        "reexec_required": reexec_required,
        "dependency_lock_snapshot": str(lock_path),
        "cpu_env_probe_path": str(cpu_probe_path),
    }
    path = artifacts / "bootstrap_report.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return BootstrapResult(
        ok=ok,
        python_executable=str(venv_python if venv_python.exists() else python_exec),
        venv_path=venv_path,
        tgi_status=tgi_info,
        steps=steps,
        report_path=path,
        target_python_executable=str(venv_python if venv_python.exists() else python_exec),
        selected_tgi_platform=str(tgi_info.get("selected_tgi_platform") or ""),
        tgi_emulation_used=bool(tgi_info.get("tgi_emulation_used")),
        reexec_required=reexec_required,
    )


def preflight_strict(
    cfg: RunConfig,
    python_exec: str | None = None,
    *,
    check_tgi_generate: bool = True,
) -> PreflightReport:
    layout = ensure_run_layout(cfg.as_path())
    artifacts = layout["artifacts"]

    checks: list[dict[str, Any]] = []

    def _check(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail, "created_at": now_iso()})

    target_python = str(python_exec or cfg.python_exec or _target_python(cfg)).strip()
    version_ok, version_detail = _python_version_from_exec(target_python)
    _check("python>=3.11", version_ok, f"target={target_python} version={version_detail}")

    for path_str in cfg.local_repo_paths:
        path = Path(path_str).expanduser().resolve()
        _check(f"local_repo_exists:{path}", path.exists(), "exists" if path.exists() else "missing")
        _check(
            f"local_repo_readable:{path}",
            os.access(path, os.R_OK),
            "readable" if os.access(path, os.R_OK) else "not_readable",
        )
        if path.exists() and path.is_dir():
            count = _local_repo_file_count(path, bool(cfg.local_repo_recursive), str(cfg.local_file_glob or "*.pdf"))
            _check(f"local_repo_file_count:{path}", count > 0, f"count={count}")

    modules_ok, missing, module_err = _module_probe(target_python, _REQUIRED_MODULES)
    if module_err:
        _check("module_probe", False, module_err)
    for module_name in _REQUIRED_MODULES:
        found = module_name not in missing if modules_ok or not module_err else False
        _check(f"module:{module_name}", found, "ok" if found else "not_installed")

    mode = str(cfg.tgi_mode or "docker").strip().lower()
    if mode == "docker":
        docker_ok = bool(shutil.which("docker"))
        _check("docker_available", docker_ok, "ok" if docker_ok else "missing")
    else:
        _check("docker_available", True, "skipped_external_mode")

    tgi_info = tgi_status(cfg, check_generate=bool(check_tgi_generate))
    _check("tgi_healthy", bool(tgi_info.get("healthy")), json.dumps(tgi_info, sort_keys=True))
    if check_tgi_generate:
        _check("tgi_generate_ok", bool(tgi_info.get("generate_ok")), json.dumps(tgi_info, sort_keys=True))

    ok = all(bool(item["ok"]) for item in checks)
    payload = {
        "ok": ok,
        "target_python_executable": target_python,
        "checks": checks,
    }
    report_path = artifacts / "preflight_report.json"
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return PreflightReport(
        ok=ok,
        checks=checks,
        report_path=report_path,
        target_python_executable=target_python,
    )
