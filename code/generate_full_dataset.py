#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(os.environ.get("COMFYUI_ROOT", REPO_ROOT / "ComfyUI-general"))
WORKFLOW_DIR = Path(os.environ.get("FILMSTOCK_WORKFLOW_DIR", REPO_ROOT / "workflows"))
OUTPUT_ROOT = Path(os.environ.get("FILMSTOCK_OUTPUT_ROOT", COMFY_ROOT / "output"))
DEFAULT_COMFY_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8288")

OPENAI_WORKFLOW = WORKFLOW_DIR / "chatgpt_image_2_square_api.json"
XAI_WORKFLOW = WORKFLOW_DIR / "xai_grok_imagine_square_api.json"


SCENES = {
    "apartment": "a 25-year-old woman with short dark hair wearing an oversized red leather jacket and black jeans, seated on the floor of a lived-in city apartment, vinyl records, a chrome floor lamp, and a half-open window in the background",
    "backstage": "a 22-year-old man wearing a faded denim jacket and white tank top, sitting on a worn velvet couch in a small backstage green room, an electric guitar, coiled cables, a makeup mirror, and posters without readable text in the background",
    "corner_store": "a 30-year-old woman wearing a black trench coat over a white dress, standing outside a corner store with glass doors, chrome fixtures, parked cars, and layered reflections in the background",
}

FILM_STOCKS = {
    "portra400": "Shot on Kodak Portra 400, 35mm color negative film, natural skin tones, soft pastel color response, gentle highlight rolloff",
    "cinestill800t": "Shot on CineStill 800T, 35mm tungsten-balanced color negative film, cinematic color separation, pronounced red halation around bright highlights when present",
}

LIGHT_SOURCES = {
    "cool_ambient": "Cool diffuse ambient light, soft open shadows, no dominant warm practical light source",
    "warm_practical": "Warm tungsten practical lights with subtle neon spill, visible small bright light sources in the frame, deeper localized shadows",
}

DEVELOPMENT_SCANS = {
    "clean_scan": "Rated at box speed, clean neutral Frontier lab scan, restrained contrast, fine natural grain",
    "pushed_scan": "Pushed one stop in development, stronger contrast, denser shadows, coarser visible grain, slight color shift",
}

MODELS = {
    "chatgpt_image_2": {
        "workflow": OPENAI_WORKFLOW,
        "provider": "openai",
        "api_model": "gpt-image-2",
    },
    "xai_grok_imagine": {
        "workflow": XAI_WORKFLOW,
        "provider": "xai",
        "api_model": "grok-imagine-image-pro",
    },
}


PROMPT_SHELL = (
    "Square photorealistic editorial 35mm photograph of {scene}. "
    "Natural human proportions, realistic skin texture, art-directed but not glossy. "
    "Camera at eye level, 50mm lens, medium shot, subject clearly visible. "
    "{film_stock}. {light_source}. {development_scan}. "
    "No text, no logos, no watermark, no artificial digital smoothness."
)


@dataclass(frozen=True)
class Variant:
    model_key: str
    scene_key: str
    film_key: str
    light_key: str
    scan_key: str
    replicate: int

    @property
    def variant_id(self) -> str:
        return (
            f"{self.model_key}_{self.scene_key}_{self.film_key}_"
            f"{self.light_key}_{self.scan_key}_rep{self.replicate:02d}"
        )

    @property
    def condition_id(self) -> str:
        return f"{self.scene_key}_{self.film_key}_{self.light_key}_{self.scan_key}"


def request_json(url: str, *, method: str = "GET", payload: Any = None, timeout: float = 30.0) -> Any:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    return json.loads(raw.decode("utf-8")) if raw else {}


def output_image_paths(history_item: dict[str, Any]) -> list[str]:
    roots = {
        "output": COMFY_ROOT / "output",
        "temp": COMFY_ROOT / "temp",
        "input": COMFY_ROOT / "input",
    }
    paths: list[str] = []
    for node_output in history_item.get("outputs", {}).values():
        for image in node_output.get("images", []) or []:
            image_type = image.get("type", "output")
            base = roots.get(image_type, roots["output"])
            subfolder = image.get("subfolder") or ""
            filename = image.get("filename")
            if filename:
                paths.append(str(base / subfolder / filename))
    return paths


def wait_for_history(comfy_url: str, prompt_id: str, timeout: float) -> tuple[dict[str, Any], list[str]]:
    deadline = time.time() + timeout
    last_status = None
    while time.time() < deadline:
        history = request_json(f"{comfy_url}/history/{prompt_id}", timeout=30.0)
        item = history.get(prompt_id)
        if item:
            status = item.get("status", {})
            status_str = status.get("status_str")
            if status_str != last_status:
                print(f"    status: {status_str}", flush=True)
                last_status = status_str
            if status_str == "error":
                raise RuntimeError(json.dumps(status, indent=2)[:5000])
            paths = output_image_paths(item)
            if paths:
                return item, paths
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for ComfyUI prompt {prompt_id}")


def build_prompt(variant: Variant) -> str:
    return PROMPT_SHELL.format(
        scene=SCENES[variant.scene_key],
        film_stock=FILM_STOCKS[variant.film_key],
        light_source=LIGHT_SOURCES[variant.light_key],
        development_scan=DEVELOPMENT_SCANS[variant.scan_key],
    )


def iter_variants() -> list[Variant]:
    variants: list[Variant] = []
    for model_key in MODELS:
        for scene_key in SCENES:
            for film_key in FILM_STOCKS:
                for light_key in LIGHT_SOURCES:
                    for scan_key in DEVELOPMENT_SCANS:
                        for replicate in (1, 2):
                            variants.append(
                                Variant(
                                    model_key=model_key,
                                    scene_key=scene_key,
                                    film_key=film_key,
                                    light_key=light_key,
                                    scan_key=scan_key,
                                    replicate=replicate,
                                )
                            )
    return variants


def load_workflow(variant: Variant, run_id: str, dataset_rel: str) -> dict[str, Any]:
    model_cfg = MODELS[variant.model_key]
    workflow = json.loads(Path(model_cfg["workflow"]).read_text(encoding="utf-8"))
    prompt = build_prompt(variant)
    run_token = f"{run_id}_{variant.variant_id}"

    normalized_prefix = (
        f"{dataset_rel}/normalized/{variant.model_key}/{variant.scene_key}/"
        f"{variant.film_key}/{variant.light_key}/{variant.scan_key}/"
        f"{variant.variant_id}"
    )
    source_prefix = f"{dataset_rel}/source/{variant.model_key}/{variant.scene_key}"

    workflow["10"]["inputs"]["prompt"] = prompt
    workflow["10"]["inputs"]["run_token"] = run_token
    workflow["10"]["inputs"]["output_prefix"] = source_prefix
    workflow["30"]["inputs"]["filename_prefix"] = normalized_prefix
    return workflow


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    fieldnames = [
        "status",
        "model_key",
        "provider",
        "api_model",
        "scene_key",
        "film_key",
        "light_key",
        "scan_key",
        "replicate",
        "variant_id",
        "condition_id",
        "prompt_id",
        "image_path",
        "width",
        "height",
        "mode",
        "prompt",
        "error",
        "started_at",
        "completed_at",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def serializable_models() -> dict[str, dict[str, str]]:
    clean: dict[str, dict[str, str]] = {}
    for key, value in MODELS.items():
        clean[key] = {
            "workflow": str(value["workflow"]),
            "provider": str(value["provider"]),
            "api_model": str(value["api_model"]),
        }
    return clean


def existing_successes(jsonl_path: Path) -> set[str]:
    done: set[str] = set()
    if not jsonl_path.exists():
        return done
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("status") == "success":
            done.add(str(record.get("variant_id")))
    return done


def make_base_record(variant: Variant, prompt: str) -> dict[str, Any]:
    model_cfg = MODELS[variant.model_key]
    return {
        "model_key": variant.model_key,
        "provider": model_cfg["provider"],
        "api_model": model_cfg["api_model"],
        "scene_key": variant.scene_key,
        "film_key": variant.film_key,
        "light_key": variant.light_key,
        "scan_key": variant.scan_key,
        "replicate": variant.replicate,
        "variant_id": variant.variant_id,
        "condition_id": variant.condition_id,
        "prompt": prompt,
    }


def run_dataset(args: argparse.Namespace) -> None:
    system = request_json(f"{args.comfy_url}/system_stats", timeout=5.0).get("system", {})
    run_id = args.run_id or time.strftime("filmstock_%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    dataset_rel = f"filmstock_benchmark/{run_id}"
    dataset_dir = OUTPUT_ROOT / dataset_rel
    manifest_jsonl = dataset_dir / "manifest.jsonl"
    manifest_csv = dataset_dir / "manifest.csv"
    design_json = dataset_dir / "design.json"
    records: list[dict[str, Any]] = []

    dataset_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        design_json,
        {
            "run_id": run_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "comfy_url": args.comfy_url,
            "comfyui_version": system.get("comfyui_version"),
            "image_size": "1024x1024",
            "models": serializable_models(),
            "scenes": SCENES,
            "film_stocks": FILM_STOCKS,
            "light_sources": LIGHT_SOURCES,
            "development_scans": DEVELOPMENT_SCANS,
            "prompt_shell": PROMPT_SHELL,
        },
    )

    successes = existing_successes(manifest_jsonl) if args.resume else set()
    variants = iter_variants()
    if args.limit:
        variants = variants[: args.limit]

    print(f"run_id: {run_id}")
    print(f"dataset_dir: {dataset_dir}")
    print(f"planned_variants: {len(variants)}")
    print(f"already_successful: {len(successes)}")

    for idx, variant in enumerate(variants, start=1):
        prompt = build_prompt(variant)
        base_record = make_base_record(variant, prompt)
        if variant.variant_id in successes:
            print(f"[{idx:03d}/{len(variants):03d}] SKIP {variant.variant_id}", flush=True)
            continue

        started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        print(f"[{idx:03d}/{len(variants):03d}] RUN {variant.variant_id}", flush=True)
        record = {**base_record, "status": "started", "started_at": started_at, "completed_at": ""}
        append_jsonl(manifest_jsonl, record)

        try:
            workflow = load_workflow(variant, run_id, dataset_rel)
            submitted = request_json(
                f"{args.comfy_url}/prompt",
                method="POST",
                payload={"prompt": workflow, "client_id": f"filmstock-full-{variant.variant_id}-{uuid.uuid4().hex}"},
                timeout=30.0,
            )
            prompt_id = submitted["prompt_id"]
            print(f"    prompt_id: {prompt_id}", flush=True)
            _, paths = wait_for_history(args.comfy_url, prompt_id, args.timeout)
            image_path = paths[-1]
            with Image.open(image_path) as image:
                width, height = image.size
                mode = image.mode
            completed = {
                **base_record,
                "status": "success",
                "prompt_id": prompt_id,
                "image_path": image_path,
                "width": width,
                "height": height,
                "mode": mode,
                "error": "",
                "started_at": started_at,
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            append_jsonl(manifest_jsonl, completed)
            records.append(completed)
            write_csv(manifest_csv, records)
            print(f"    saved: {image_path} size={width}x{height}", flush=True)
        except Exception as exc:
            failed = {
                **base_record,
                "status": "error",
                "prompt_id": "",
                "image_path": "",
                "width": "",
                "height": "",
                "mode": "",
                "error": str(exc),
                "started_at": started_at,
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            append_jsonl(manifest_jsonl, failed)
            records.append(failed)
            write_csv(manifest_csv, records)
            print(f"    ERROR: {exc}", flush=True)
            if not args.continue_on_error:
                raise

    all_records = []
    for line in manifest_jsonl.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("status") in {"success", "error"}:
            all_records.append(item)
    write_csv(manifest_csv, all_records)
    write_json(
        dataset_dir / "summary.json",
        {
            "run_id": run_id,
            "dataset_dir": str(dataset_dir),
            "manifest_csv": str(manifest_csv),
            "manifest_jsonl": str(manifest_jsonl),
            "total_success": sum(1 for r in all_records if r.get("status") == "success"),
            "total_error": sum(1 for r in all_records if r.get("status") == "error"),
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )
    print(f"manifest_csv: {manifest_csv}")
    print(f"summary: {dataset_dir / 'summary.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the 96-image filmstock benchmark dataset through ComfyUI.")
    parser.add_argument("--comfy-url", default=DEFAULT_COMFY_URL)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_dataset(parse_args())
