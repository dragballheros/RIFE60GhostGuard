from __future__ import annotations

import hmac
import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.responses import FileResponse

APP_ROOT = Path(os.environ.get("TRAINER_JOB_ROOT", "/workspace/rife60-anima-jobs"))
MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "/models"))
SD_SCRIPTS_ROOT = Path(os.environ.get("SD_SCRIPTS_ROOT", "/opt/sd-scripts"))
ANIMA_DIT = Path(os.environ.get("ANIMA_DIT_PATH", str(MODEL_ROOT / "anima-base-v1.0.safetensors")))
QWEN3 = Path(os.environ.get("QWEN3_PATH", str(MODEL_ROOT / "qwen_3_06b_base.safetensors")))
ANIMA_VAE = Path(os.environ.get("ANIMA_VAE_PATH", str(MODEL_ROOT / "qwen_image_vae.safetensors")))
LORA_INSTALL_DIR = os.environ.get("LORA_INSTALL_DIR", "").strip()
TRAINER_API_TOKEN = os.environ.get("TRAINER_API_TOKEN", "")
MAX_IMAGES = int(os.environ.get("MAX_TRAINING_IMAGES", "40"))
MAX_DATASET_BYTES = int(os.environ.get("MAX_TRAINING_BYTES", str(180 * 1024 * 1024)))
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
AUTH_SCHEME = HTTPBearer(auto_error=False)

app = FastAPI(title="RIFE60 ANIMA LoRA Trainer", version="1.0.0")
state_lock = threading.Lock()
job_threads: dict[str, threading.Thread] = {}


def require_token(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(AUTH_SCHEME)],
) -> None:
    if not TRAINER_API_TOKEN:
        raise HTTPException(status_code=503, detail="TRAINER_API_TOKEN is not configured on this service.")
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Bearer token required.")
    if not hmac.compare_digest(credentials.credentials, TRAINER_API_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid trainer token.")


def job_dir(job_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise HTTPException(status_code=404, detail="Unknown training job.")
    return APP_ROOT / job_id


def read_state(job_id: str) -> dict[str, Any]:
    path = job_dir(job_id) / "status.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Unknown training job.")
    try:
        return json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        raise HTTPException(status_code=500, detail="Could not read job state.")


def write_state(job_id: str, state: dict[str, Any]) -> None:
    folder = job_dir(job_id)
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / "status.tmp"
    temporary.write_text(json.dumps(state, ensure_ascii=False), "utf-8")
    temporary.replace(folder / "status.json")


def ensure_model_files() -> None:
    missing = [str(path) for path in (ANIMA_DIT, QWEN3, ANIMA_VAE) if not path.is_file()]
    script = SD_SCRIPTS_ROOT / "anima_train_network.py"
    if not script.is_file():
        missing.append(str(script))
    if missing:
        raise RuntimeError("Missing ANIMA training files: " + ", ".join(missing))


def toml_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ") + '"'


def create_dataset_config(dataset_dir: Path, config_path: Path) -> None:
    text = (
        "[[datasets]]\n"
        "resolution = 1024\n"
        "batch_size = 1\n"
        "enable_bucket = true\n"
        "min_bucket_reso = 256\n"
        "max_bucket_reso = 2048\n"
        "bucket_reso_steps = 16\n\n"
        "[[datasets.subsets]]\n"
        f"image_dir = {toml_quote(str(dataset_dir))}\n"
        "num_repeats = 10\n"
        'caption_extension = ".txt"\n'
    )
    config_path.write_text(text, "utf-8")


def run_training(job_id: str, options: dict[str, Any]) -> None:
    folder = job_dir(job_id)
    dataset_dir = folder / "dataset"
    output_dir = folder / "output"
    log_path = folder / "training.log"
    try:
        ensure_model_files()
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = folder / "dataset.toml"
        create_dataset_config(dataset_dir, config_path)

        command = [
            "accelerate", "launch", "--num_cpu_threads_per_process", "1",
            str(SD_SCRIPTS_ROOT / "anima_train_network.py"),
            f"--pretrained_model_name_or_path={ANIMA_DIT}",
            f"--qwen3={QWEN3}",
            f"--vae={ANIMA_VAE}",
            f"--dataset_config={config_path}",
            f"--output_dir={output_dir}",
            f"--output_name=rife60_anima_{job_id[:8]}",
            "--save_model_as=safetensors",
            "--network_module=networks.lora_anima",
            f"--network_dim={options['rank']}",
            f"--network_alpha={options['rank']}",
            f"--learning_rate={options['learning_rate']}",
            "--optimizer_type=AdamW8bit",
            "--lr_scheduler=constant",
            "--timestep_sampling=sigmoid",
            "--discrete_flow_shift=1.0",
            f"--max_train_epochs={options['epochs']}",
            "--save_every_n_epochs=1",
            "--mixed_precision=bf16",
            "--gradient_checkpointing",
            "--cache_latents",
            "--cache_text_encoder_outputs",
            "--network_train_unet_only",
            "--vae_chunk_size=64",
            "--vae_disable_cache",
            "--max_data_loader_n_workers=2",
            "--logging_dir=" + str(folder / "logs"),
        ]

        state = read_state(job_id)
        state.update(status="running", progress=0.0, message="Training process starting.", started_at=time.time())
        write_state(job_id, state)
        epoch_pattern = re.compile(r"(?:epoch\s*[:= ]\s*|\bEpoch\s+)(\d+)", re.IGNORECASE)
        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            log.write("RIFE60 ANIMA LoRA training started.\n")
            log.flush()
            process = subprocess.Popen(
                command,
                cwd=str(SD_SCRIPTS_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                match = epoch_pattern.search(line)
                if match:
                    epoch = max(0, min(options["epochs"], int(match.group(1))))
                    current = read_state(job_id)
                    current.update(
                        progress=max(float(current.get("progress", 0.0)), epoch / max(options["epochs"], 1)),
                        message=f"Training epoch {epoch} of {options['epochs']}",
                    )
                    write_state(job_id, current)
            exit_code = process.wait()

        state = read_state(job_id)
        state["finished_at"] = time.time()
        state["exit_code"] = exit_code
        if exit_code != 0:
            tail = log_path.read_text("utf-8", errors="replace")[-3000:]
            state.update(status="failed", progress=float(state.get("progress", 0.0)), error="Training process failed. Last log output:\n" + tail)
        else:
            candidates = sorted(output_dir.glob("*.safetensors"), key=lambda path: path.stat().st_mtime, reverse=True)
            if not candidates:
                state.update(status="failed", error="Training exited successfully but produced no .safetensors file.")
            else:
                installed_filename = None
                install_error = None
                if LORA_INSTALL_DIR:
                    try:
                        install_dir = Path(LORA_INSTALL_DIR)
                        install_dir.mkdir(parents=True, exist_ok=True)
                        installed_path = install_dir / candidates[0].name
                        shutil.copy2(candidates[0], installed_path)
                        installed_filename = installed_path.name
                    except OSError as exc:
                        install_error = f"LoRA finished but could not be installed to the shared model directory: {exc}"
                state.update(
                    status="completed",
                    progress=1.0,
                    message="Training finished.",
                    output_filename=candidates[0].name,
                    installed_filename=installed_filename,
                    install_error=install_error,
                )
        write_state(job_id, state)
    except Exception as exc:
        try:
            state = read_state(job_id)
        except Exception:
            state = {"job_id": job_id}
        state.update(status="failed", error=str(exc), finished_at=time.time())
        write_state(job_id, state)
    finally:
        # The training images/captions may be personal; remove them after training.
        shutil.rmtree(dataset_dir, ignore_errors=True)
        with state_lock:
            job_threads.pop(job_id, None)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "token_configured": bool(TRAINER_API_TOKEN),
        "anima_models_present": all(path.is_file() for path in (ANIMA_DIT, QWEN3, ANIMA_VAE)),
        "sd_scripts_present": (SD_SCRIPTS_ROOT / "anima_train_network.py").is_file(),
    }


@app.post("/api/anima/lora/train")
async def submit_training(
    request: Request,
    authorization: Annotated[HTTPAuthorizationCredentials | None, Depends(AUTH_SCHEME)],
    caption: Annotated[str, Form()],
    trigger_word: Annotated[str, Form()],
    rank: Annotated[int, Form()],
    epochs: Annotated[int, Form()],
    learning_rate: Annotated[float, Form()],
    base_model: Annotated[str, Form()],
    images: Annotated[list[UploadFile], File()],
) -> dict[str, Any]:
    require_token(authorization)
    if base_model.lower() != "anima":
        raise HTTPException(status_code=400, detail="This trainer currently supports ANIMA LoRAs only.")
    if len(images) < 3 or len(images) > MAX_IMAGES:
        raise HTTPException(status_code=400, detail=f"Select between 3 and {MAX_IMAGES} training images.")
    if rank < 4 or rank > 64 or rank % 4 != 0:
        raise HTTPException(status_code=400, detail="rank must be 4 to 64 in increments of 4.")
    if epochs < 1 or epochs > 30:
        raise HTTPException(status_code=400, detail="epochs must be between 1 and 30.")
    if learning_rate < 0.00001 or learning_rate > 0.0003:
        raise HTTPException(status_code=400, detail="learning_rate must be between 0.00001 and 0.0003.")
    if len(trigger_word.strip()) > 80 or not trigger_word.strip():
        raise HTTPException(status_code=400, detail="Enter a non-empty trigger word up to 80 characters.")
    try:
        ensure_model_files()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    job_id = uuid.uuid4().hex
    folder = job_dir(job_id)
    dataset_dir = folder / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=False)
    total_bytes = 0
    try:
        for index, upload in enumerate(images, start=1):
            original_ext = Path(upload.filename or "").suffix.lower()
            extension = original_ext if original_ext in SUPPORTED_EXTENSIONS else ".png"
            contents = await upload.read()
            total_bytes += len(contents)
            if total_bytes > MAX_DATASET_BYTES:
                raise HTTPException(status_code=413, detail=f"Dataset exceeds {MAX_DATASET_BYTES // (1024 * 1024)} MB.")
            if not contents:
                raise HTTPException(status_code=400, detail=f"Training image {index} is empty.")
            image_path = dataset_dir / f"image_{index:04d}{extension}"
            image_path.write_bytes(contents)
            caption_text = ", ".join(part.strip() for part in (trigger_word, caption) if part.strip())
            image_path.with_suffix(".txt").write_text(caption_text, "utf-8")
    except Exception:
        shutil.rmtree(dataset_dir, ignore_errors=True)
        raise

    initial_state = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0.0,
        "message": f"Queued {len(images)} images for ANIMA LoRA training.",
        "image_count": len(images),
        "rank": rank,
        "epochs": epochs,
        "created_at": time.time(),
    }
    write_state(job_id, initial_state)
    options = {"rank": rank, "epochs": epochs, "learning_rate": learning_rate}
    thread = threading.Thread(target=run_training, args=(job_id, options), daemon=True)
    with state_lock:
        if any(active.is_alive() for active in job_threads.values()):
            shutil.rmtree(folder, ignore_errors=True)
            raise HTTPException(status_code=409, detail="This worker is already training a LoRA. Wait for it to finish.")
        job_threads[job_id] = thread
        thread.start()
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/anima/lora/train/{job_id}")
def training_status(
    job_id: str,
    request: Request,
    authorization: Annotated[HTTPAuthorizationCredentials | None, Depends(AUTH_SCHEME)],
) -> dict[str, Any]:
    require_token(authorization)
    state = read_state(job_id)
    response = {
        "job_id": job_id,
        "status": state.get("status", "unknown"),
        "progress": state.get("progress", 0.0),
        "message": state.get("message", ""),
        "error": state.get("error"),
        "installed_filename": state.get("installed_filename"),
        "install_error": state.get("install_error"),
    }
    if response["status"] == "completed":
        response["lora_url"] = str(request.base_url).rstrip("/") + f"/api/anima/lora/train/{job_id}/download"
        response["download_url"] = response["lora_url"]
    return response


@app.get("/api/anima/lora/train/{job_id}/download")
def download_lora(
    job_id: str,
    authorization: Annotated[HTTPAuthorizationCredentials | None, Depends(AUTH_SCHEME)],
) -> FileResponse:
    require_token(authorization)
    state = read_state(job_id)
    if state.get("status") != "completed":
        raise HTTPException(status_code=409, detail="The LoRA is not ready yet.")
    output_dir = job_dir(job_id) / "output"
    filename = state.get("output_filename", "")
    if not filename or Path(filename).name != filename:
        raise HTTPException(status_code=404, detail="The trained LoRA file is unavailable.")
    path = output_dir / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="The trained LoRA file was removed.")
    return FileResponse(path, media_type="application/octet-stream", filename=filename)


@app.get("/api/anima/lora/train/{job_id}/log")
def training_log(
    job_id: str,
    authorization: Annotated[HTTPAuthorizationCredentials | None, Depends(AUTH_SCHEME)],
) -> FileResponse:
    require_token(authorization)
    path = job_dir(job_id) / "training.log"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Training log is not available yet.")
    return FileResponse(path, media_type="text/plain", filename=f"{job_id}.log")
