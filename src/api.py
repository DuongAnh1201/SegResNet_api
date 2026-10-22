# src/api.py
"""
REST API for Brain3D.

POST /jobs                 upload 4 MRI files → {"job_id": ...}
GET  /jobs/{id}            status: queued | running | done | failed
GET  /jobs/{id}/report     report.json
GET  /jobs/{id}/model.glb  3D meshes
GET  /jobs/{id}/seg.nii.gz label map
GET  /health               server check
"""
import json
import shutil
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.pipeline import SEQUENCES, run
from src.predict import get_net

JOBS = Path("jobs")
JOBS.mkdir(exist_ok=True)
MAX_FILE_MB = 200

executor = ThreadPoolExecutor(max_workers=1)   # one job at a time: protects CPU/GPU memory


# ---------- status stored on disk, survives server restarts ----------
def _status_path(job_id: str) -> Path:
    return JOBS / job_id / "status.json"

def set_status(job_id: str, **data):
    _status_path(job_id).write_text(json.dumps(data))

def get_status_data(job_id: str) -> dict:
    p = _status_path(job_id)
    if not p.exists():
        raise HTTPException(404, "Unknown job")
    return json.loads(p.read_text())

def _check_id(job_id: str):
    if not job_id.isalnum():                    # blocks paths like "../"
        raise HTTPException(400, "Invalid job id")


# ---------- background worker ----------
def _work(job_id: str):
    set_status(job_id, status="running")
    try:
        run(JOBS / job_id)
        set_status(job_id, status="done")
    except Exception as e:
        traceback.print_exc()
        set_status(job_id, status="failed", error=str(e))


# ---------- app ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    get_net()                                   # load model once at startup, not on first request
    yield
    executor.shutdown(wait=False)

app = FastAPI(title="Brain3D API", lifespan=lifespan)
import os

origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/jobs", status_code=202)
async def create_job(
    t1c: UploadFile = File(...),
    t1: UploadFile = File(...),
    t2: UploadFile = File(...),
    flair: UploadFile = File(...),
):
    files = {"t1c": t1c, "t1": t1, "t2": t2, "flair": flair}

    # validate before saving anything
    for name, f in files.items():
        if not f.filename or not f.filename.endswith((".nii", ".nii.gz")):
            raise HTTPException(400, f"{name}: expected a .nii or .nii.gz file")
        if f.size and f.size > MAX_FILE_MB * 1024 * 1024:
            raise HTTPException(413, f"{name}: larger than {MAX_FILE_MB} MB")

    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS / job_id
    job_dir.mkdir()

    for name in SEQUENCES:
        f = files[name]
        ext = ".nii.gz" if f.filename.endswith(".gz") else ".nii"
        dest = job_dir / f"{name}{ext}"
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        if ext == ".nii":                       # pipeline reads <name>.nii.gz
            import nibabel as nib
            nib.save(nib.load(dest), job_dir / f"{name}.nii.gz")
            dest.unlink()

    set_status(job_id, status="queued")
    executor.submit(_work, job_id)
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    _check_id(job_id)
    return get_status_data(job_id)


def _result(job_id: str, name: str) -> Path:
    _check_id(job_id)
    if get_status_data(job_id)["status"] != "done":
        raise HTTPException(409, "Job not finished")
    p = JOBS / job_id / name
    if not p.exists():
        raise HTTPException(404, f"{name} not found")
    return p


@app.get("/jobs/{job_id}/report")
def job_report(job_id: str):
    return json.loads(_result(job_id, "report.json").read_text())


@app.get("/jobs/{job_id}/model.glb")
def job_model(job_id: str):
    return FileResponse(_result(job_id, "model.glb"), media_type="model/gltf-binary",
                        filename=f"{job_id}.glb")


@app.get("/jobs/{job_id}/seg.nii.gz")
def job_seg(job_id: str):
    return FileResponse(_result(job_id, "seg.nii.gz"), media_type="application/gzip",
                        filename=f"{job_id}_seg.nii.gz")

