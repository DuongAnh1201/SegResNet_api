# src/pipeline.py
"""
Full analysis for one job:
  4 MRI files → predict() → clean masks → seg.nii.gz + model.glb + report.json
"""
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import trimesh
from scipy import ndimage

from src.predict import predict
from src.mask_to_mesh import to_mesh

MIN_VOXELS = 100          # blobs smaller than this (≈0.1 cm³ at 1 mm) are treated as noise
SEQUENCES = ("t1c", "t1", "t2", "flair")   # order the model expects

# Layer name → RGBA color for the 3D model
COLORS = {
    "brain":     [200, 200, 200,  60],   # transparent gray
    "edema":     [255, 220,   0, 140],   # yellow
    "core":      [255, 120,   0, 255],   # orange (necrotic / non-enhancing)
    "enhancing": [220,   0,   0, 255],   # red
}


def remove_small_blobs(mask, min_voxels=MIN_VOXELS):
    """Keep only connected regions with at least min_voxels voxels."""
    labeled, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    keep_ids = np.where(sizes >= min_voxels)[0] + 1
    return np.isin(labeled, keep_ids)


def save_label_map(wt, tc, et, affine, path):
    """One NIfTI file in BraTS convention: 0 background, 1 core, 2 edema, 4 enhancing."""
    seg = np.zeros(wt.shape, dtype=np.uint8)
    seg[wt] = 2
    seg[tc] = 1
    seg[et] = 4
    nib.save(nib.Nifti1Image(seg, affine), path)


def build_glb(brain, wt, tc, et, affine, path):
    """Mesh each layer in mm coordinates and pack them into one .glb file."""
    layers = {
        "brain":     brain,
        "edema":     wt & ~tc,     # whole tumor minus core
        "core":      tc & ~et,     # core minus enhancing
        "enhancing": et,
    }
    scene = trimesh.Scene()
    for name, mask in layers.items():
        mesh = to_mesh(mask, affine, COLORS[name])
        if mesh is not None:
            scene.add_geometry(mesh, node_name=name, geom_name=name)
    scene.export(path)


def build_report(brain, wt, tc, et, affine, zooms):
    """Detection, position (mm) and volume (cm³) from the masks."""
    cm3_per_voxel = float(np.prod(zooms)) / 1000.0
    report = {
        "has_tumor": bool(wt.any()),
        "units": "mm",
        "axes": "RAS",
        "centroid_mm": None,
        "bbox_mm": None,
        "side": None,
        "n_lesions": int(ndimage.label(wt)[1]),
        "volume_cm3": {
            "whole_tumor": round(wt.sum() * cm3_per_voxel, 2),
            "core":        round(tc.sum() * cm3_per_voxel, 2),
            "enhancing":   round(et.sum() * cm3_per_voxel, 2),
        },
    }
    if not wt.any():
        return report

    idx = np.argwhere(wt)                                   # voxel indices of the tumor
    centroid = nib.affines.apply_affine(affine, idx.mean(0))
    report["centroid_mm"] = centroid.round(1).tolist()
    report["bbox_mm"] = {
        "min": nib.affines.apply_affine(affine, idx.min(0)).round(1).tolist(),
        "max": nib.affines.apply_affine(affine, idx.max(0)).round(1).tolist(),
    }

    # Side: compare tumor centroid x with the brain's own center x (RAS: +x = right)
    brain_center_x = nib.affines.apply_affine(affine, np.argwhere(brain).mean(0))[0]
    report["side"] = "right" if centroid[0] > brain_center_x else "left"
    return report


def run(job_dir: Path) -> dict:
    job_dir = Path(job_dir)
    paths = [job_dir / f"{s}.nii.gz" for s in SEQUENCES]

    # 1. Model
    r = predict(paths)
    affine, zooms, brain = r["affine"], r["zooms"], r["brain"]

    # 2. Clean: remove specks from WT, keep TC/ET consistent (inside WT)
    wt = remove_small_blobs(r["wt"])
    tc = r["tc"] & wt
    et = r["et"] & tc

    # 3. Outputs
    save_label_map(wt, tc, et, affine, job_dir / "seg.nii.gz")
    build_glb(brain, wt, tc, et, affine, job_dir / "model.glb")
    report = build_report(brain, wt, tc, et, affine, zooms)
    (job_dir / "report.json").write_text(json.dumps(report, indent=2))

    return report


if __name__ == "__main__":
    # Quick test without the API:  uv run python -m src.pipeline jobs/test
    import sys
    print(json.dumps(run(Path(sys.argv[1])), indent=2))