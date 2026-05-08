# -*- coding: utf-8 -*-
import os, sys
os.environ.setdefault("PYTHONUTF8", "1")
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    try: sys.stderr.reconfigure(encoding='utf-8')
    except Exception: pass

"""
============================================================
  YOLOv8 Helmet Detection - Optimized Training Pipeline
  Classes: helmet (0), without_helmet (1)
  Author: Auto-generated training script
============================================================
"""

import os
import sys
import shutil
import yaml
import json
import time
import glob
import logging
from pathlib import Path
from datetime import datetime

# ── Dependency check ──────────────────────────────────────
def check_and_install(pkg, import_name=None):
    import importlib
    name = import_name or pkg
    try:
        importlib.import_module(name)
    except ImportError:
        print(f"[SETUP] Installing {pkg}...")
        os.system(f"{sys.executable} -m pip install {pkg} -q")

for pkg, imp in [("ultralytics", "ultralytics"), ("matplotlib", "matplotlib"),
                 ("numpy", "numpy"), ("pyyaml", "yaml"), ("seaborn", "seaborn")]:
    check_and_install(pkg, imp)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Logging setup ─────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
REPORT_DIR = BASE_DIR / "reports" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(REPORT_DIR / "training.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
#  1. DATASET ANALYSIS & YAML GENERATION
# ═══════════════════════════════════════════════════════════
DATASET_ROOT = BASE_DIR / "normalized_dataset"
CLASSES      = ["helmet", "without_helmet"]

def count_files(path, ext="*.jpg"):
    return len(list(path.glob(ext))) + len(list(path.glob("*.png"))) + len(list(path.glob("*.jpeg")))

def analyze_dataset():
    log.info("=" * 60)
    log.info("  DATASET ANALYSIS")
    log.info("=" * 60)

    stats = {}
    for split in ["train", "valid", "test"]:
        stats[split] = {"images": 0, "labels": 0, "annotations": 0, "classes": {c: 0 for c in CLASSES}}
        for cls_folder, cls_idx in [("With_helmet", 0), ("Without_helmet", 1)]:
            img_dir = DATASET_ROOT / cls_folder / split / "images"
            lbl_dir = DATASET_ROOT / cls_folder / split / "labels"
            if not img_dir.exists():
                log.warning(f"  Missing: {img_dir}")
                continue
            n_imgs = count_files(img_dir)
            n_lbls = count_files(lbl_dir, "*.txt")
            stats[split]["images"]  += n_imgs
            stats[split]["labels"]  += n_lbls
            stats[split]["classes"][CLASSES[cls_idx]] += n_imgs
            # Count annotation lines
            for lbl_file in lbl_dir.glob("*.txt"):
                try:
                    lines = [l.strip() for l in lbl_file.read_text().splitlines() if l.strip()]
                    stats[split]["annotations"] += len(lines)
                except Exception:
                    pass
        log.info(f"  [{split:5s}] images={stats[split]['images']:5d}  "
                 f"labels={stats[split]['labels']:5d}  "
                 f"annotations={stats[split]['annotations']:6d}  "
                 f"helmet={stats[split]['classes']['helmet']:5d}  "
                 f"without_helmet={stats[split]['classes']['without_helmet']:5d}")

    # Save stats
    with open(REPORT_DIR / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Plot class distribution
    _plot_class_distribution(stats)
    return stats

def _plot_class_distribution(stats):
    splits = list(stats.keys())
    helmet_counts  = [stats[s]["classes"]["helmet"] for s in splits]
    no_helmet_counts = [stats[s]["classes"]["without_helmet"] for s in splits]
    x = np.arange(len(splits))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, helmet_counts,   width, label="helmet",         color="#2ecc71", edgecolor="black")
    ax.bar(x + width/2, no_helmet_counts, width, label="without_helmet", color="#e74c3c", edgecolor="black")
    ax.set_xlabel("Split"); ax.set_ylabel("Image Count")
    ax.set_title("Class Distribution per Split")
    ax.set_xticks(x); ax.set_xticklabels([s.capitalize() for s in splits])
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "class_distribution.png", dpi=150)
    plt.close()
    log.info("  Saved class_distribution.png")

# ═══════════════════════════════════════════════════════════
#  2. VALIDATE YOLO ANNOTATIONS
# ═══════════════════════════════════════════════════════════
def validate_annotations():
    log.info("=" * 60)
    log.info("  ANNOTATION VALIDATION")
    log.info("=" * 60)

    total, valid_count, invalid = 0, 0, []
    for cls_folder in ["With_helmet", "Without_helmet"]:
        for split in ["train", "valid", "test"]:
            lbl_dir = DATASET_ROOT / cls_folder / split / "labels"
            if not lbl_dir.exists():
                continue
            for lbl_file in lbl_dir.glob("*.txt"):
                total += 1
                try:
                    content = lbl_file.read_text().strip()
                    if not content:          # empty label → background image (OK)
                        valid_count += 1
                        continue
                    ok = True
                    for line in content.splitlines():
                        parts = line.strip().split()
                        if len(parts) != 5:
                            ok = False; break
                        cls_id = int(parts[0])
                        vals   = [float(p) for p in parts[1:]]
                        if cls_id not in (0, 1):
                            ok = False; break
                        if not all(0.0 <= v <= 1.0 for v in vals):
                            ok = False; break
                    if ok:
                        valid_count += 1
                    else:
                        invalid.append(str(lbl_file))
                except Exception as e:
                    invalid.append(f"{lbl_file} ({e})")

    log.info(f"  Total label files : {total}")
    log.info(f"  Valid             : {valid_count}")
    log.info(f"  Invalid / Corrupt : {len(invalid)}")
    if invalid:
        log.warning("  First 10 invalid files:")
        for f in invalid[:10]:
            log.warning(f"    {f}")
    else:
        log.info("  [OK] All annotations are valid YOLO format!")
    return len(invalid) == 0

# ═══════════════════════════════════════════════════════════
#  3. BUILD FLAT YOLO DATASET YAML
# ═══════════════════════════════════════════════════════════
def build_flat_dataset():
    """
    YOLOv8 works best with a flat structure:
      dataset/
        train/images  train/labels
        valid/images  valid/labels
        test/images   test/labels
    We symlink (or copy on Windows) images+labels from both classes
    into a unified flat structure and write a clean YAML.
    """
    log.info("=" * 60)
    log.info("  BUILDING FLAT DATASET STRUCTURE")
    log.info("=" * 60)

    flat_dir = BASE_DIR / "flat_dataset"

    for split in ["train", "valid", "test"]:
        (flat_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (flat_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    copied_imgs, copied_lbls = 0, 0
    for cls_folder, cls_idx in [("With_helmet", 0), ("Without_helmet", 1)]:
        for split in ["train", "valid", "test"]:
            src_img = DATASET_ROOT / cls_folder / split / "images"
            src_lbl = DATASET_ROOT / cls_folder / split / "labels"
            dst_img = flat_dir / split / "images"
            dst_lbl = flat_dir / split / "labels"

            if not src_img.exists():
                continue

            prefix = f"{'hel' if cls_idx == 0 else 'noh'}_"

            for img_path in list(src_img.glob("*.jpg")) + list(src_img.glob("*.png")) + list(src_img.glob("*.jpeg")):
                dst = dst_img / (prefix + img_path.name)
                if not dst.exists():
                    shutil.copy2(img_path, dst)
                    copied_imgs += 1

            if src_lbl.exists():
                for lbl_path in src_lbl.glob("*.txt"):
                    dst = dst_lbl / (prefix + lbl_path.name)
                    if not dst.exists():
                        shutil.copy2(lbl_path, dst)
                        copied_lbls += 1

    log.info(f"  Copied {copied_imgs} images, {copied_lbls} label files → {flat_dir}")

    # Write YAML
    yaml_path = BASE_DIR / "helmet_train.yaml"
    cfg = {
        "path": str(flat_dir.resolve()),
        "train": "train/images",
        "val":   "valid/images",
        "test":  "test/images",
        "nc":    2,
        "names": CLASSES,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    log.info(f"  Dataset YAML written → {yaml_path}")
    return yaml_path

# ═══════════════════════════════════════════════════════════
#  4. DETECT GPU
# ═══════════════════════════════════════════════════════════
def get_device():
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            log.info(f"  ✅ GPU detected: {gpu} ({mem:.1f} GB)")
            return "0", True
        else:
            log.info("  ⚠ No GPU found – training on CPU (will be slow)")
            return "cpu", False
    except Exception:
        return "cpu", False

# ═══════════════════════════════════════════════════════════
#  5. TRAIN
# ═══════════════════════════════════════════════════════════
def train(yaml_path, dataset_stats, use_gpu):
    from ultralytics import YOLO

    log.info("=" * 60)
    log.info("  STARTING YOLOV8 TRAINING")
    log.info("=" * 60)

    weights = BASE_DIR / "yolov8n.pt"
    if not weights.exists():
        log.info("  Downloading yolov8n.pt ...")
        weights = "yolov8n.pt"

    # Compute class weights for imbalance handling
    train_stats = dataset_stats.get("train", {})
    n_helmet    = max(train_stats.get("classes", {}).get("helmet", 1), 1)
    n_no_helmet = max(train_stats.get("classes", {}).get("without_helmet", 1), 1)
    total_train = n_helmet + n_no_helmet
    log.info(f"  Train set: helmet={n_helmet}, without_helmet={n_no_helmet}")

    # Adaptive batch size
    device, has_gpu = use_gpu
    batch  = 8 if has_gpu else 4
    epochs = 50
    imgsz  = 640
    workers = 0  # 0 = main process loading (fixes Windows DataLoader memory errors)

    model = YOLO(str(weights))

    run_name = f"helmet_yolov8n_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    train_args = dict(
        data          = str(yaml_path),
        epochs        = epochs,
        imgsz         = imgsz,
        batch         = batch,
        device        = device,
        workers       = workers,
        project       = str(BASE_DIR / "runs" / "train"),
        name          = run_name,
        exist_ok      = True,
        pretrained    = True,
        optimizer     = "AdamW",
        lr0           = 0.001,
        lrf           = 0.01,
        momentum      = 0.937,
        weight_decay  = 0.0005,
        warmup_epochs = 5,
        warmup_momentum = 0.8,
        cos_lr        = True,        # Cosine LR schedule
        patience      = 20,          # Early stopping
        save          = True,
        save_period   = 10,
        cache         = False,
        amp           = has_gpu,     # Mixed precision on GPU
        # Built-in augmentation
        mosaic        = 1.0,
        mixup         = 0.15,
        copy_paste    = 0.1,
        degrees       = 10.0,
        translate     = 0.1,
        scale         = 0.5,
        shear         = 2.0,
        perspective   = 0.0005,
        flipud        = 0.01,
        fliplr        = 0.5,
        hsv_h         = 0.015,
        hsv_s         = 0.7,
        hsv_v         = 0.4,
        # Logging
        plots         = True,
        verbose       = True,
        seed          = 42,
    )

    log.info(f"  Config: epochs={epochs}, batch={batch}, imgsz={imgsz}, device={device}")
    log.info(f"  Output: runs/train/{run_name}")

    t0 = time.time()
    results = model.train(**train_args)
    elapsed = time.time() - t0
    log.info(f"  Training complete in {elapsed/60:.1f} minutes")

    return model, results, run_name

# ═══════════════════════════════════════════════════════════
#  6. EVALUATE
# ═══════════════════════════════════════════════════════════
def evaluate(model, yaml_path, run_name):
    log.info("=" * 60)
    log.info("  MODEL EVALUATION")
    log.info("=" * 60)

    # Validation metrics
    val_metrics = model.val(data=str(yaml_path), split="val",  verbose=True)
    test_metrics = model.val(data=str(yaml_path), split="test", verbose=True)

    def extract_metrics(m, label):
        try:
            mp   = float(m.box.mp)          # mean precision
            mr   = float(m.box.mr)          # mean recall
            map50= float(m.box.map50)       # mAP@0.5
            map  = float(m.box.map)         # mAP@0.5:0.95
            f1   = 2 * mp * mr / (mp + mr + 1e-9)
            log.info(f"  [{label}] Precision={mp:.4f}  Recall={mr:.4f}  "
                     f"F1={f1:.4f}  mAP@0.5={map50:.4f}  mAP@0.5:0.95={map:.4f}")
            return {"precision": mp, "recall": mr, "f1": f1, "map50": map50, "map": map}
        except Exception as e:
            log.warning(f"  Could not extract metrics for {label}: {e}")
            return {}

    val_dict  = extract_metrics(val_metrics,  "Validation")
    test_dict = extract_metrics(test_metrics, "Test")

    report = {"validation": val_dict, "test": test_dict, "run_name": run_name}
    with open(REPORT_DIR / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    _plot_metrics_bar(val_dict, test_dict)
    return val_dict, test_dict

def _plot_metrics_bar(val_dict, test_dict):
    if not val_dict:
        return
    metrics = ["precision", "recall", "f1", "map50", "map"]
    labels  = ["Precision", "Recall", "F1", "mAP@0.5", "mAP@0.5:0.95"]
    val_v   = [val_dict.get(m, 0) for m in metrics]
    test_v  = [test_dict.get(m, 0) for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, val_v,  width, label="Validation", color="#3498db", edgecolor="black")
    ax.bar(x + width/2, test_v, width, label="Test",       color="#e67e22", edgecolor="black")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score"); ax.set_title("Model Evaluation Metrics")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    for i, (v, t) in enumerate(zip(val_v, test_v)):
        ax.text(i - width/2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + width/2, t + 0.01, f"{t:.3f}", ha="center", va="bottom", fontsize=8)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "evaluation_metrics.png", dpi=150)
    plt.close()
    log.info("  Saved evaluation_metrics.png")

# ═══════════════════════════════════════════════════════════
#  7. EXPORT BEST MODEL
# ═══════════════════════════════════════════════════════════
def export_best_model(run_name):
    log.info("=" * 60)
    log.info("  EXPORTING BEST MODEL")
    log.info("=" * 60)

    from ultralytics import YOLO

    best_pt = BASE_DIR / "runs" / "train" / run_name / "weights" / "best.pt"
    if not best_pt.exists():
        # Fallback search
        candidates = list((BASE_DIR / "runs" / "train").glob("**/best.pt"))
        if candidates:
            best_pt = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        else:
            log.error("  best.pt not found – skipping export")
            return

    log.info(f"  Best weights: {best_pt}")

    # Copy to models/
    models_dir = BASE_DIR / "models"
    models_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst_pt = models_dir / f"helmet_best_{ts}.pt"
    shutil.copy2(best_pt, dst_pt)
    shutil.copy2(best_pt, models_dir / "best.pt")   # overwrite latest
    log.info(f"  Saved → {dst_pt}")
    log.info(f"  Saved → {models_dir / 'best.pt'} (latest)")

    # ONNX export
    try:
        model = YOLO(str(best_pt))
        model.export(format="onnx", imgsz=640, simplify=True)
        onnx_src = best_pt.parent / "best.onnx"
        if onnx_src.exists():
            shutil.copy2(onnx_src, models_dir / f"helmet_best_{ts}.onnx")
            shutil.copy2(onnx_src, models_dir / "best.onnx")
            log.info("  ONNX export saved → models/best.onnx")
    except Exception as e:
        log.warning(f"  ONNX export failed: {e}")

    # Benchmark inference speed
    try:
        import torch
        model = YOLO(str(best_pt))
        dummy = BASE_DIR / "normalized_dataset" / "With_helmet" / "valid" / "images"
        sample = next(dummy.glob("*.jpg"), None)
        if sample:
            t0 = time.time()
            for _ in range(20):
                model.predict(str(sample), imgsz=640, verbose=False)
            avg_ms = (time.time() - t0) / 20 * 1000
            fps = 1000 / avg_ms
            log.info(f"  Inference speed: {avg_ms:.1f} ms/img  ({fps:.1f} FPS)")
            with open(REPORT_DIR / "inference_speed.json", "w") as f:
                json.dump({"avg_ms": round(avg_ms, 2), "fps": round(fps, 2)}, f, indent=2)
    except Exception as e:
        log.warning(f"  Benchmark failed: {e}")

    return dst_pt

# ═══════════════════════════════════════════════════════════
#  8. FINAL SUMMARY REPORT
# ═══════════════════════════════════════════════════════════
def write_summary(dataset_stats, val_dict, test_dict, run_name):
    lines = []
    lines.append("=" * 60)
    lines.append("  HELMET DETECTION - TRAINING SUMMARY REPORT")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)
    lines.append("")
    lines.append("DATASET STATISTICS")
    lines.append("-" * 40)
    for split, s in dataset_stats.items():
        lines.append(f"  {split:6s}: {s['images']:5d} images | "
                     f"helmet={s['classes']['helmet']:5d} | "
                     f"without_helmet={s['classes']['without_helmet']:5d} | "
                     f"annotations={s['annotations']:6d}")
    lines.append("")
    lines.append("VALIDATION METRICS")
    lines.append("-" * 40)
    for k, v in val_dict.items():
        lines.append(f"  {k:20s}: {v:.4f}")
    lines.append("")
    lines.append("TEST METRICS")
    lines.append("-" * 40)
    for k, v in test_dict.items():
        lines.append(f"  {k:20s}: {v:.4f}")
    lines.append("")
    lines.append("OUTPUT FILES")
    lines.append("-" * 40)
    lines.append(f"  Best weights  : models/best.pt")
    lines.append(f"  ONNX model    : models/best.onnx")
    lines.append(f"  Training runs : runs/train/{run_name}/")
    lines.append(f"  Report dir    : {REPORT_DIR}")
    lines.append("")
    lines.append("DEPLOYMENT NOTES")
    lines.append("-" * 40)
    lines.append("  from ultralytics import YOLO")
    lines.append("  model = YOLO('models/best.pt')")
    lines.append("  results = model.predict('image.jpg', conf=0.25, iou=0.45)")
    lines.append("")

    summary_text = "\n".join(lines)
    print("\n" + summary_text)
    with open(REPORT_DIR / "summary_report.txt", "w") as f:
        f.write(summary_text)
    log.info(f"  Summary saved → {REPORT_DIR / 'summary_report.txt'}")

# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════
def main():
    log.info("")
    log.info("=" * 60)
    log.info("  YOLOv8 HELMET DETECTION TRAINING PIPELINE")
    log.info("=" * 60)
    log.info(f"  Report directory: {REPORT_DIR}")
    log.info("")

    # Step 1: Analyze dataset
    dataset_stats = analyze_dataset()

    # Step 2: Validate annotations
    validate_annotations()

    # Step 3: Build flat YOLO dataset
    yaml_path = build_flat_dataset()

    # Step 4: Detect GPU
    device_info = get_device()
    device_str, has_gpu = device_info

    # Step 5: Train
    model, results, run_name = train(yaml_path, dataset_stats, (device_str, has_gpu))

    # Step 6: Evaluate
    val_dict, test_dict = evaluate(model, yaml_path, run_name)

    # Step 7: Export
    export_best_model(run_name)

    # Step 8: Summary
    write_summary(dataset_stats, val_dict, test_dict, run_name)

    log.info("")
    log.info("✅ Pipeline complete! Best model saved to models/best.pt")

if __name__ == "__main__":
    main()
