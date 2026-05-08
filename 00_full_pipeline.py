# -*- coding: utf-8 -*-
"""
Full Dataset Preprocessing Pipeline for YOLOv8 Helmet Detection
================================================================
Works with:  With Helmet/  and  Without Helmet/  folder structure
Steps:
  1. Validate folder structure & YOLO annotation format
  2. Detect corrupted / missing images and labels
  3. Remove duplicate images (MD5 hash)
  4. Remap class IDs  (With Helmet -> 0, Without Helmet -> 1)
  5. Resize & normalize all images to 640x640
  6. Balance class distribution via augmentation
  7. Apply augmentations: flip, rotate, brightness, blur, noise
  8. Split dataset  70 / 20 / 10  (train / val / test)
  9. Generate combined_data.yaml
"""

import os, sys, shutil, hashlib, random, json
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from pathlib import Path
from collections import Counter, defaultdict

try:
    from PIL import Image, ImageEnhance, ImageFilter
    import numpy as np
    PIL_OK = True
except ImportError:
    PIL_OK = False
    print("[ERROR] Install:  pip install Pillow numpy")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MPL_OK = True
except ImportError:
    MPL_OK = False

try:
    import yaml
    YAML_OK = True
except ImportError:
    YAML_OK = False
    print("[WARN] PyYAML not found – will write YAML manually.")

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
BASE_DIR      = Path(r"D:\Helmet_detection")
SRC_HELMET    = BASE_DIR / "With Helmet"
SRC_NOHELMET  = BASE_DIR / "Without Helmet"
OUT_DIR       = BASE_DIR / "combined_dataset"
QUARANTINE    = BASE_DIR / "quarantine"
REPORTS_DIR   = BASE_DIR / "reports"
YAML_OUT      = BASE_DIR / "combined_data.yaml"

CLASS_NAMES   = ["helmet", "no_helmet"]   # 0, 1
IMG_SIZE      = 640
TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.20
TEST_RATIO    = 0.10
TARGET_RATIO  = 1.0          # desired majority/minority ratio after aug
IMG_EXTS      = {'.jpg','.jpeg','.png','.bmp','.webp','.tiff'}
RANDOM_SEED   = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def ensure_dirs():
    for split in ('train','val','test'):
        (OUT_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
        (OUT_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)
    QUARANTINE.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def img_files(d):
    if not d.exists(): return []
    return [f for f in d.iterdir() if f.suffix.lower() in IMG_EXTS]

def lbl_files(d):
    if not d.exists(): return []
    return [f for f in d.iterdir() if f.suffix.lower() == '.txt']

def md5(fp):
    h = hashlib.md5()
    with open(fp,'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def resize_image(img: Image.Image, size=IMG_SIZE) -> Image.Image:
    """Resize with letterbox to preserve aspect ratio."""
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = int(w*scale), int(h*scale)
    resized = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new('RGB', (size, size), (114, 114, 114))
    pad_x, pad_y = (size - nw)//2, (size - nh)//2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, pad_x, pad_y, scale

def adjust_labels_for_letterbox(labels, orig_w, orig_h,
                                  pad_x, pad_y, scale, size=IMG_SIZE):
    """Recalculate YOLO coords after letterbox resize."""
    out = []
    for line in labels:
        p = line.strip().split()
        if len(p) < 5: continue
        cls = p[0]
        cx, cy, bw, bh = float(p[1]), float(p[2]), float(p[3]), float(p[4])
        # Convert to pixel coords in original image
        cx_px = cx * orig_w;  cy_px = cy * orig_h
        bw_px = bw * orig_w;  bh_px = bh * orig_h
        # Scale and shift
        cx_new = (cx_px * scale + pad_x) / size
        cy_new = (cy_px * scale + pad_y) / size
        bw_new = bw_px * scale / size
        bh_new = bh_px * scale / size
        # Clamp
        cx_new = max(0.0, min(1.0, cx_new))
        cy_new = max(0.0, min(1.0, cy_new))
        bw_new = max(0.001, min(1.0, bw_new))
        bh_new = max(0.001, min(1.0, bh_new))
        out.append(f"{cls} {cx_new:.6f} {cy_new:.6f} {bw_new:.6f} {bh_new:.6f}")
    return out

# ──────────────────────────────────────────────
# STEP 1 – VALIDATE STRUCTURE & ANNOTATIONS
# ──────────────────────────────────────────────

def validate(source_map):
    print("\n" + "="*60)
    print("STEP 1: VALIDATE FOLDER STRUCTURE & ANNOTATIONS")
    print("="*60)

    issues = []
    class_counts = Counter()
    total_imgs = total_lbls = 0

    for cls_id, (cls_name, src) in source_map.items():
        for split in ('train','valid','test'):
            img_dir = src / split / 'images'
            lbl_dir = src / split / 'labels'
            if not img_dir.exists():
                print(f"  [SKIP] {cls_name}/{split}/images – not found")
                continue
            imgs = img_files(img_dir)
            lbls = lbl_files(lbl_dir) if lbl_dir.exists() else []
            img_stems = {f.stem for f in imgs}
            lbl_stems = {f.stem for f in lbls}
            total_imgs += len(imgs)
            total_lbls += len(lbls)
            orphans = lbl_stems - img_stems
            missing = img_stems - lbl_stems
            if orphans:
                issues.append(f"  Orphan labels ({cls_name}/{split}): {len(orphans)}")
            if missing:
                issues.append(f"  Images without labels ({cls_name}/{split}): {len(missing)}")
            for lf in lbls:
                try:
                    with open(lf) as f:
                        for i, line in enumerate(f,1):
                            line = line.strip()
                            if not line: continue
                            p = line.split()
                            if len(p) < 5:
                                issues.append(f"  Short annotation {lf.name}:L{i}")
                                continue
                            cid = int(p[0])
                            class_counts[cls_id] += 1
                except Exception as e:
                    issues.append(f"  Read error {lf}: {e}")
            print(f"  [OK] {cls_name}/{split}: {len(imgs)} images, {len(lbls)} labels")

    print(f"\n  Total: {total_imgs} images, {total_lbls} labels")
    print(f"  Annotation counts: helmet={class_counts[0]}, no_helmet={class_counts[1]}")
    if issues:
        print(f"\n  [!] {len(issues)} issues found:")
        for iss in issues[:20]: print(f"    {iss}")
    else:
        print("\n  [OK] No structural issues found.")
    return class_counts

# ──────────────────────────────────────────────
# STEP 2 – DETECT CORRUPTED IMAGES
# ──────────────────────────────────────────────

def detect_corrupted(source_map):
    print("\n" + "="*60)
    print("STEP 2: DETECT CORRUPTED IMAGES")
    print("="*60)
    corrupted = []
    checked = 0
    for cls_id, (cls_name, src) in source_map.items():
        for split in ('train','valid','test'):
            img_dir = src / split / 'images'
            if not img_dir.exists(): continue
            for img in img_files(img_dir):
                checked += 1
                try:
                    with Image.open(img) as im: im.verify()
                    with Image.open(img) as im: im.load()
                except Exception as e:
                    corrupted.append((img, str(e)))
                    print(f"  [X] CORRUPT: {cls_name}/{split}/{img.name}")
    if corrupted:
        print(f"\n  Quarantining {len(corrupted)} corrupted images...")
        for fp, _ in corrupted:
            shutil.move(str(fp), str(QUARANTINE / fp.name))
            lbl = fp.parent.parent / 'labels' / (fp.stem + '.txt')
            if lbl.exists():
                shutil.move(str(lbl), str(QUARANTINE / lbl.name))
    else:
        print(f"  [OK] All {checked} images OK.")
    return corrupted

# ──────────────────────────────────────────────
# STEP 3 – REMOVE DUPLICATES
# ──────────────────────────────────────────────

def remove_duplicates(source_map):
    print("\n" + "="*60)
    print("STEP 3: REMOVE DUPLICATE IMAGES")
    print("="*60)
    hash_map = defaultdict(list)
    for cls_id, (cls_name, src) in source_map.items():
        for split in ('train','valid','test'):
            img_dir = src / split / 'images'
            if not img_dir.exists(): continue
            for img in img_files(img_dir):
                hash_map[md5(img)].append((cls_name, split, img))
    dups = {h:v for h,v in hash_map.items() if len(v)>1}
    removed = 0
    if dups:
        print(f"  Found {len(dups)} duplicate groups:")
        for h, files in dups.items():
            for i,(cn,sp,fp) in enumerate(files):
                if i==0:
                    print(f"    KEEP   {cn}/{sp}/{fp.name}")
                else:
                    if not fp.exists():
                        continue   # already moved in a prior run
                    print(f"    REMOVE {cn}/{sp}/{fp.name}")
                    try:
                        dest = QUARANTINE / f"dup_{fp.name}"
                        shutil.copy2(str(fp), str(dest))
                        fp.unlink()
                    except Exception as e:
                        print(f"      [!] Could not quarantine {fp.name}: {e}")
                        continue
                    lbl = fp.parent.parent / 'labels' / (fp.stem + '.txt')
                    if lbl.exists():
                        try:
                            shutil.copy2(str(lbl), str(QUARANTINE / f"dup_{lbl.name}"))
                            lbl.unlink()
                        except Exception:
                            pass
                    removed += 1
        print(f"  Removed {removed} duplicates.")
    else:
        total = sum(len(v) for v in hash_map.values())
        print(f"  [OK] No duplicates in {total} images.")
    return removed

# ──────────────────────────────────────────────
# STEP 4+5 – COLLECT, REMAP CLASS IDs, RESIZE & SAVE
# ──────────────────────────────────────────────

def collect_and_resize(source_map):
    """
    Collect all image+label pairs, remap class IDs,
    resize to 640x640, and return list of (img_path, lbl_lines, cls_id).
    Saves processed images/labels to a temp staging area.
    """
    print("\n" + "="*60)
    print("STEP 4+5: COLLECT, REMAP CLASS IDs & RESIZE TO 640×640")
    print("="*60)

    STAGE = BASE_DIR / "_stage"
    STAGE.mkdir(exist_ok=True)

    entries = []   # (staged_img_path, staged_lbl_path, dominant_cls_id)
    counter = Counter()

    for cls_id, (cls_name, src) in source_map.items():
        for split in ('train','valid','test'):
            img_dir = src / split / 'images'
            lbl_dir = src / split / 'labels'
            if not img_dir.exists(): continue

            for img_fp in img_files(img_dir):
                lbl_fp = lbl_dir / (img_fp.stem + '.txt')

                # Read label lines
                raw_labels = []
                if lbl_fp.exists():
                    with open(lbl_fp) as f:
                        raw_labels = [l.strip() for l in f if l.strip()]

                # Remap: always force class to cls_id for this source folder
                remapped = []
                for line in raw_labels:
                    parts = line.split()
                    if len(parts) >= 5:
                        parts[0] = str(cls_id)
                        remapped.append(' '.join(parts))

                # Resize image
                try:
                    with Image.open(img_fp) as im:
                        orig_w, orig_h = im.size
                        rgb = im.convert('RGB')
                    resized, px, py, sc = resize_image(rgb, IMG_SIZE)
                    adj_labels = adjust_labels_for_letterbox(
                        remapped, orig_w, orig_h, px, py, sc, IMG_SIZE)
                except Exception as e:
                    print(f"  [SKIP] {img_fp.name}: {e}")
                    continue

                # Save to stage
                safe_name = f"{cls_name.replace(' ','_')}_{split}_{img_fp.stem}"
                s_img = STAGE / (safe_name + '.jpg')
                s_lbl = STAGE / (safe_name + '.txt')
                resized.save(str(s_img), 'JPEG', quality=95)
                with open(s_lbl,'w') as f:
                    f.write('\n'.join(adj_labels) + ('\n' if adj_labels else ''))

                entries.append((s_img, s_lbl, cls_id))
                counter[cls_id] += 1

    print(f"  helmet   (class 0): {counter[0]} images")
    print(f"  no_helmet(class 1): {counter[1]} images")
    print(f"  Total processed:    {len(entries)}")
    return entries, STAGE

# ──────────────────────────────────────────────
# STEP 6 – AUGMENTATION FUNCTIONS
# ──────────────────────────────────────────────

def aug_flip(img, labels):
    out = img.transpose(Image.FLIP_LEFT_RIGHT)
    new = []
    for ln in labels:
        p = ln.split()
        if len(p)<5: continue
        cx = 1.0 - float(p[1])
        new.append(f"{p[0]} {cx:.6f} {p[2]} {p[3]} {p[4]}")
    return out, new

def aug_rotate(img, labels, deg=12):
    angle = random.uniform(-deg, deg)
    out = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(114,114,114))
    return out, labels   # small angle – keep labels approx

def aug_brightness(img, labels):
    factor = random.uniform(0.6, 1.4)
    out = ImageEnhance.Brightness(img).enhance(factor)
    factor2 = random.uniform(0.7, 1.3)
    out = ImageEnhance.Contrast(out).enhance(factor2)
    return out, labels

def aug_blur(img, labels):
    radius = random.uniform(0.5, 1.5)
    out = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return out, labels

def aug_noise(img, labels):
    arr = np.array(img, dtype=np.int16)
    noise = np.random.randint(-20, 20, arr.shape, dtype=np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr), labels

def aug_saturation(img, labels):
    factor = random.uniform(0.6, 1.4)
    out = ImageEnhance.Color(img).enhance(factor)
    return out, labels

AUG_POOL = [aug_flip, aug_rotate, aug_brightness, aug_blur, aug_noise, aug_saturation]

def apply_augmentation(img, labels):
    n = random.randint(1, 3)
    fns = random.sample(AUG_POOL, n)
    for fn in fns:
        img, labels = fn(img, labels)
    return img, labels

# ──────────────────────────────────────────────
# STEP 6 – BALANCE VIA AUGMENTATION
# ──────────────────────────────────────────────

def balance_and_augment(entries):
    print("\n" + "="*60)
    print("STEP 6: BALANCE CLASS DISTRIBUTION VIA AUGMENTATION")
    print("="*60)

    by_cls = defaultdict(list)
    for e in entries:
        by_cls[e[2]].append(e)

    counts = {c: len(v) for c, v in by_cls.items()}
    print(f"  Before: " + ", ".join(f"{CLASS_NAMES[c]}={v}" for c,v in counts.items()))

    if len(counts) < 2:
        print("  Only one class – skipping balance.")
        return entries

    max_cls  = max(counts, key=counts.get)
    min_cls  = min(counts, key=counts.get)
    target   = int(counts[max_cls] * TARGET_RATIO)
    deficit  = target - counts[min_cls]

    if deficit <= 0:
        print("  Classes already balanced.")
        return entries

    print(f"  Need {deficit} more '{CLASS_NAMES[min_cls]}' images...")

    STAGE_AUG = Path(str(entries[0][0].parent) + "_aug")
    STAGE_AUG.mkdir(exist_ok=True)

    src_pool = by_cls[min_cls]
    new_entries = list(entries)
    gen = 0

    while gen < deficit:
        s_img, s_lbl, cls_id = random.choice(src_pool)
        try:
            with Image.open(s_img) as im:
                img = im.convert('RGB')
            with open(s_lbl) as f:
                labels = [l.strip() for l in f if l.strip()]
            aug_img, aug_labels = apply_augmentation(img, labels)
        except Exception as e:
            continue

        stem = f"aug_{gen}_{s_img.stem}"
        out_img = STAGE_AUG / (stem + '.jpg')
        out_lbl = STAGE_AUG / (stem + '.txt')
        aug_img.save(str(out_img), 'JPEG', quality=90)
        with open(out_lbl,'w') as f:
            f.write('\n'.join(aug_labels) + '\n')

        new_entries.append((out_img, out_lbl, cls_id))
        gen += 1
        if gen % 100 == 0:
            print(f"    Generated {gen}/{deficit}...")

    print(f"  [OK] Generated {gen} augmented images.")

    after = Counter(e[2] for e in new_entries)
    print(f"  After:  " + ", ".join(f"{CLASS_NAMES[c]}={after[c]}" for c in sorted(after)))
    return new_entries

# ──────────────────────────────────────────────
# STEP 7 – SPLIT  70 / 20 / 10
# ──────────────────────────────────────────────

def split_dataset(entries):
    print("\n" + "="*60)
    print("STEP 7: SPLIT  70% train / 20% val / 10% test")
    print("="*60)

    shuffled = list(entries)
    random.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    splits = {
        'train': shuffled[:n_train],
        'val':   shuffled[n_train:n_train+n_val],
        'test':  shuffled[n_train+n_val:],
    }

    for split, items in splits.items():
        print(f"  {split}: {len(items)} images")
        for idx, (s_img, s_lbl, _) in enumerate(items):
            dst_img = OUT_DIR / split / 'images' / s_img.name
            dst_lbl = OUT_DIR / split / 'labels' / (s_img.stem + '.txt')
            shutil.copy2(str(s_img), str(dst_img))
            if s_lbl.exists():
                shutil.copy2(str(s_lbl), str(dst_lbl))

    return splits

# ──────────────────────────────────────────────
# STEP 8 – GENERATE combined_data.yaml
# ──────────────────────────────────────────────

def generate_yaml():
    print("\n" + "="*60)
    print("STEP 8: GENERATE combined_data.yaml")
    print("="*60)

    path_str = str(OUT_DIR).replace('\\','/')
    content = {
        'path':  path_str,
        'train': 'train/images',
        'val':   'val/images',
        'test':  'test/images',
        'nc':    2,
        'names': CLASS_NAMES,
    }

    if YAML_OK:
        import yaml
        with open(YAML_OUT,'w') as f:
            yaml.dump(content, f, default_flow_style=False, sort_keys=False)
    else:
        with open(YAML_OUT,'w') as f:
            f.write(f"path: {path_str}\n")
            f.write("train: train/images\nval: val/images\ntest: test/images\n")
            f.write("nc: 2\nnames:\n- helmet\n- no_helmet\n")

    print(f"  [OK] Saved: {YAML_OUT}")
    with open(YAML_OUT) as f:
        print("  " + f.read().replace('\n','\n  '))

# ──────────────────────────────────────────────
# STEP 9 – REPORT & PLOTS
# ──────────────────────────────────────────────

def final_report(splits):
    print("\n" + "="*60)
    print("STEP 9: FINAL REPORT")
    print("="*60)

    split_counts = {}
    for split in ('train','val','test'):
        lbl_dir = OUT_DIR / split / 'labels'
        cls_cnt = Counter()
        if lbl_dir.exists():
            for lf in lbl_dir.iterdir():
                if lf.suffix != '.txt': continue
                with open(lf) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            cls_cnt[int(line.split()[0])] += 1
        split_counts[split] = cls_cnt
        n_imgs = len(list((OUT_DIR/split/'images').iterdir()))
        print(f"\n  [{split.upper()}]  {n_imgs} images")
        for cid in sorted(cls_cnt):
            print(f"    {CLASS_NAMES[cid]}: {cls_cnt[cid]} annotations")

    # Save JSON summary
    summary = {
        'output_dir': str(OUT_DIR),
        'yaml':       str(YAML_OUT),
        'img_size':   IMG_SIZE,
        'classes':    CLASS_NAMES,
        'splits': {
            s: {
                'images': len(list((OUT_DIR/s/'images').iterdir())),
                'annotations': {CLASS_NAMES[c]: cnt
                                for c,cnt in split_counts[s].items()}
            } for s in ('train','val','test')
        }
    }
    with open(REPORTS_DIR/'pipeline_summary.json','w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  [OK] Summary saved: {REPORTS_DIR/'pipeline_summary.json'}")

    # Class-distribution bar charts
    if MPL_OK:
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        colors = ['#3498db','#e74c3c']
        for ax, split in zip(axes, ('train','val','test')):
            cc = split_counts[split]
            cls_labels = [CLASS_NAMES[i] for i in sorted(cc)]
            vals = [cc[i] for i in sorted(cc)]
            bars = ax.bar(cls_labels, vals, color=colors[:len(vals)], edgecolor='white')
            ax.set_title(f'{split.upper()}', fontsize=13, fontweight='bold')
            ax.set_ylabel('Annotations')
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, v+2, str(v),
                        ha='center', fontsize=10, fontweight='bold')
        fig.suptitle('Class Distribution After Pipeline', fontsize=15, fontweight='bold')
        plt.tight_layout()
        plot_path = REPORTS_DIR / 'class_distribution_final.png'
        plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Plot saved: {plot_path}")

# ──────────────────────────────────────────────
# CLEANUP TEMP STAGE
# ──────────────────────────────────────────────

def cleanup_stage():
    for d in (BASE_DIR/'_stage', BASE_DIR/'_stage_aug'):
        if d.exists():
            shutil.rmtree(d)

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    print("="*60)
    print("   HELMET DETECTION – FULL PREPROCESSING PIPELINE")
    print("="*60)

    # Source map: class_id -> (class_name, source_folder)
    source_map = {
        0: ("With_Helmet",    SRC_HELMET),
        1: ("Without_Helmet", SRC_NOHELMET),
    }

    # Validate sources exist
    for cls_id, (cn, src) in source_map.items():
        if not src.exists():
            print(f"[ERROR] Source folder not found: {src}")
            sys.exit(1)

    ensure_dirs()

    # Clear previous output
    if OUT_DIR.exists():
        print(f"\n  Clearing previous output at {OUT_DIR}...")
        shutil.rmtree(OUT_DIR)
        ensure_dirs()

    # Run pipeline
    validate(source_map)
    detect_corrupted(source_map)
    remove_duplicates(source_map)
    entries, stage = collect_and_resize(source_map)
    entries = balance_and_augment(entries)
    splits  = split_dataset(entries)
    generate_yaml()
    final_report(splits)
    cleanup_stage()

    # Done
    total = sum(len(v) for v in splits.values())
    print("\n" + "="*60)
    print("  [OK] PIPELINE COMPLETE")
    print(f"  Output:     {OUT_DIR}")
    print(f"  Total imgs: {total}")
    print(f"  YAML:       {YAML_OUT}")
    print(f"  Reports:    {REPORTS_DIR}")
    print("="*60)
    print("\n  To train YOLOv8, run:")
    print("    python 03_train_yolov8.py")
    print("  or directly:")
    print(f"    yolo train model=yolov8n.pt data={YAML_OUT} imgsz={IMG_SIZE} epochs=50")


if __name__ == '__main__':
    main()
