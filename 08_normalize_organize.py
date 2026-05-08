# -*- coding: utf-8 -*-
"""
08_normalize_organize.py  (v3 – no temp stage)
===============================================
Input  : organized_dataset/helmet/      (images already 640x640)
         organized_dataset/without_helmet/

Output : normalized_dataset/
           train/helmet/images+labels
           train/without_helmet/images+labels
           valid/helmet/images+labels
           valid/without_helmet/images+labels
           test/helmet/images+labels
           test/without_helmet/images+labels
         normalized_data.yaml

Split  : 70% train / 20% valid / 10% test  (stratified per class)
"""

import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os, shutil, hashlib, random, json
from pathlib import Path
from collections import Counter, defaultdict

try:
    from PIL import Image, ImageEnhance, ImageFilter
    import numpy as np
except ImportError:
    print("[ERROR] pip install Pillow numpy"); sys.exit(1)

try:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MPL_OK = True
except ImportError:
    MPL_OK = False

try:
    import yaml; YAML_OK = True
except ImportError:
    YAML_OK = False

# ── CONFIG ────────────────────────────────────────────────────
BASE        = Path(r"D:\Helmet_detection")
SRC         = BASE / "organized_dataset"
OUT         = BASE / "normalized_dataset"
REPORTS     = BASE / "reports"
YAML_PATH   = BASE / "normalized_data.yaml"

CLASS_NAMES = ["helmet", "without_helmet"]
IMG_SIZE    = 640
TRAIN_R     = 0.70
VAL_R       = 0.20
# TEST_R    = 0.10  (remainder)
IMG_EXTS    = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
SEED        = 42
random.seed(SEED); np.random.seed(SEED)

# ── HELPERS ───────────────────────────────────────────────────

def img_files(d):
    d = Path(d)
    return sorted([f for f in d.iterdir()
                   if f.suffix.lower() in IMG_EXTS]) if d.exists() else []

def lbl_files(d):
    d = Path(d)
    return sorted([f for f in d.iterdir()
                   if f.suffix.lower() == '.txt']) if d.exists() else []

def md5(fp):
    h = hashlib.md5()
    with open(fp, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''): h.update(chunk)
    return h.hexdigest()

def build_dirs():
    for sp in ('train', 'valid', 'test'):
        for cls in CLASS_NAMES:
            (OUT / sp / cls / 'images').mkdir(parents=True, exist_ok=True)
            (OUT / sp / cls / 'labels').mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

# ── STEP 1: VALIDATE ─────────────────────────────────────────

def step_validate():
    print("\n" + "="*60)
    print("STEP 1: VALIDATE INPUT (organized_dataset/)")
    print("="*60)
    ok = True
    for cls in CLASS_NAMES:
        img_dir = SRC / cls / 'images'
        lbl_dir = SRC / cls / 'labels'
        if not img_dir.exists():
            print(f"  [!] MISSING: {img_dir}")
            ok = False
            continue
        imgs = img_files(img_dir)
        lbls = lbl_files(lbl_dir) if lbl_dir.exists() else []
        img_stems = {f.stem for f in imgs}
        lbl_stems = {f.stem for f in lbls}
        missing_lbl = img_stems - lbl_stems
        orphan_lbl  = lbl_stems - img_stems
        print(f"  [OK] {cls}/: {len(imgs)} images, {len(lbls)} labels")
        if missing_lbl:
            print(f"       [!] {len(missing_lbl)} images without labels (will include anyway)")
        if orphan_lbl:
            print(f"       [!] {len(orphan_lbl)} orphan labels (will skip)")
    if not ok:
        print("\n  [ERROR] Source folders missing. Run 00_full_pipeline_v2.py first.")
        sys.exit(1)

# ── STEP 2: AUDIT (CORRUPT + DUPLICATES) ─────────────────────

def step_audit():
    print("\n" + "="*60)
    print("STEP 2: DETECT CORRUPTED & DUPLICATE IMAGES")
    print("="*60)
    corrupt = set()
    hash_map = defaultdict(list)
    total = 0

    for cls in CLASS_NAMES:
        img_dir = SRC / cls / 'images'
        if not img_dir.exists(): continue
        for img in img_files(img_dir):
            total += 1
            try:
                with Image.open(img) as im: im.verify()
                with Image.open(img) as im: im.load()
                h = md5(img)
                hash_map[h].append((cls, img))
            except Exception as e:
                corrupt.add(img)
                print(f"  [X] CORRUPT: {cls}/{img.name}: {e}")

    skip = set(corrupt)
    dup_count = 0
    for h, files in hash_map.items():
        if len(files) > 1:
            for _, fp in files[1:]:
                skip.add(fp); dup_count += 1

    print(f"  Scanned : {total} images")
    print(f"  Corrupt : {len(corrupt)}")
    print(f"  Dups    : {dup_count}")
    print(f"  Usable  : {total - len(skip)}")
    return skip

# ── STEP 3: COLLECT VALID ENTRIES ────────────────────────────

def step_collect(skip_set):
    print("\n" + "="*60)
    print("STEP 3: COLLECT VALID IMAGE-LABEL PAIRS")
    print("="*60)
    # entries[cls_id] = list of (img_path, lbl_path_or_None)
    per_class = {i: [] for i in range(len(CLASS_NAMES))}

    for cls_id, cls_name in enumerate(CLASS_NAMES):
        img_dir = SRC / cls_name / 'images'
        lbl_dir = SRC / cls_name / 'labels'
        if not img_dir.exists(): continue
        for img_fp in img_files(img_dir):
            if img_fp in skip_set: continue
            lbl_fp = lbl_dir / (img_fp.stem + '.txt')
            per_class[cls_id].append((img_fp, lbl_fp if lbl_fp.exists() else None))
        print(f"  {cls_name}: {len(per_class[cls_id])} valid images")

    return per_class

# ── STEP 4: BALANCE ──────────────────────────────────────────

AUG_FNS = [
    lambda i, l: (_flip(i), _flip_lbls(l)),
    lambda i, l: (i.rotate(random.uniform(-12,12), resample=Image.BILINEAR, fillcolor=(114,114,114)), l),
    lambda i, l: (ImageEnhance.Brightness(i).enhance(random.uniform(0.6,1.4)), l),
    lambda i, l: (ImageEnhance.Contrast(i).enhance(random.uniform(0.7,1.3)), l),
    lambda i, l: (ImageEnhance.Color(i).enhance(random.uniform(0.6,1.4)), l),
    lambda i, l: (i.filter(ImageFilter.GaussianBlur(random.uniform(0.3,1.5))), l),
    lambda i, l: (_noise(i), l),
]

def _flip(img): return img.transpose(Image.FLIP_LEFT_RIGHT)
def _flip_lbls(lbls):
    out = []
    for ln in lbls:
        p = ln.strip().split()
        if len(p) >= 5:
            out.append(f"{p[0]} {1.0-float(p[1]):.6f} {p[2]} {p[3]} {p[4]}")
    return out
def _noise(img):
    a = np.array(img, dtype=np.int16)
    a = np.clip(a + np.random.randint(-20, 20, a.shape, dtype=np.int16), 0, 255).astype(np.uint8)
    return Image.fromarray(a)

def _read_lbls(lbl_fp):
    if lbl_fp is None or not Path(lbl_fp).exists(): return []
    with open(lbl_fp) as f: return [l.strip() for l in f if l.strip()]

def step_balance(per_class):
    print("\n" + "="*60)
    print("STEP 4: BALANCE CLASSES VIA AUGMENTATION")
    print("="*60)
    counts = {i: len(v) for i, v in per_class.items()}
    print("  Before: " + ", ".join(f"{CLASS_NAMES[i]}={counts[i]}" for i in sorted(counts)))

    if len(counts) < 2:
        print("  Single class - skip."); return per_class

    maj_id = max(counts, key=counts.get)
    min_id = min(counts, key=counts.get)
    deficit = counts[maj_id] - counts[min_id]

    if deficit <= 0:
        print("  Already balanced."); return per_class

    print(f"  Generating {deficit} augmented '{CLASS_NAMES[min_id]}' images...")

    AUG_OUT = OUT / '_aug_tmp'
    AUG_OUT.mkdir(parents=True, exist_ok=True)

    pool = per_class[min_id]
    generated = 0

    while generated < deficit:
        src_img, src_lbl = random.choice(pool)
        try:
            with Image.open(src_img) as im: img = im.convert('RGB')
            lbls = _read_lbls(src_lbl)
            n = random.randint(1, 3)
            for fn in random.sample(AUG_FNS, n):
                img, lbls = fn(img, lbls)
            stem = f"aug_{generated:05d}_{src_img.stem}"
            out_img = AUG_OUT / (stem + '.jpg')
            out_lbl = AUG_OUT / (stem + '.txt')
            img.save(str(out_img), 'JPEG', quality=90)
            with open(out_lbl, 'w') as f:
                f.write('\n'.join(lbls) + '\n')
            per_class[min_id].append((out_img, out_lbl))
            generated += 1
            if generated % 100 == 0:
                print(f"    {generated}/{deficit}...")
        except Exception:
            continue

    after = {i: len(v) for i, v in per_class.items()}
    print(f"  [OK] Generated {generated} images.")
    print("  After:  " + ", ".join(f"{CLASS_NAMES[i]}={after[i]}" for i in sorted(after)))
    return per_class

# ── STEP 5: SPLIT & COPY ─────────────────────────────────────

def step_split_copy(per_class):
    print("\n" + "="*60)
    print("STEP 5: STRATIFIED SPLIT 70/20/10 & COPY TO OUTPUT")
    print("="*60)
    split_log = {'train': Counter(), 'valid': Counter(), 'test': Counter()}

    for cls_id, cls_name in enumerate(CLASS_NAMES):
        entries = list(per_class[cls_id])
        random.shuffle(entries)
        n      = len(entries)
        n_tr   = int(n * TRAIN_R)
        n_val  = int(n * VAL_R)

        buckets = {
            'train': entries[:n_tr],
            'valid': entries[n_tr:n_tr + n_val],
            'test' : entries[n_tr + n_val:],
        }

        for sp, items in buckets.items():
            img_out = OUT / sp / cls_name / 'images'
            lbl_out = OUT / sp / cls_name / 'labels'
            for img_fp, lbl_fp in items:
                dst_img = img_out / img_fp.name
                shutil.copy2(str(img_fp), str(dst_img))
                if lbl_fp and Path(lbl_fp).exists():
                    shutil.copy2(str(lbl_fp), str(lbl_out / (img_fp.stem + '.txt')))
                split_log[sp][cls_name] += 1

        print(f"  {cls_name}: train={len(buckets['train'])}  valid={len(buckets['valid'])}  test={len(buckets['test'])}")

    print("\n  Split totals:")
    for sp in ('train', 'valid', 'test'):
        total = sum(split_log[sp].values())
        detail = "  ".join(f"{c}={split_log[sp][c]}" for c in CLASS_NAMES)
        print(f"    {sp}: {total} images  ({detail})")

    return split_log

# ── STEP 6: YAML ─────────────────────────────────────────────

def step_yaml():
    print("\n" + "="*60)
    print("STEP 6: GENERATE normalized_data.yaml")
    print("="*60)
    path_str = str(OUT).replace('\\', '/')
    data = {
        'path' : path_str,
        'train': [f'train/{c}/images' for c in CLASS_NAMES],
        'val'  : [f'valid/{c}/images' for c in CLASS_NAMES],
        'test' : [f'test/{c}/images'  for c in CLASS_NAMES],
        'nc'   : 2,
        'names': CLASS_NAMES,
    }
    if YAML_OK:
        import yaml
        with open(YAML_PATH, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    else:
        with open(YAML_PATH, 'w') as f:
            f.write(f"path: {path_str}\n")
            for key, sp in [('train','train'),('val','valid'),('test','test')]:
                f.write(f"{key}:\n" + "".join(f"- {sp}/{c}/images\n" for c in CLASS_NAMES))
            f.write("nc: 2\nnames:\n- helmet\n- without_helmet\n")
    print(f"  [OK] Saved: {YAML_PATH}")
    with open(YAML_PATH) as f:
        print("  " + f.read().replace('\n', '\n  '))

# ── STEP 7: REPORT ───────────────────────────────────────────

def step_report(split_log):
    print("\n" + "="*60)
    print("STEP 7: FINAL REPORT & VERIFICATION")
    print("="*60)

    summary = {'output': str(OUT), 'yaml': str(YAML_PATH),
               'img_size': IMG_SIZE, 'classes': CLASS_NAMES,
               'splits': {}}

    for sp in ('train', 'valid', 'test'):
        print(f"\n  [{sp.upper()}]")
        sp_data = {}
        for cls in CLASS_NAMES:
            img_dir = OUT / sp / cls / 'images'
            lbl_dir = OUT / sp / cls / 'labels'
            n_imgs = len(img_files(img_dir)) if img_dir.exists() else 0
            n_lbls = len(lbl_files(lbl_dir)) if lbl_dir.exists() else 0
            ann_cnt = 0
            if lbl_dir.exists():
                for lf in lbl_dir.iterdir():
                    if lf.suffix == '.txt':
                        with open(lf) as f:
                            ann_cnt += sum(1 for l in f if l.strip())
            print(f"    {cls}/: {n_imgs} images  {n_lbls} labels  {ann_cnt} annotations")
            sp_data[cls] = {'images': n_imgs, 'annotations': ann_cnt}
        summary['splits'][sp] = sp_data

    with open(REPORTS / 'normalize_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  [OK] Summary saved: {REPORTS/'normalize_summary.json'}")

    # Bar chart
    if MPL_OK:
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        colors = ['#3498db', '#e74c3c']
        for ax, sp in zip(axes, ('train', 'valid', 'test')):
            vals = [split_log[sp].get(c, 0) for c in CLASS_NAMES]
            bars = ax.bar(CLASS_NAMES, vals, color=colors, edgecolor='white', width=0.5)
            ax.set_title(sp.upper(), fontsize=13, fontweight='bold')
            ax.set_ylabel('Images')
            ax.set_ylim(0, max(vals) * 1.15 if vals else 10)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, v + 1,
                        str(v), ha='center', fontsize=10, fontweight='bold')
        fig.suptitle('normalized_dataset — Class Distribution per Split',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        pp = REPORTS / 'normalized_distribution.png'
        plt.savefig(str(pp), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Plot: {pp}")

# ── CLEANUP AUG TMP ──────────────────────────────────────────

def cleanup():
    aug_tmp = OUT / '_aug_tmp'
    if aug_tmp.exists():
        shutil.rmtree(aug_tmp, ignore_errors=True)

# ── MAIN ─────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  08_normalize_organize.py  v3")
    print("  organized_dataset/ → normalized_dataset/")
    print("=" * 60)

    if not SRC.exists():
        print(f"\n[ERROR] Source not found: {SRC}")
        print("  Run 00_full_pipeline_v2.py first.")
        sys.exit(1)

    # Clear previous output only (not source)
    if OUT.exists():
        print(f"\n  Clearing previous output: {OUT.name}/")
        shutil.rmtree(OUT, ignore_errors=True)

    build_dirs()

    step_validate()
    skip_set  = step_audit()
    per_class = step_collect(skip_set)
    per_class = step_balance(per_class)
    split_log = step_split_copy(per_class)
    step_yaml()
    step_report(split_log)
    cleanup()

    total = sum(sum(v.values()) for v in split_log.values())
    print("\n" + "=" * 60)
    print("  [OK] NORMALIZED DATASET READY")
    print(f"  Output : {OUT}")
    print(f"  YAML   : {YAML_PATH}")
    print(f"  Total  : {total} images")
    print("=" * 60)
    print("\n  Final folder structure:")
    print("    normalized_dataset/")
    for sp in ('train', 'valid', 'test'):
        for cls in CLASS_NAMES:
            n = split_log[sp].get(cls, 0)
            print(f"      {sp}/{cls}/images/  ({n} imgs)  + labels/")
    print("\n  Train YOLOv8:")
    print(f"    yolo train model=yolov8n.pt data={YAML_PATH} imgsz={IMG_SIZE} epochs=50")


if __name__ == '__main__':
    main()
