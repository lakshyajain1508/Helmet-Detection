# -*- coding: utf-8 -*-
"""
09_augment_training.py
======================
Augments ONLY the training folders of normalized_dataset/.
Originals are NEVER modified or deleted.
Augmented copies are added alongside originals.

Target:
  normalized_dataset/train/helmet/images+labels
  normalized_dataset/train/without_helmet/images+labels

Techniques:
  - Horizontal flip
  - Rotation (bbox corners transformed)
  - Brightness / Contrast adjustment
  - Blur (Gaussian)
  - Sharpening
  - Zoom in / out
  - Gaussian noise
  - Perspective warp
  - Combined (2-3 techniques)
"""

import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import math, random, json
from pathlib import Path
from collections import Counter

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

# ── CONFIG ────────────────────────────────────────────────────
BASE        = Path(r"D:\Helmet_detection")
NORM_DIR    = BASE / "normalized_dataset"

# Actual class folder names on disk (discovered automatically)
CLASS_FOLDERS = ["With_helmet", "Without_helmet"]

REPORTS     = BASE / "reports"
IMG_SIZE    = 640
SEED        = 42

# How many augmented copies per original image
COPIES_PER_IMAGE = 3

random.seed(SEED)
np.random.seed(SEED)

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}

# ── YOLO BBOX HELPERS ─────────────────────────────────────────

def yolo_to_corners(cx, cy, w, h, img_w, img_h):
    """YOLO (cx,cy,w,h) normalised → pixel corner points [[x,y],...]"""
    cx_px = cx * img_w;  cy_px = cy * img_h
    w_px  = w  * img_w;  h_px  = h  * img_h
    x1, y1 = cx_px - w_px/2, cy_px - h_px/2
    x2, y2 = cx_px + w_px/2, cy_px + h_px/2
    return np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype=np.float32)

def corners_to_yolo(corners, img_w, img_h):
    """Pixel corner points → YOLO (cx,cy,w,h) normalised, clamped."""
    xs = corners[:,0]; ys = corners[:,1]
    x1,x2 = xs.min(), xs.max()
    y1,y2 = ys.min(), ys.max()
    cx = ((x1+x2)/2) / img_w;  cy = ((y1+y2)/2) / img_h
    bw = (x2-x1) / img_w;      bh = (y2-y1) / img_h
    cx = max(0.001, min(0.999, cx));  cy = max(0.001, min(0.999, cy))
    bw = max(0.001, min(1.0,   bw));  bh = max(0.001, min(1.0,   bh))
    return cx, cy, bw, bh

def parse_labels(lbl_path):
    """Returns list of [cls_id, cx, cy, w, h] (floats)."""
    if not lbl_path.exists(): return []
    rows = []
    with open(lbl_path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 5:
                rows.append([int(p[0])] + [float(x) for x in p[1:5]])
    return rows

def write_labels(lbl_path, rows):
    with open(lbl_path, 'w') as f:
        for r in rows:
            f.write(f"{r[0]} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f}\n")

# ── AUGMENTATION FUNCTIONS ────────────────────────────────────
# Each returns (aug_img, aug_rows) — img is PIL RGB, rows same format as parse_labels

def aug_hflip(img, rows):
    out = img.transpose(Image.FLIP_LEFT_RIGHT)
    new = []
    for r in rows:
        new.append([r[0], 1.0 - r[1], r[2], r[3], r[4]])
    return out, new


def aug_rotation(img, rows, angle=None):
    if angle is None:
        angle = random.uniform(-15, 15)
    W, H = img.size
    rad  = math.radians(-angle)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    cx_img, cy_img = W / 2, H / 2

    out = img.rotate(angle, resample=Image.BILINEAR,
                     expand=False, fillcolor=(114, 114, 114))
    new = []
    for r in rows:
        corners = yolo_to_corners(r[1], r[2], r[3], r[4], W, H)
        # Rotate corners around image centre
        rot_corners = []
        for (x, y) in corners:
            dx, dy = x - cx_img, y - cy_img
            xr = cos_a * dx - sin_a * dy + cx_img
            yr = sin_a * dx + cos_a * dy + cy_img
            rot_corners.append([xr, yr])
        rot_corners = np.array(rot_corners, dtype=np.float32)
        # Clamp to image bounds
        rot_corners[:, 0] = np.clip(rot_corners[:, 0], 0, W)
        rot_corners[:, 1] = np.clip(rot_corners[:, 1], 0, H)
        ncx, ncy, nw, nh = corners_to_yolo(rot_corners, W, H)
        new.append([r[0], ncx, ncy, nw, nh])
    return out, new


def aug_brightness(img, rows):
    f = random.uniform(0.5, 1.5)
    return ImageEnhance.Brightness(img).enhance(f), rows


def aug_contrast(img, rows):
    f = random.uniform(0.6, 1.4)
    return ImageEnhance.Contrast(img).enhance(f), rows


def aug_saturation(img, rows):
    f = random.uniform(0.5, 1.5)
    return ImageEnhance.Color(img).enhance(f), rows


def aug_blur(img, rows):
    r = random.uniform(0.5, 2.0)
    return img.filter(ImageFilter.GaussianBlur(radius=r)), rows


def aug_sharpen(img, rows):
    f = random.uniform(1.5, 3.0)
    return ImageEnhance.Sharpness(img).enhance(f), rows


def aug_noise(img, rows):
    arr = np.array(img, dtype=np.int16)
    noise = np.random.randint(-25, 25, arr.shape, dtype=np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr), rows


def aug_zoom(img, rows):
    """Zoom in (crop) or zoom out (pad) then resize back to original."""
    W, H = img.size
    scale = random.uniform(0.75, 1.25)
    new = []

    if scale >= 1.0:
        # Zoom in: crop centre
        new_w, new_h = int(W / scale), int(H / scale)
        x1 = (W - new_w) // 2;  y1 = (H - new_h) // 2
        cropped = img.crop((x1, y1, x1 + new_w, y1 + new_h))
        out = cropped.resize((W, H), Image.BILINEAR)
        for r in rows:
            corners = yolo_to_corners(r[1], r[2], r[3], r[4], W, H)
            corners[:, 0] = (corners[:, 0] - x1) * (W / new_w)
            corners[:, 1] = (corners[:, 1] - y1) * (H / new_h)
            corners[:, 0] = np.clip(corners[:, 0], 0, W)
            corners[:, 1] = np.clip(corners[:, 1], 0, H)
            ncx, ncy, nw, nh = corners_to_yolo(corners, W, H)
            new.append([r[0], ncx, ncy, nw, nh])
    else:
        # Zoom out: pad
        new_w, new_h = int(W * scale), int(H * scale)
        resized = img.resize((new_w, new_h), Image.BILINEAR)
        out = Image.new('RGB', (W, H), (114, 114, 114))
        px, py = (W - new_w) // 2, (H - new_h) // 2
        out.paste(resized, (px, py))
        for r in rows:
            corners = yolo_to_corners(r[1], r[2], r[3], r[4], W, H)
            corners[:, 0] = corners[:, 0] * scale + px
            corners[:, 1] = corners[:, 1] * scale + py
            ncx, ncy, nw, nh = corners_to_yolo(corners, W, H)
            new.append([r[0], ncx, ncy, nw, nh])

    return out, (new if new else rows)


def aug_perspective(img, rows):
    """Mild random perspective warp using PIL's PERSPECTIVE transform."""
    W, H = img.size
    jitter = random.uniform(0.03, 0.08)

    def rnd(): return random.uniform(-jitter, jitter)

    # Source corners: TL, TR, BR, BL
    src = np.float32([[0,0],[W,0],[W,H],[0,H]])
    dst = np.float32([
        [W * rnd(),       H * rnd()],
        [W * (1+rnd()),   H * rnd()],
        [W * (1+rnd()),   H * (1+rnd())],
        [W * rnd(),       H * (1+rnd())],
    ])

    # Compute 8-coefficient perspective transform
    def get_coeffs(pa, pb):
        matrix = []
        for p1, p2 in zip(pa, pb):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])
        A = np.matrix(matrix, dtype=np.float64)
        B = np.array(pb).reshape(8)
        res = np.linalg.solve(A, B)
        return np.array(res).flatten()

    try:
        coeffs = get_coeffs(src, dst)
        out = img.transform((W, H), Image.PERSPECTIVE, coeffs,
                            Image.BILINEAR, fillcolor=(114, 114, 114))
    except Exception:
        return img, rows   # fallback: return original

    def transform_point(x, y, c):
        denom = c[6]*x + c[7]*y + 1
        xn = (c[0]*x + c[1]*y + c[2]) / denom
        yn = (c[3]*x + c[4]*y + c[5]) / denom
        return xn, yn

    new = []
    for r in rows:
        corners = yolo_to_corners(r[1], r[2], r[3], r[4], W, H)
        warped = []
        for (x, y) in corners:
            xn, yn = transform_point(x, y, coeffs)
            warped.append([np.clip(xn, 0, W), np.clip(yn, 0, H)])
        warped = np.array(warped, dtype=np.float32)
        ncx, ncy, nw, nh = corners_to_yolo(warped, W, H)
        new.append([r[0], ncx, ncy, nw, nh])
    return out, new


# Pool of individual augmentations
AUG_POOL = [
    ('flip',        aug_hflip),
    ('rot',         aug_rotation),
    ('bright',      aug_brightness),
    ('contrast',    aug_contrast),
    ('sat',         aug_saturation),
    ('blur',        aug_blur),
    ('sharp',       aug_sharpen),
    ('noise',       aug_noise),
    ('zoom',        aug_zoom),
    ('persp',       aug_perspective),
]


def apply_augmentation(img, rows, n_techs=None):
    """Apply 1-3 random augmentation techniques and return (aug_img, aug_rows, tag)."""
    if n_techs is None:
        n_techs = random.randint(1, 3)
    selected = random.sample(AUG_POOL, min(n_techs, len(AUG_POOL)))
    tag_parts = []
    for name, fn in selected:
        try:
            img, rows = fn(img, rows)
            tag_parts.append(name)
        except Exception:
            pass
    return img, rows, '_'.join(tag_parts) if tag_parts else 'aug'


# ── PROCESS ONE CLASS FOLDER ──────────────────────────────────

def augment_class(cls_folder):
    img_dir = NORM_DIR / cls_folder / 'train' / 'images'
    lbl_dir = NORM_DIR / cls_folder / 'train' / 'labels'

    if not img_dir.exists():
        print(f"  [!] Not found: {img_dir}")
        return 0, 0

    orig_imgs = sorted([f for f in img_dir.iterdir()
                        if f.suffix.lower() in IMG_EXTS
                        and not f.stem.startswith('augx')])  # skip prior augmented
    n_orig = len(orig_imgs)

    print(f"\n  [{cls_folder}]  {n_orig} original images → generating {COPIES_PER_IMAGE} copies each")

    generated = 0
    skipped   = 0

    for img_fp in orig_imgs:
        lbl_fp = lbl_dir / (img_fp.stem + '.txt')
        rows   = parse_labels(lbl_fp)

        for copy_idx in range(COPIES_PER_IMAGE):
            try:
                with Image.open(img_fp) as im:
                    img = im.convert('RGB')

                aug_img, aug_rows, tag = apply_augmentation(img, [r[:] for r in rows])

                stem_new = f"augx{copy_idx}_{tag}_{img_fp.stem}"
                out_img  = img_dir / (stem_new + '.jpg')
                out_lbl  = lbl_dir / (stem_new + '.txt')

                # Never overwrite – skip if already exists
                if out_img.exists():
                    generated += 1
                    continue

                aug_img.save(str(out_img), 'JPEG', quality=92)
                write_labels(out_lbl, aug_rows)
                generated += 1

            except Exception as e:
                skipped += 1

        if (orig_imgs.index(img_fp) + 1) % 100 == 0:
            done = orig_imgs.index(img_fp) + 1
            print(f"    Progress: {done}/{n_orig} originals processed...")

    return n_orig, generated


# ── MAIN ──────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  09_augment_training.py")
    print("  Advanced augmentation on normalized_dataset/train/")
    print(f"  {COPIES_PER_IMAGE} augmented copies per original image")
    print("=" * 60)
    print(f"\n  Target : {NORM_DIR}")
    print(f"  Classes: {CLASS_FOLDERS}")
    print(f"  Techniques: flip, rotation, brightness, contrast,")
    print(f"              saturation, blur, sharpen, noise, zoom, perspective")

    if not NORM_DIR.exists():
        print(f"\n  [ERROR] Not found: {NORM_DIR}")
        print("  Run 08_normalize_organize.py first.")
        sys.exit(1)

    REPORTS.mkdir(parents=True, exist_ok=True)

    total_orig = 0; total_gen = 0
    results = {}

    for cls_folder in CLASS_FOLDERS:
        n_orig, n_gen = augment_class(cls_folder)
        total_orig += n_orig; total_gen += n_gen
        results[cls_folder] = {'originals': n_orig, 'augmented': n_gen,
                               'total': n_orig + n_gen}
        print(f"  [OK] {cls_folder}: {n_orig} original + {n_gen} augmented = {n_orig+n_gen} total")

    # Final count verification
    print("\n" + "=" * 60)
    print("  FINAL COUNT VERIFICATION")
    print("=" * 60)
    for cls_folder in CLASS_FOLDERS:
        img_dir = NORM_DIR / cls_folder / 'train' / 'images'
        lbl_dir = NORM_DIR / cls_folder / 'train' / 'labels'
        n_imgs = len([f for f in img_dir.iterdir() if f.suffix.lower() in IMG_EXTS]) if img_dir.exists() else 0
        n_lbls = len([f for f in lbl_dir.iterdir() if f.suffix == '.txt']) if lbl_dir.exists() else 0
        print(f"  {cls_folder}/train/: {n_imgs} images, {n_lbls} labels")

    # Save report
    report = {
        'target_dir'    : str(NORM_DIR),
        'copies_per_img': COPIES_PER_IMAGE,
        'techniques'    : [n for n,_ in AUG_POOL],
        'classes'       : results,
        'total_original': total_orig,
        'total_augmented': total_gen,
        'total_dataset' : total_orig + total_gen,
    }
    rp = REPORTS / 'augmentation_report.json'
    with open(rp, 'w') as f: json.dump(report, f, indent=2)
    print(f"\n  [OK] Report: {rp}")

    # Bar chart
    if MPL_OK:
        fig, ax = plt.subplots(figsize=(9, 5))
        x      = range(len(CLASS_FOLDERS))
        origs  = [results[c]['originals']  for c in CLASS_FOLDERS]
        augs   = [results[c]['augmented']  for c in CLASS_FOLDERS]
        width  = 0.35
        bars1 = ax.bar([i-width/2 for i in x], origs, width, label='Original',  color='#3498db')
        bars2 = ax.bar([i+width/2 for i in x], augs,  width, label='Augmented', color='#2ecc71')
        ax.set_xticks(list(x)); ax.set_xticklabels(CLASS_FOLDERS, fontsize=11)
        ax.set_ylabel('Images'); ax.set_title('Training Set — Original vs Augmented', fontsize=13, fontweight='bold')
        ax.legend()
        for bar in list(bars1)+list(bars2):
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+2, str(int(h)),
                    ha='center', fontsize=9, fontweight='bold')
        plt.tight_layout()
        pp = REPORTS / 'augmentation_report.png'
        plt.savefig(str(pp), dpi=150, bbox_inches='tight'); plt.close()
        print(f"  [OK] Plot : {pp}")

    print("\n" + "=" * 60)
    print("  [OK] AUGMENTATION COMPLETE")
    print(f"  Original images : {total_orig}")
    print(f"  Augmented added : {total_gen}")
    print(f"  Total training  : {total_orig + total_gen}")
    print("=" * 60)
    print("\n  Train YOLOv8:")
    yaml_p = BASE / "normalized_data.yaml"
    print(f"    yolo train model=yolov8n.pt data={yaml_p} imgsz={IMG_SIZE} epochs=50")


if __name__ == '__main__':
    main()
