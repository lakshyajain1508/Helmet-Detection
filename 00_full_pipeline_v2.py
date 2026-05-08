# -*- coding: utf-8 -*-
"""
Full Dataset Preprocessing Pipeline v2 - PRESERVES ORIGINALS
=============================================================
- Restores any files moved to quarantine by previous runs
- NEVER modifies original folders (copy-only)
- Validates, deduplicates, resizes, augments, balances
- Outputs organized_dataset/  with helmet/ and without_helmet/ class folders
- Outputs combined_dataset/   with train/ val/ test/ YOLO splits
- Generates combined_data.yaml  (nc=2, helmet / without_helmet)
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
    print("[ERROR] pip install Pillow numpy")
    sys.exit(1)

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

# ── CONFIG ──────────────────────────────────────────────────
BASE        = Path(r"D:\Helmet_detection")
SRC_HELMET  = BASE / "With Helmet"
SRC_NOWEAR  = BASE / "Without Helmet"
QUARANTINE  = BASE / "quarantine"
OUT_YOLO    = BASE / "combined_dataset"      # YOLO train/val/test splits
OUT_ORG     = BASE / "organized_dataset"     # class-sorted view
REPORTS     = BASE / "reports"
YAML_PATH   = BASE / "combined_data.yaml"

CLASS_NAMES = ["helmet", "without_helmet"]   # class 0, class 1
IMG_SIZE    = 640
TRAIN_R, VAL_R, TEST_R = 0.70, 0.20, 0.10
IMG_EXTS    = {'.jpg','.jpeg','.png','.bmp','.webp','.tiff'}
SEED        = 42
random.seed(SEED); np.random.seed(SEED)

# ── HELPERS ─────────────────────────────────────────────────

def img_files(d):
    return [f for f in Path(d).iterdir()
            if f.suffix.lower() in IMG_EXTS] if Path(d).exists() else []

def lbl_files(d):
    return [f for f in Path(d).iterdir()
            if f.suffix.lower() == '.txt'] if Path(d).exists() else []

def md5(fp):
    h = hashlib.md5()
    with open(fp,'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''): h.update(chunk)
    return h.hexdigest()

def ensure_out():
    for sp in ('train','val','test'):
        (OUT_YOLO/sp/'images').mkdir(parents=True, exist_ok=True)
        (OUT_YOLO/sp/'labels').mkdir(parents=True, exist_ok=True)
    for cls in CLASS_NAMES:
        (OUT_ORG/cls/'images').mkdir(parents=True, exist_ok=True)
        (OUT_ORG/cls/'labels').mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    QUARANTINE.mkdir(parents=True, exist_ok=True)

# ── STEP 0: RESTORE QUARANTINE ──────────────────────────────

def restore_quarantine():
    print("\n" + "="*60)
    print("STEP 0: RESTORE QUARANTINED FILES TO ORIGINALS")
    print("="*60)
    if not QUARANTINE.exists():
        print("  No quarantine folder found - nothing to restore.")
        return

    restored = 0
    for f in list(QUARANTINE.iterdir()):
        name = f.name
        # Strip 'dup_' prefix to find original name
        orig_name = name[4:] if name.startswith('dup_') else name
        stem = Path(orig_name).stem
        ext  = Path(orig_name).suffix.lower()

        # Determine which source folder it came from by checking both
        found = False
        for src_root in (SRC_HELMET, SRC_NOWEAR):
            for split in ('train','valid','test'):
                if ext in IMG_EXTS:
                    dest_dir = src_root / split / 'images'
                elif ext == '.txt':
                    dest_dir = src_root / split / 'labels'
                else:
                    continue
                if dest_dir.exists():
                    dest = dest_dir / orig_name
                    if not dest.exists():
                        try:
                            shutil.copy2(str(f), str(dest))
                            found = True
                            break
                        except Exception:
                            pass
            if found:
                f.unlink(missing_ok=True)
                restored += 1
                break

    print(f"  Restored {restored} files from quarantine.")

# ── STEP 1: VALIDATE (READ-ONLY) ────────────────────────────

def validate(source_map):
    print("\n" + "="*60)
    print("STEP 1: VALIDATE FOLDER STRUCTURE & ANNOTATIONS")
    print("="*60)
    issues = []; class_counts = Counter()
    total_imgs = total_lbls = 0

    for cls_id, (cls_name, src) in source_map.items():
        for split in ('train','valid','test'):
            img_dir = src/split/'images'
            lbl_dir = src/split/'labels'
            if not img_dir.exists(): continue
            imgs = img_files(img_dir)
            lbls = lbl_files(lbl_dir) if lbl_dir.exists() else []
            img_stems = {f.stem for f in imgs}
            lbl_stems = {f.stem for f in lbls}
            total_imgs += len(imgs); total_lbls += len(lbls)
            orphans = lbl_stems - img_stems
            missing = img_stems - lbl_stems
            if orphans: issues.append(f"  Orphan labels ({cls_name}/{split}): {len(orphans)}")
            if missing: issues.append(f"  Missing labels ({cls_name}/{split}): {len(missing)}")
            for lf in lbls:
                try:
                    with open(lf) as f:
                        for i,line in enumerate(f,1):
                            p = line.strip().split()
                            if len(p) < 5:
                                issues.append(f"  Short line {lf.name}:L{i}")
                                continue
                            class_counts[cls_id] += 1
                except Exception as e:
                    issues.append(f"  Read error {lf.name}: {e}")
            print(f"  [OK] {cls_name}/{split}: {len(imgs)} images, {len(lbls)} labels")

    print(f"\n  Total: {total_imgs} images, {total_lbls} labels")
    print(f"  Src annotations: helmet_src={class_counts[0]}, without_helmet_src={class_counts[1]}")
    if issues:
        print(f"\n  [!] {len(issues)} issues:")
        for iss in issues[:20]: print(f"    {iss}")
    else:
        print("  [OK] No structural issues.")
    return class_counts

# ── STEP 2: DETECT CORRUPTED (READ-ONLY, LOG ONLY) ──────────

def detect_corrupted(source_map):
    print("\n" + "="*60)
    print("STEP 2: DETECT CORRUPTED IMAGES (non-destructive)")
    print("="*60)
    bad = []; checked = 0
    for cls_id,(cls_name,src) in source_map.items():
        for split in ('train','valid','test'):
            img_dir = src/split/'images'
            if not img_dir.exists(): continue
            for img in img_files(img_dir):
                checked += 1
                try:
                    with Image.open(img) as im: im.verify()
                    with Image.open(img) as im: im.load()
                except Exception as e:
                    bad.append(img)
                    print(f"  [X] CORRUPT: {cls_name}/{split}/{img.name}")
    print(f"  [OK] Checked {checked} images. Corrupt: {len(bad)} (will skip in pipeline).")
    return set(bad)

# ── STEP 3: FIND DUPLICATES (READ-ONLY) ─────────────────────

def find_duplicates(source_map):
    print("\n" + "="*60)
    print("STEP 3: DETECT DUPLICATE IMAGES (non-destructive)")
    print("="*60)
    hash_map = defaultdict(list)
    for cls_id,(cls_name,src) in source_map.items():
        for split in ('train','valid','test'):
            img_dir = src/split/'images'
            if not img_dir.exists(): continue
            for img in img_files(img_dir):
                hash_map[md5(img)].append(img)
    skip = set()
    dups = 0
    for h,files in hash_map.items():
        if len(files) > 1:
            for f in files[1:]:
                skip.add(f); dups += 1
    total = sum(len(v) for v in hash_map.values())
    print(f"  Scanned {total} images. Found {dups} duplicates (will skip).")
    return skip

# ── STEP 4+5: COLLECT, REMAP, RESIZE ────────────────────────

def letterbox(img, size=IMG_SIZE):
    w,h = img.size; sc = size/max(w,h)
    nw,nh = int(w*sc), int(h*sc)
    r = img.resize((nw,nh), Image.BILINEAR)
    canvas = Image.new('RGB',(size,size),(114,114,114))
    px,py = (size-nw)//2, (size-nh)//2
    canvas.paste(r,(px,py))
    return canvas, px, py, sc, w, h

def remap_labels(raw_lines, force_cls_id, px, py, sc, orig_w, orig_h, size=IMG_SIZE):
    out = []
    for line in raw_lines:
        p = line.strip().split()
        if len(p) < 5: continue
        cx,cy,bw,bh = float(p[1]),float(p[2]),float(p[3]),float(p[4])
        cx = (cx*orig_w*sc + px)/size
        cy = (cy*orig_h*sc + py)/size
        bw = bw*orig_w*sc/size
        bh = bh*orig_h*sc/size
        cx = max(0.0,min(1.0,cx)); cy = max(0.0,min(1.0,cy))
        bw = max(0.001,min(1.0,bw)); bh = max(0.001,min(1.0,bh))
        out.append(f"{force_cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return out

def collect_resize(source_map, corrupt_set, dup_set):
    print("\n" + "="*60)
    print("STEP 4+5: COLLECT, REMAP CLASS IDs & RESIZE TO 640x640")
    print("="*60)
    STAGE = BASE / "_stage_v2"
    if STAGE.exists(): shutil.rmtree(STAGE)
    STAGE.mkdir()

    entries = []   # (img_path, lbl_path, cls_id)
    counter = Counter()
    idx = 0

    for cls_id,(cls_name,src) in source_map.items():
        for split in ('train','valid','test'):
            img_dir = src/split/'images'
            lbl_dir = src/split/'labels'
            if not img_dir.exists(): continue
            for img_fp in img_files(img_dir):
                if img_fp in corrupt_set or img_fp in dup_set:
                    continue
                lbl_fp = lbl_dir/(img_fp.stem+'.txt')
                raw = []
                if lbl_fp.exists():
                    with open(lbl_fp) as f:
                        raw = [l for l in f.read().splitlines() if l.strip()]
                try:
                    with Image.open(img_fp) as im:
                        res, px, py, sc, ow, oh = letterbox(im.convert('RGB'))
                    adj = remap_labels(raw, cls_id, px, py, sc, ow, oh)
                except Exception as e:
                    print(f"  [!] Skip {img_fp.name}: {e}")
                    continue

                name = f"c{cls_id}_{idx:05d}"
                si = STAGE/(name+'.jpg'); sl = STAGE/(name+'.txt')
                res.save(str(si),'JPEG',quality=95)
                with open(sl,'w') as f:
                    f.write('\n'.join(adj)+('\n' if adj else ''))
                entries.append((si, sl, cls_id))
                counter[cls_id] += 1; idx += 1

    for cid in sorted(counter):
        print(f"  {CLASS_NAMES[cid]} (class {cid}): {counter[cid]} images")
    print(f"  Total staged: {len(entries)}")
    return entries, STAGE

# ── AUGMENTATION ─────────────────────────────────────────────

def aug_flip(img, lbls):
    out = img.transpose(Image.FLIP_LEFT_RIGHT)
    new = []
    for ln in lbls:
        p = ln.split()
        if len(p)<5: continue
        new.append(f"{p[0]} {1.0-float(p[1]):.6f} {p[2]} {p[3]} {p[4]}")
    return out, new

def aug_rotate(img, lbls):
    ang = random.uniform(-12,12)
    return img.rotate(ang,resample=Image.BILINEAR,fillcolor=(114,114,114)), lbls

def aug_brightness(img, lbls):
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.6,1.4))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7,1.3))
    return img, lbls

def aug_blur(img, lbls):
    return img.filter(ImageFilter.GaussianBlur(random.uniform(0.3,1.5))), lbls

def aug_noise(img, lbls):
    a = np.array(img,dtype=np.int16)
    a = np.clip(a+np.random.randint(-20,20,a.shape,dtype=np.int16),0,255).astype(np.uint8)
    return Image.fromarray(a), lbls

def aug_saturation(img, lbls):
    return ImageEnhance.Color(img).enhance(random.uniform(0.6,1.4)), lbls

AUG_POOL = [aug_flip,aug_rotate,aug_brightness,aug_blur,aug_noise,aug_saturation]

def augment_one(si, sl):
    with Image.open(si) as im: img = im.convert('RGB')
    with open(sl) as f: lbls = [l.strip() for l in f if l.strip()]
    n = random.randint(1,3)
    for fn in random.sample(AUG_POOL,n):
        img,lbls = fn(img,lbls)
    return img, lbls

# ── STEP 6: BALANCE ──────────────────────────────────────────

def balance(entries, stage):
    print("\n" + "="*60)
    print("STEP 6: BALANCE CLASSES VIA AUGMENTATION")
    print("="*60)
    by_cls = defaultdict(list)
    for e in entries: by_cls[e[2]].append(e)
    counts = {c:len(v) for c,v in by_cls.items()}
    print("  Before: " + ", ".join(f"{CLASS_NAMES[c]}={v}" for c,v in sorted(counts.items())))
    if len(counts) < 2:
        print("  Only one class - skip balance."); return entries
    maj = max(counts,key=counts.get); mn = min(counts,key=counts.get)
    deficit = counts[maj] - counts[mn]
    if deficit <= 0:
        print("  Already balanced."); return entries
    print(f"  Generating {deficit} augmented '{CLASS_NAMES[mn]}' images...")
    SAUG = stage.parent/"_stage_v2_aug"
    SAUG.mkdir(exist_ok=True)
    pool = by_cls[mn]; new_entries = list(entries); gen = 0
    while gen < deficit:
        si,sl,cid = random.choice(pool)
        try:
            aug_img,aug_lbls = augment_one(si,sl)
            nm = f"aug_{gen:05d}_{si.stem}"
            oi = SAUG/(nm+'.jpg'); ol = SAUG/(nm+'.txt')
            aug_img.save(str(oi),'JPEG',quality=90)
            with open(ol,'w') as f: f.write('\n'.join(aug_lbls)+'\n')
            new_entries.append((oi,ol,cid)); gen += 1
            if gen%100==0: print(f"    {gen}/{deficit}...")
        except Exception: continue
    print(f"  [OK] Generated {gen} augmented images.")
    after = Counter(e[2] for e in new_entries)
    print("  After:  " + ", ".join(f"{CLASS_NAMES[c]}={after[c]}" for c in sorted(after)))
    return new_entries

# ── STEP 7: SPLIT & COPY ─────────────────────────────────────

def split_and_copy(entries):
    print("\n" + "="*60)
    print("STEP 7: SPLIT 70/20/10 & COPY TO OUTPUT FOLDERS")
    print("="*60)
    shuffled = list(entries); random.shuffle(shuffled)
    n = len(shuffled)
    n_tr = int(n*TRAIN_R); n_val = int(n*VAL_R)
    splits = {'train':shuffled[:n_tr],'val':shuffled[n_tr:n_tr+n_val],'test':shuffled[n_tr+n_val:]}

    # YOLO combined_dataset
    for sp,items in splits.items():
        for si,sl,cid in items:
            shutil.copy2(str(si), str(OUT_YOLO/sp/'images'/si.name))
            if sl.exists():
                shutil.copy2(str(sl), str(OUT_YOLO/sp/'labels'/(si.stem+'.txt')))
        print(f"  {sp}: {len(items)} images")

    # Organized by class (dominant annotation class)
    print("\n  Organizing class folders...")
    org_counts = Counter()
    for si,sl,cid in entries:
        cls_name = CLASS_NAMES[cid]
        shutil.copy2(str(si), str(OUT_ORG/cls_name/'images'/si.name))
        if sl.exists():
            shutil.copy2(str(sl), str(OUT_ORG/cls_name/'labels'/(si.stem+'.txt')))
        org_counts[cls_name] += 1

    for cn,cnt in org_counts.items():
        print(f"  organized_dataset/{cn}/: {cnt} images")
    return splits

# ── STEP 8: YAML ─────────────────────────────────────────────

def make_yaml():
    print("\n" + "="*60)
    print("STEP 8: GENERATE combined_data.yaml")
    print("="*60)
    path_str = str(OUT_YOLO).replace('\\','/')
    data = {'path':path_str,'train':'train/images','val':'val/images',
            'test':'test/images','nc':2,'names':CLASS_NAMES}
    if YAML_OK:
        import yaml
        with open(YAML_PATH,'w') as f: yaml.dump(data,f,default_flow_style=False,sort_keys=False)
    else:
        with open(YAML_PATH,'w') as f:
            f.write(f"path: {path_str}\ntrain: train/images\nval: val/images\n"
                    f"test: test/images\nnc: 2\nnames:\n- helmet\n- without_helmet\n")
    print(f"  [OK] Saved: {YAML_PATH}")
    with open(YAML_PATH) as f: print("  "+f.read().replace('\n','\n  '))

# ── STEP 9: REPORT ───────────────────────────────────────────

def report(splits, entries):
    print("\n" + "="*60)
    print("STEP 9: FINAL REPORT")
    print("="*60)
    split_counts = {}
    for sp in ('train','val','test'):
        lbl_dir = OUT_YOLO/sp/'labels'
        cc = Counter()
        for lf in lbl_dir.iterdir():
            if lf.suffix!='.txt': continue
            with open(lf) as f:
                for line in f:
                    p=line.strip().split()
                    if p: cc[int(p[0])]+=1
        split_counts[sp]=cc
        n = len(list((OUT_YOLO/sp/'images').iterdir()))
        print(f"\n  [{sp.upper()}] {n} images")
        for cid in sorted(cc): print(f"    {CLASS_NAMES[cid]}: {cc[cid]} annotations")

    # Org dataset summary
    print("\n  [ORGANIZED DATASET]")
    for cls in CLASS_NAMES:
        n = len(list((OUT_ORG/cls/'images').iterdir())) if (OUT_ORG/cls/'images').exists() else 0
        print(f"    {cls}/: {n} images")

    summary = {
        'output_yolo': str(OUT_YOLO), 'output_organized': str(OUT_ORG),
        'yaml': str(YAML_PATH), 'img_size': IMG_SIZE, 'classes': CLASS_NAMES,
        'originals_modified': False,
        'splits': {sp: {'images': len(list((OUT_YOLO/sp/'images').iterdir())),
                        'annotations': {CLASS_NAMES[c]:cnt for c,cnt in split_counts[sp].items()}}
                   for sp in ('train','val','test')}
    }
    with open(REPORTS/'pipeline_v2_summary.json','w') as f:
        json.dump(summary,f,indent=2)
    print(f"\n  [OK] Summary: {REPORTS/'pipeline_v2_summary.json'}")

    if MPL_OK:
        fig,axes = plt.subplots(1,3,figsize=(14,5))
        colors=['#3498db','#e74c3c']
        for ax,sp in zip(axes,('train','val','test')):
            cc=split_counts[sp]
            labs=[CLASS_NAMES[i] for i in sorted(cc)]
            vals=[cc[i] for i in sorted(cc)]
            bars=ax.bar(labs,vals,color=colors[:len(vals)],edgecolor='white')
            ax.set_title(sp.upper(),fontsize=13,fontweight='bold')
            ax.set_ylabel('Annotations')
            for bar,v in zip(bars,vals):
                ax.text(bar.get_x()+bar.get_width()/2,v+2,str(v),ha='center',fontsize=10,fontweight='bold')
        fig.suptitle('Class Distribution - Pipeline v2',fontsize=15,fontweight='bold')
        plt.tight_layout()
        pp = REPORTS/'class_distribution_v2.png'
        plt.savefig(str(pp),dpi=150,bbox_inches='tight'); plt.close()
        print(f"  [OK] Plot: {pp}")

# ── CLEANUP STAGE ────────────────────────────────────────────

def cleanup():
    for d in (BASE/'_stage_v2', BASE/'_stage_v2_aug'):
        if d.exists(): shutil.rmtree(d)

# ── MAIN ─────────────────────────────────────────────────────

def main():
    print("="*60)
    print("  HELMET DETECTION - FULL PIPELINE v2 (ORIGINALS PRESERVED)")
    print("="*60)

    source_map = {0:("With_Helmet",   SRC_HELMET),
                  1:("Without_Helmet",SRC_NOWEAR)}

    for cid,(cn,src) in source_map.items():
        if not src.exists():
            print(f"[ERROR] Not found: {src}"); sys.exit(1)

    ensure_out()

    # Clear previous outputs (NOT originals)
    for d in (OUT_YOLO, OUT_ORG):
        if d.exists():
            print(f"  Clearing {d.name}/...")
            shutil.rmtree(d)
    ensure_out()

    restore_quarantine()
    validate(source_map)
    corrupt_set = detect_corrupted(source_map)
    dup_set     = find_duplicates(source_map)
    entries, stage = collect_resize(source_map, corrupt_set, dup_set)
    entries     = balance(entries, stage)
    splits      = split_and_copy(entries)
    make_yaml()
    report(splits, entries)
    cleanup()

    total = sum(len(v) for v in splits.values())
    print("\n" + "="*60)
    print("  [OK] PIPELINE v2 COMPLETE - ORIGINALS UNTOUCHED")
    print(f"  YOLO dataset:      {OUT_YOLO}")
    print(f"  Organized folders: {OUT_ORG}")
    print(f"  Total images:      {total}")
    print(f"  YAML config:       {YAML_PATH}")
    print("="*60)
    print("\n  Next step - train YOLOv8:")
    print(f"    yolo train model=yolov8n.pt data={YAML_PATH} imgsz=640 epochs=50")


if __name__ == '__main__':
    main()
