import argparse
import csv
import itertools
import json
import time
from pathlib import Path

import cv2
import numpy as np

from vision.calibration import CalibrationError
from vision.config import load_config
from vision.tracker import ENHANCEMENTS, HandTracker, cumulative, preset

HIT_RADIUS = 50
METRICS = (
    ("within_20", "palm within 20 px", True),
    ("err_median", "median palm error (px)", False),
    ("detection_rate", "detection rate", True),
    ("false_pos_per_frame", "false hands per frame", False),
    ("iou", "mask IoU", True),
    ("precision", "mask precision", True),
    ("recall", "mask recall", True),
    ("jitter_px", "jitter while still (px)", False),
    ("lag_ms", "lag (ms)", False),
    ("losses_per_min", "track losses per minute", False),
    ("id_switches", "identity switches", False),
    ("ms_per_frame", "tracker ms per frame", False),
)


def variants(loo=False):
    out = [("baseline", preset("baseline"))]
    out += [(f"+E{k}", cumulative(k)) for k in range(1, len(ENHANCEMENTS) + 1)]
    if loo:
        for e in ENHANCEMENTS:
            t = preset("enhanced")
            t[e] = False
            out.append((f"-{e}", t))
    return out


def find_clips(folder, only=None):
    clips = []
    for meta_path in sorted(Path(folder).glob("*.json")):
        if meta_path.name.endswith(".labels.json"):
            continue
        meta = json.loads(meta_path.read_text())
        if only and meta["name"] not in only:
            continue
        video = meta_path.with_name(meta["video"])
        labels_path = video.with_suffix(".labels.json")
        if not labels_path.exists():
            print(f"skip {meta['name']}: no labels")
            continue
        clips.append((meta, video, json.loads(labels_path.read_text())))
    return clips


def run(cfg, meta, video, labels, toggles, hands):
    tracker = HandTracker(cfg, hands=hands, toggles=toggles)
    cap = cv2.VideoCapture(str(video))
    size = tuple(meta["size"])
    start = 0
    if meta.get("calib"):
        c0, c1 = meta["calib"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, c0)
        for _ in range(c1 - c0):
            ok, frame = cap.read()
            if not ok:
                break
            tracker.add_calibration_frame(frame)
        tracker.finish_calibration(size, tuple(meta["roi"]) if meta.get("roi") else None)
        start = c1
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    poly_frames = {int(k) for k, e in labels["frames"].items() if e.get("polys")}
    fps = meta["fps"]
    records, masks = {}, {}
    busy, i = 0.0, start
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        states = tracker.process(frame, i / fps)
        busy += time.perf_counter() - t0
        records[i] = [(s.id, s.palm, s.raw, s.found) for s in states]
        if i in poly_frames:
            masks[i] = cv2.resize(tracker.hand_mask, size, interpolation=cv2.INTER_NEAREST) > 0
        i += 1
    cap.release()
    n = max(len(records), 1)
    perf = {"ms_per_frame": 1000 * busy / n}
    perf.update({f"ms_{k}": v for k, v in tracker.timings.items() if k != "total"})
    return records, masks, perf


def dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def match(gt, preds, radius=HIT_RADIUS):
    best_key, best = None, []
    for combo in itertools.product([None] + list(range(len(preds))), repeat=len(gt)):
        used = [c for c in combo if c is not None]
        if len(used) != len(set(used)):
            continue
        pairs = [(g, c) for g, c in enumerate(combo) if c is not None]
        if any(dist(gt[g], preds[c]) > radius for g, c in pairs):
            continue
        key = (len(pairs), -sum(dist(gt[g], preds[c]) for g, c in pairs))
        if best_key is None or key > best_key:
            best_key, best = key, pairs
    return best


def lag_ms(records, fps, max_lag=10):
    series = {}
    for i in sorted(records):
        for sid, palm, raw, found in records[i]:
            if found and raw is not None:
                series.setdefault(sid, []).append((i, palm, raw))
    best = []
    for seq in series.values():
        run = [seq[0]]
        for prev, cur in zip(seq, seq[1:]):
            if cur[0] == prev[0] + 1:
                run.append(cur)
            else:
                best = max(best, run, key=len)
                run = [cur]
        best = max(best, run, key=len)
    if len(best) < 30:
        return float("nan")
    out = np.diff(np.array([p for _, p, _ in best]), axis=0)
    raw = np.diff(np.array([r for _, _, r in best]), axis=0)
    scores = []
    for k in range(max_lag + 1):
        a, b = raw[:len(raw) - k], out[k:]
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        scores.append(float((a * b).sum() / denom) if denom > 0 else 0.0)
    return 1000.0 * int(np.argmax(scores)) / fps


def metrics(meta, labels, records, masks):
    fps = meta["fps"]
    errors, n_gt, fp, idsw, n_frames = [], 0, 0, 0, 0
    last_id = {}
    for key in sorted(labels["frames"], key=int):
        i = int(key)
        if i not in records:
            continue
        e = labels["frames"][key]
        n_frames += 1
        preds = [(sid, palm) for sid, palm, _, _ in records[i]]
        gt = [(name, e[name]) for name in ("A", "B") if e.get(name)]
        n_gt += len(gt)
        pairs = match([g[1] for g in gt], [p[1] for p in preds])
        for g, c in pairs:
            errors.append(dist(gt[g][1], preds[c][1]))
            name, sid = gt[g][0], preds[c][0]
            if name in last_id and last_id[name] != sid:
                idsw += 1
            last_id[name] = sid
        fp += len(preds) - len(pairs)
    errs = np.array(errors)
    nan = float("nan")
    res = {
        "labeled_hands": n_gt,
        "detection_rate": len(errs) / n_gt if n_gt else nan,
        "err_mean": float(errs.mean()) if len(errs) else nan,
        "err_median": float(np.median(errs)) if len(errs) else nan,
        "within_20": float((errs <= 20).sum() / n_gt) if n_gt else nan,
        "false_pos_per_frame": fp / n_frames if n_frames else nan,
        "id_switches": idsw,
    }

    losses, prev = 0, None
    for i in sorted(records):
        found = sum(1 for r in records[i] if r[3])
        if prev is not None and found < prev:
            losses += 1
        prev = found
    minutes = len(records) / fps / 60 if records else 0
    res["losses_per_min"] = losses / minutes if minutes else nan

    res["jitter_px"] = nan
    if meta.get("still"):
        pts, last = [], None
        for i in range(*meta["still"]):
            if records.get(i):
                palms = [r[1] for r in records[i]]
                p = palms[0] if last is None else min(palms, key=lambda q: dist(q, last))
                pts.append(p)
                last = p
        if len(pts) > 5:
            pts = np.array(pts)
            res["jitter_px"] = float(np.sqrt(((pts - pts.mean(axis=0)) ** 2).sum(axis=1).mean()))
    res["lag_ms"] = lag_ms(records, fps)

    ious, precs, recs = [], [], []
    w, h = meta["size"]
    for key, e in labels["frames"].items():
        i = int(key)
        if not e.get("polys") or i not in masks:
            continue
        gt = np.zeros((h, w), np.uint8)
        for poly in e["polys"]:
            cv2.fillPoly(gt, [np.array(poly, np.int32)], 1)
        gt = gt > 0
        pred = masks[i]
        inter = np.logical_and(gt, pred).sum()
        union = np.logical_or(gt, pred).sum()
        ious.append(inter / union if union else 1.0)
        precs.append(inter / pred.sum() if pred.sum() else float(gt.sum() == 0))
        recs.append(inter / gt.sum() if gt.sum() else 1.0)
    res["iou"] = float(np.mean(ious)) if ious else nan
    res["precision"] = float(np.mean(precs)) if precs else nan
    res["recall"] = float(np.mean(recs)) if recs else nan
    return res


def fmt(value, pct):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    return f"{100 * value:.1f}%" if pct else f"{value:.1f}"


def write_outputs(rows, out_dir, order):
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out_dir / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    clips = list(dict.fromkeys(r["clip"] for r in rows))
    table = {(r["clip"], r["variant"]): r for r in rows}
    lines = ["# Ablation", ""]
    for key, title, pct in METRICS:
        lines += [f"## {title}", "", "| variant | " + " | ".join(clips) + " | mean |",
                  "|---|" + "---|" * (len(clips) + 1)]
        for v in order:
            vals = [table.get((c, v), {}).get(key) for c in clips]
            nums = [x for x in vals if isinstance(x, (int, float)) and not np.isnan(x)]
            mean = float(np.mean(nums)) if nums else float("nan")
            lines.append(f"| {v} | " + " | ".join(fmt(x, pct) for x in vals) + f" | {fmt(mean, pct)} |")
        lines.append("")
    (out_dir / "ablation.md").write_text("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    for key, title, pct in METRICS[:6]:
        means = []
        for v in order:
            nums = [table[(c, v)][key] for c in clips if (c, v) in table
                    and isinstance(table[(c, v)].get(key), (int, float)) and not np.isnan(table[(c, v)][key])]
            means.append(np.mean(nums) * (100 if pct else 1) if nums else 0)
        fig, ax = plt.subplots(figsize=(8, 3.5))
        bars = ax.bar(order, means, width=0.5, color="#2a78d6", zorder=2)
        ax.bar_label(bars, fmt="%.0f" if pct else "%.1f", padding=2, fontsize=8, color="#52514e")
        ax.set_title(title + (" (%)" if pct else ""), loc="left", color="#0b0b0b")
        ax.grid(axis="y", color="#e4e3df", linewidth=0.8, zorder=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#b5b4ae")
        ax.tick_params(axis="x", rotation=45, colors="#52514e")
        ax.tick_params(axis="y", length=0, colors="#52514e")
        fig.tight_layout()
        fig.savefig(out_dir / f"{key}.png", dpi=120)
        plt.close(fig)


def summarize_practice(path, out_dir):
    if not Path(path).exists():
        return
    with open(path) as f:
        rows = list(csv.DictReader(f))
    groups = {}
    for r in rows:
        groups.setdefault(r["tracker"], []).append(r)
    lines = ["# Practice mode", "", "| tracker | sessions | hits/min | drops/min | longest streak |", "|---|---|---|---|---|"]
    for tracker, rs in groups.items():
        mean = lambda k: np.mean([float(r[k]) for r in rs])
        lines.append(f"| {tracker} | {len(rs)} | {mean('hits_per_min'):.1f} | {mean('drops_per_min'):.1f} | "
                     f"{mean('longest_streak'):.1f} |")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "practice.md").write_text("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run baseline vs enhanced ablation on labelled clips")
    ap.add_argument("--clips", default="clips")
    ap.add_argument("--out", default="eval/results")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--loo", action="store_true", help="also run leave-one-out variants")
    ap.add_argument("--hands", type=int, choices=(1, 2))
    ap.add_argument("--practice-log", default="practice_log.csv")
    ap.add_argument("--anchor", choices=("palm", "top"), default="palm",
                    help="tracked point; labels mark palm centres, so palm is the default")
    ap.add_argument("--config")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    cfg["vision"]["cursor_anchor"] = args.anchor
    order = [name for name, _ in variants(args.loo)]
    rows = []
    for meta, video, labels in find_clips(args.clips, args.only):
        hands = args.hands or (2 if any(e.get("B") for e in labels["frames"].values()) else 1)
        for name, toggles in variants(args.loo):
            try:
                records, masks, perf = run(cfg, meta, video, labels, toggles, hands)
            except CalibrationError as e:
                print(f"{meta['name']:>14} {name:>9}  calibration failed: {e}")
                rows.append({"clip": meta["name"], "variant": name, "hands": hands, "error": str(e)})
                continue
            m = metrics(meta, labels, records, masks)
            rows.append({"clip": meta["name"], "variant": name, "hands": hands, **m, **perf})
            print(f"{meta['name']:>14} {name:>9}  within20 {fmt(m['within_20'], True):>6}  "
                  f"median {fmt(m['err_median'], False):>5}px  IoU {fmt(m['iou'], True):>6}  "
                  f"{perf['ms_per_frame']:.1f} ms")
    out_dir = Path(args.out)
    if rows:
        write_outputs(rows, out_dir, order)
        print(f"wrote {out_dir / 'results.csv'} and {out_dir / 'ablation.md'}")
    summarize_practice(args.practice_log, out_dir)


if __name__ == "__main__":
    main()
