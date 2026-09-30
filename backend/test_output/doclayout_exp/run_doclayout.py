"""Throwaway DocLayout-YOLO probe. Not wired into the app."""
from __future__ import annotations

import json
import time
from pathlib import Path

import fitz
from huggingface_hub import hf_hub_download
from doclayout_yolo import YOLOv10

OUT = Path(__file__).resolve().parent
PDF = Path(r"C:\Users\eakes\Documents\chapterwise\backend\active_doc.pdf")
DPI = 200
SCALE = DPI / 72.0
CONF = 0.2
IMGSZ = 1024

PAGES = [6, 68, 684]
CROPS = {
    "xylem_p68": Path(r"C:\Users\eakes\Documents\chapterwise\backend\test_output\p68_xref593.png"),
    "wheat_p684": Path(r"C:\Users\eakes\Documents\chapterwise\backend\test_output\p684_i0_source.png"),
}


def boxes_from_result(result) -> list[dict]:
    names = result.names
    out = []
    if result.boxes is None:
        return out
    xyxy = result.boxes.xyxy.cpu().tolist()
    cls = result.boxes.cls.cpu().tolist()
    conf = result.boxes.conf.cpu().tolist()
    for i, (box, c, score) in enumerate(zip(xyxy, cls, conf)):
        x1, y1, x2, y2 = box
        out.append(
            {
                "i": i,
                "cls": names[int(c)],
                "cls_id": int(c),
                "conf": round(float(score), 3),
                "x1": round(x1, 1),
                "y1": round(y1, 1),
                "x2": round(x2, 1),
                "y2": round(y2, 1),
                "cx": round((x1 + x2) / 2, 1),
                "cy": round((y1 + y2) / 2, 1),
                "area": round((x2 - x1) * (y2 - y1), 0),
            }
        )
    return out


def column_reading_order(dets: list[dict], page_w: float) -> list[dict]:
    """Split at page midpoint; left column top-to-bottom, then right."""
    mid = page_w / 2
    left = [d for d in dets if d["cx"] < mid]
    right = [d for d in dets if d["cx"] >= mid]
    left.sort(key=lambda d: (d["y1"], d["x1"]))
    right.sort(key=lambda d: (d["y1"], d["x1"]))
    ordered = []
    for col_name, col in (("left", left), ("right", right)):
        for rank, d in enumerate(col):
            item = dict(d)
            item["column"] = col_name
            item["col_rank"] = rank
            ordered.append(item)
    return ordered


def pymupdf_heading_pixels(page_num: int) -> list[dict]:
    """Find 1.4 / 1.4.1-ish heading lines on the page, in 200dpi pixels."""
    doc = fitz.open(PDF)
    page = doc[page_num - 1]
    hits = []
    for block in page.get_text("dict", sort=True).get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            text = "".join(span.get("text", "") for span in line.get("spans", []))
            stripped = " ".join(text.split())
            if not stripped:
                continue
            interesting = (
                stripped.startswith("1.4")
                or "EFFECT" in stripped.upper()
                or "Can Matter" in stripped
                or stripped.startswith("Activity 1.12")
            )
            if not interesting:
                continue
            x0, y0, x1, y1 = line["bbox"]
            hits.append(
                {
                    "text": stripped[:80],
                    "pdf_y0": round(y0, 1),
                    "x1": round(x0 * SCALE, 1),
                    "y1": round(y0 * SCALE, 1),
                    "x2": round(x1 * SCALE, 1),
                    "y2": round(y1 * SCALE, 1),
                    "cx": round((x0 + x1) * SCALE / 2, 1),
                    "cy": round((y0 + y1) * SCALE / 2, 1),
                }
            )
    doc.close()
    return hits


def containing_box(point_x: float, point_y: float, dets: list[dict]) -> dict | None:
    inside = [
        d
        for d in dets
        if d["x1"] <= point_x <= d["x2"] and d["y1"] <= point_y <= d["y2"]
    ]
    if not inside:
        return None
    return min(inside, key=lambda d: d["area"])


def predict_one(model, path: Path, device: str) -> dict:
    t0 = time.perf_counter()
    results = model.predict(
        str(path),
        imgsz=IMGSZ,
        conf=CONF,
        device=device,
        verbose=False,
    )
    elapsed = round(time.perf_counter() - t0, 2)
    result = results[0]
    dets = boxes_from_result(result)
    import cv2
    import numpy as np
    from PIL import Image

    annotated = result.plot(line_width=3, font_size=16)
    ann_path = OUT / f"{path.stem}_annotated.jpg"
    if isinstance(annotated, Image.Image):
        annotated.convert("RGB").save(ann_path, quality=85)
    else:
        cv2.imwrite(str(ann_path), np.asarray(annotated))
    h, w = result.orig_shape[:2] if hasattr(result, "orig_shape") else (None, None)
    return {
        "path": str(path),
        "elapsed_s": elapsed,
        "names": result.names,
        "n": len(dets),
        "dets": dets,
        "annotated": str(ann_path),
        "orig_wh": [w, h],
    }


def main() -> None:
    device = "cpu"
    print("downloading weights…")
    t0 = time.perf_counter()
    weights = hf_hub_download(
        repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
        filename="doclayout_yolo_docstructbench_imgsz1024.pt",
    )
    print(f"weights {round(time.perf_counter()-t0, 2)}s -> {weights}")
    model = YOLOv10(weights)
    print("classes", model.names if hasattr(model, "names") else "?")

    summary = {"device": device, "conf": CONF, "imgsz": IMGSZ, "pages": {}, "crops": {}}

    for page_num in PAGES:
        img = OUT / f"page_{page_num}_200dpi.png"
        print(f"\n=== PAGE {page_num} {img.name} ===")
        payload = predict_one(model, img, device)
        print(f"elapsed {payload['elapsed_s']}s detections {payload['n']} classes {payload['names']}")
        dets = payload["dets"]
        by_cls: dict[str, int] = {}
        for d in dets:
            by_cls[d["cls"]] = by_cls.get(d["cls"], 0) + 1
        print("class_counts", by_cls)
        naive = sorted(dets, key=lambda d: (d["y1"], d["x1"]))
        print("-- naive y-then-x --")
        for d in naive:
            print(
                f"  {d['cls']:16} conf={d['conf']:.2f} "
                f"({d['x1']:.0f},{d['y1']:.0f})-({d['x2']:.0f},{d['y2']:.0f})"
            )
        w = payload["orig_wh"][0] or 1600
        col_order = column_reading_order(dets, w)
        print("-- left-column then right-column --")
        for d in col_order:
            print(
                f"  [{d['column']:5} #{d['col_rank']}] {d['cls']:16} "
                f"conf={d['conf']:.2f} y1={d['y1']:.0f} cx={d['cx']:.0f}"
            )

        if page_num == 6:
            headings = pymupdf_heading_pixels(6)
            print("-- pymupdf heading lines (200dpi px) --")
            for h in headings:
                box = containing_box(h["cx"], h["cy"], dets)
                print(f"  {h['text']!r} at ({h['cx']:.0f},{h['cy']:.0f}) -> {box['cls'] if box else None} {box}")
            # Rank of boxes containing 1.4 vs 1.4.1 in both orders
            def find_heading(substr: str) -> dict | None:
                for h in headings:
                    if substr.lower() in h["text"].lower():
                        return h
                return None

            h14 = find_heading("Can Matter")
            h141 = find_heading("1.4.1")
            print("1.4 heading", h14)
            print("1.4.1 heading", h141)
            if h14 and h141:
                b14 = containing_box(h14["cx"], h14["cy"], dets)
                b141 = containing_box(h141["cx"], h141["cy"], dets)
                naive_ids = [(d["x1"], d["y1"], d["cls"]) for d in naive]
                col_ids = [(d["column"], d["col_rank"], d["cls"], d["y1"]) for d in col_order]
                print("box for 1.4", b14)
                print("box for 1.4.1", b141)
                if b14 and b141:
                    def rank(seq, box):
                        for i, d in enumerate(seq):
                            if abs(d["x1"] - box["x1"]) < 1 and abs(d["y1"] - box["y1"]) < 1:
                                return i
                        return None

                    print(
                        "naive rank 1.4 vs 1.4.1",
                        rank(naive, b14),
                        rank(naive, b141),
                        "1.4 first?" ,
                        (rank(naive, b14) or 99) < (rank(naive, b141) or 99),
                    )
                    print(
                        "column-order rank 1.4 vs 1.4.1",
                        rank(col_order, b14),
                        rank(col_order, b141),
                        "1.4 first?",
                        (rank(col_order, b14) or 99) < (rank(col_order, b141) or 99),
                    )
            payload["headings"] = headings
        payload.pop("dets")  # keep file smaller; dets already printed
        payload["class_counts"] = by_cls
        payload["dets"] = dets
        summary["pages"][str(page_num)] = payload

    for label, path in CROPS.items():
        print(f"\n=== CROP {label} {path.name} ===")
        payload = predict_one(model, path, device)
        print(f"elapsed {payload['elapsed_s']}s n={payload['n']} names={payload['names']}")
        for d in sorted(payload["dets"], key=lambda x: -x["conf"]):
            print(f"  {d['cls']:16} conf={d['conf']:.2f} area={d['area']}")
        summary["crops"][label] = payload

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nwrote", OUT / "summary.json")


if __name__ == "__main__":
    main()
