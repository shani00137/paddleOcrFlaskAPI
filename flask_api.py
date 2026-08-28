import os
import re
import time
import tempfile
from flask import Flask, request, jsonify
from werkzeug.exceptions import HTTPException
import numpy as np
import cv2

import paddle
from paddleocr import PaddleOCR
from pyzbar.pyzbar import decode

app = Flask(__name__)

# -----------------------------
# CONFIG
# -----------------------------
MAX_SIZE = 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".pdf"}

DATE_REGEX = r'''(?xi)
    \b
    (?:
        (?:0?[1-9]|[12]\d|3[01])[.\-/\s](?:0?[1-9]|1[0-2])[.\-/\s](?:19|20)\d{2}
        |
        (?:19|20)\d{2}[.\-/](?:0?[1-9]|1[0-2])[.\-/](?:0?[1-9]|[12]\d|3[01])
        |
        (?:0?[1-9]|[12]\d|3[01])[.\-/](?:0?[1-9]|1[0-2])[.\-/]\d{2}
        |
        (?:0?[1-9]|1[0-2])[/\-](?:0?[1-9]|[12]\d|3[01])[/\-](?:19|20)\d{2}
        |
        (?:0?[1-9]|[12]\d|3[01])[\s.\-/]?(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[\s.\-/]?(?:19|20)\d{2}
        |
        (?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[\s.\-/]?(?:0?[1-9]|[12]\d|3[01])[\s.\-/]?(?:19|20)\d{2}
        |
        (?:19|20)\d{2}[\s.\-/]?(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[\s.\-/]?(?:0?[1-9]|[12]\d|3[01])
        |
        (?:0[1-9]|[12]\d|3[01])(?:0[1-9]|1[0-2])(?:19|20)\d{2}
        |
        (?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])
        |
        (?:0?[1-9]|1[0-2])[.\-/]\d{2}
        |
        (?:0?[1-9]|1[0-2])[.\-/](?:19|20)\d{2}
    )
    \b
'''

# -----------------------------
# ERROR HANDLERS
# -----------------------------
@app.errorhandler(HTTPException)
def handle_http_error(e):
    return jsonify({"error": e.description}), e.code


@app.errorhandler(Exception)
def handle_error(e):
    return jsonify({"error": f"Internal error: {e}"}), 500


# -----------------------------
# DEVICE INIT (GPU if available, else CPU — never crashes startup)
# -----------------------------
def init_device():
    try:
        if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            paddle.set_device("gpu:0")
            return "gpu:0"
    except Exception as e:
        print(f"GPU init failed, falling back to CPU: {e}", flush=True)
    paddle.set_device("cpu")
    return "cpu"


device = init_device()
print(f"Using device: {device}", flush=True)

ocr = PaddleOCR(lang="en", use_textline_orientation=True)

# Best-effort warm-up — must not crash the app if it fails
try:
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    ocr.ocr(dummy)
except Exception as e:
    print(f"Warm-up skipped: {e}", flush=True)


# -----------------------------
# HELPERS
# -----------------------------
def extract_dates(text):
    return re.findall(DATE_REGEX, text)


def resize_image_np(img):
    h, w = img.shape[:2]
    max_dim = max(h, w)
    if max_dim > MAX_SIZE:
        scale = MAX_SIZE / max_dim
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def mono8_to_rgb(img):
    if img is None:
        return img
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def extract_barcodes_from_np(img_rgb):
    barcodes = []
    try:
        for obj in decode(img_rgb):
            try:
                barcodes.append(obj.data.decode("utf-8"))
            except Exception:
                pass
    except Exception:
        pass
    return barcodes


def extract_possible_barcodes(texts):
    barcodes = []
    for t in texts:
        cleaned = re.sub(r'\D', '', t)
        if len(cleaned) >= 6:
            barcodes.append(cleaned)
    return barcodes


def parse_ocr_result(result):
    texts = []
    for item in result:
        if item is None:
            continue
        if isinstance(item, dict):
            texts.extend(item.get("rec_texts", []))
        elif isinstance(item, list):
            for entry in item:
                try:
                    _, (text, _) = entry
                    texts.append(text)
                except Exception:
                    pass
    return texts


def process_image_np(img_np):
    start = time.time()
    img_np = mono8_to_rgb(img_np)
    img_np = resize_image_np(img_np)

    result = ocr.ocr(img_np)
    texts = parse_ocr_result(result)

    dates = []
    for t in texts:
        dates.extend(extract_dates(t))

    barcodes_image = extract_barcodes_from_np(img_np)
    barcodes_text = extract_possible_barcodes(texts)
    barcodes = list(set(barcodes_image + barcodes_text))

    elapsed = round(time.time() - start, 3)

    return {
        "text_count": len(texts),
        "raw_text": texts,
        "dates": dates,
        "date_count": len(dates),
        "barcodes": barcodes,
        "barcode_count": len(barcodes),
        "processing_time_sec": elapsed,
    }


def process_pdf_path(path):
    """PaddleOCR handles PDFs natively (page by page) — no cv2 needed here.
    Barcodes for PDFs are derived from OCR'd text only (no image-based
    barcode scanning, since pages aren't rasterized here)."""
    result = ocr.ocr(path)
    pages = {}
    for page_num, page_result in enumerate(result, 1):
        texts = parse_ocr_result([page_result])

        dates = []
        for t in texts:
            dates.extend(extract_dates(t))

        barcodes = list(set(extract_possible_barcodes(texts)))

        pages[f"page_{page_num}"] = {
            "text_count": len(texts),
            "raw_text": texts,
            "dates": dates,
            "date_count": len(dates),
            "barcodes": barcodes,
            "barcode_count": len(barcodes),
        }
    return pages


# -----------------------------
# FLASK ENDPOINT
# -----------------------------
@app.route("/ocr", methods=["POST"])
def ocr_endpoint():
    # Accept either field name: "file" (single) or "images" (multiple)
    if "images" in request.files:
        files = request.files.getlist("images")
    elif "file" in request.files:
        files = request.files.getlist("file")
    else:
        return jsonify({"error": "No file found. Use form-data key 'file' or 'images'."}), 400

    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "Empty filename"}), 400

    results = {}
    total_start = time.time()

    for file in files:
        suffix = os.path.splitext(file.filename)[1].lower()
        if suffix not in ALLOWED_EXTENSIONS:
            results[file.filename] = {"error": f"Unsupported file type: {suffix}"}
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file.save(tmp.name)
            path = tmp.name

        try:
            if suffix == ".pdf":
                results[file.filename] = {
                    "type": "pdf",
                    "pages": process_pdf_path(path),
                }
            else:
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    results[file.filename] = {"error": "Could not read image"}
                    continue
                results[file.filename] = {
                    "type": "image",
                    **process_image_np(img),
                }
        except Exception as e:
            results[file.filename] = {"error": str(e)}
        finally:
            os.remove(path)

    total_elapsed = time.time() - total_start

    return jsonify(
        {
            "results": results,
            "total_files": len(files),
            "total_time_sec": round(total_elapsed, 3),
        }
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": device})


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    print("CUDA available:", paddle.is_compiled_with_cuda(), flush=True)
    try:
        print("GPU count:", paddle.device.cuda.device_count(), flush=True)
    except Exception:
        print("GPU count: 0 (CPU only)", flush=True)

    # Render (and most PaaS) assign the port dynamically via $PORT
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Flask on port {port}", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)