import os
import socket
import tempfile
import time
from flask import Flask, request, jsonify
from paddleocr import PaddleOCR

app = Flask(__name__)

ocr = PaddleOCR(lang="en", use_textline_orientation=True)


def parse_result(result):
    detections = []
    texts = []
    for item in result:
        if item is None:
            continue
        for entry in item:
            try:
                box, (text, confidence) = entry
                coords = [[int(p[0]), int(p[1])] for p in box]
                detections.append({
                    "text": text,
                    "confidence": round(confidence, 4),
                    "coordinates": coords
                })
                texts.append(text)
            except Exception:
                pass
    return {"detections": detections, "text_count": len(texts), "raw_text": texts}


@app.route("/ocr", methods=["POST"])
def ocr_endpoint():
    if "file" not in request.files:
        return jsonify({"error": "No file in request"}), 400

    file = request.files["file"]
    suffix = os.path.splitext(file.filename)[1].lower()

    if suffix not in [".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]:
        return jsonify({"error": f"Unsupported file type: {suffix}"}), 400

    total_start = time.time()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        path = tmp.name

    try:
        result = ocr.ocr(path)

        if suffix == ".pdf":
            results = {}
            for page_num, page_result in enumerate(result, 1):
                results[f"page_{page_num}"] = parse_result(page_result)
            return jsonify({
                "filename": file.filename,
                "type": "pdf",
                "total_pages": len(result),
                "results": results,
                "processing_time_sec": round(time.time() - total_start, 3)
            })
        else:
            return jsonify({
                "filename": file.filename,
                "type": "image",
                "results": parse_result(result),
                "processing_time_sec": round(time.time() - total_start, 3)
            })
    finally:
        os.remove(path)


def find_free_port(preferred):
    candidates = [preferred, 8080, 8000, 8502, 9000, 5001, 3000]
    for p in candidates:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("0.0.0.0", p))
            s.close()
            return p
        except OSError:
            continue
    return preferred


if __name__ == "__main__":
    port = find_free_port(int(os.environ.get("PORT", 5010)))
    print(f"Starting Flask on port {port}", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
