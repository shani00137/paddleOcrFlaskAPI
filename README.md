# PaddleOCR Flask API

OCR API using PaddleOCR - accepts PDF and image files, returns detected text with coordinates.

## API

```
POST /ocr
Content-Type: multipart/form-data
Body: file (pdf, jpg, jpeg, png, bmp, tiff)
```

### Response

```json
{
  "filename": "doc.pdf",
  "type": "pdf",
  "total_pages": 1,
  "results": {
    "page_1": {
      "detections": [
        {
          "text": "Hello World",
          "confidence": 0.9876,
          "coordinates": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        }
      ],
      "text_count": 1,
      "raw_text": ["Hello World"]
    }
  },
  "processing_time_sec": 1.234
}
```

## Local Run

```bash
pip install -r requirements.txt
python flask_api.py
```

Server runs on `http://0.0.0.0:5000`
