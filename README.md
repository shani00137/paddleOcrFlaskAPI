# PaddleOCR Flask API

OCR API using PaddleOCR. Accepts PDF and image files, returns all recognized raw text in a single JSON object.

## Features

- Upload one or more images (jpg, jpeg, png, bmp, tiff) or PDFs
- Returns raw recognized text as one JSON object: `{ "text": "..." }`
- Built-in Swagger UI at `/apidocs/` for testing uploads
- Health check at `/health`
- Runs on CPU or GPU automatically

## API

### `POST /ocr`

Upload a file for OCR. Use `multipart/form-data` with field key `file` (or `images` for multiple).

**Request body (multipart/form-data):**
- `file` = image or PDF file (repeat the field for multiple files)
- `images` = alternative key for multiple files

**Response 200:**
```json
{
  "text": "first line of text\nsecond line of text\n..."
}
```

**Response 400 (missing / empty file):**
```json
{
  "error": "No file found. Use form-data key 'file' or 'images'."
}
```

### `GET /health`

```json
{
  "status": "ok",
  "device": "cpu"
}
```

### Swagger UI

Open `http://localhost:5100/apidocs/` in your browser. You can upload a file directly from the interface and see the JSON result.

## Run with Docker (Windows / macOS / Linux)

The project includes a ready-to-use `Dockerfile` and `docker-compose.yml`.

### Option A - Docker Compose

```bash
docker compose up --build
```

### Option B - Docker build & run

```bash
# Build the image
docker build -t paddleocr-api .

# Run it (host port 5100 -> container port 5100)
docker run -p 5100:5100 paddleocr-api
```

Then open:
- Swagger UI: `http://localhost:5100/apidocs/`
- Health check: `http://localhost:5100/health`
- OCR endpoint: `POST http://localhost:5100/ocr`

To use a different host port, e.g. `8080`:

```bash
docker run -p 8080:5100 paddleocr-api
```

Then the app is available at `http://localhost:8080/apidocs/`.

## Local Run

```bash
pip install -r requirements.txt
python flask_api.py
```

Server runs on port `5100` by default (override with the `PORT` env var).

## Sample Test (curl)

```bash
curl -X POST http://localhost:5100/ocr \
  -F "file=@/path/to/image.png"
```

## Deployment (Render)

- `render.yaml` + `render-build.sh` are included for Render.com
- Or use the Docker image on any platform that supports containers
