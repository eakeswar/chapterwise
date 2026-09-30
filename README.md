# Chapterwise

Internship project at **ADS Softek**. Upload a textbook PDF and get an **ordered topic tree** (chapter → section → subsection), a grounded explanation, topic-scoped Q&A, page images, and text-to-speech.

**GitHub:** https://github.com/eakeswar/chapterwise  
**Internship report:** [docs/Internship_Progress_Report_Chapterwise.md](docs/Internship_Progress_Report_Chapterwise.md) (also `.docx` / `.pdf`)

This repo does **not** include `backend/.env` (API keys). After you clone, you must create that file again.

---

## What you need

| Requirement | Notes |
|-------------|--------|
| Windows, macOS, or Linux | Developed on Windows |
| **Python 3.11+** | Backend |
| **Node.js 18+** (npm) | Frontend |
| Azure OpenAI / Foundry | Chat + TTS + image edit (recommended laptop path) |
| Optional: Kaggle account + GPU notebook | Only if you want remote extract or Llama chat |

Two terminals: backend on **port 8766**, frontend on **port 5173**.

---

## 1. Clone

```powershell
git clone https://github.com/eakeswar/chapterwise.git
cd chapterwise
```

---

## 2. Backend setup

### Install Python packages (into `backend/vendor`, not global site-packages)

From the **repo root**:

```powershell
python -m pip install --target backend/vendor -r backend/requirements.txt
```

`backend/main.py` adds `backend/vendor` to `sys.path` automatically.

### Create `backend/.env`

Copy the example, then fill in real keys (never commit this file):

```powershell
copy backend\.env.example backend\.env
```

On macOS/Linux: `cp backend/.env.example backend/.env`

### Recommended laptop defaults (what we used at the end)

These match the working internship setup. Kaggle is **optional**; leave extract and topics **local**.

```env
# --- Chat (your Azure resource, e.g. chapterwise-text) ---
# Base URL MUST include /openai/v1 or chat returns 404.
TEXT_PROVIDER=azure
AZURE_CHAT_API_KEY=...
AZURE_CHAT_BASE_URL=https://YOUR-RESOURCE.openai.azure.com/openai/v1
AZURE_TEXT_DEPLOYMENT=gpt-4.1-mini
CHAPTERWISE_LLM_TIMEOUT=600

# --- TTS + images (separate key / services endpoint) ---
# Do not reuse the TTS/image key for chat.
AZURE_OPENAI_API_KEY=...
AZURE_SERVICES_BASE_URL=https://YOUR-SERVICES-HOST.services.ai.azure.com/openai/v1
AZURE_TTS_BASE_URL=https://YOUR-SERVICES-HOST.services.ai.azure.com/openai/v1
AZURE_TTS_DEPLOYMENT=gpt-4o-mini-tts
AZURE_IMAGES_BASE_URL=https://YOUR-SERVICES-HOST.services.ai.azure.com/openai/v1
AZURE_IMAGE_DEPLOYMENT=gpt-image-2

# --- Local extract + regex TOC ---
EXTRACT_PROVIDER=local
CHAPTERWISE_TOPIC_BUILDER=regex
CHAPTERWISE_TEXT_EXTRACTOR=rapidocr
CHAPTERWISE_RAPIDOCR_FORCE=0
CHAPTERWISE_DEFER_IMAGE_B64=1
CHAPTERWISE_EXTRACT_WORKERS=16
CHAPTERWISE_MAX_PDF_BYTES=524288000
CHAPTERWISE_CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
CHAPTERWISE_TOPIC_CHUNK_PAGES=4
CHAPTERWISE_DEBUG_TIMING=0

BLUR_VARIANCE_THRESHOLD=100
SMALL_IMAGE_MIN_DIM=400
```

`CHAPTERWISE_TOPIC_BUILDER` is not always listed in `.env.example`; if it is missing, add `regex` yourself. Code default is already `regex`.

**Azure rules that bit us:**

- Use the **deployment name** as `AZURE_TEXT_DEPLOYMENT`, not the marketing model name.
- Chat URL must end with `/openai/v1`.
- TTS and images used a **different** Foundry/services resource than chat.
- Restart the backend after every `.env` change (in-memory extract/topics/explanations reset).

Full template of other variables: `backend/.env.example`.

### Start the backend

```powershell
python backend/main.py
```

Check: http://127.0.0.1:8766/health  

Smoke tests (after keys are set):

- http://127.0.0.1:8766/debug/text — chat
- http://127.0.0.1:8766/debug/azure — TTS and images

If port 8766 is in use, stop the old `python backend/main.py` (Ctrl+C) and start again.

---

## 3. Frontend setup

```powershell
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173**.

The API base is only hardcoded in `frontend/src/config/api.js` (`http://127.0.0.1:8766`). Override with `frontend/.env`:

```env
VITE_API_BASE_URL=http://127.0.0.1:8766
```

Restart Vite after changing that.

---

## 4. How to use the app

1. Upload a PDF. Optional: **Test first pages only** — limits **extract** page range (the browser still sends the whole file).
2. Wait for extract + topic build. Timing is shown on the loading screen.
3. Click a topic in the sidebar (document order; do not expect client-side sort).
4. Explanation streams (SSE). **Listen** uses Azure TTS (default voice `shimmer`; picker stays across topic switches).
5. **Ask** is the floating button — answers only from that topic; otherwise it says the section does not have enough information.
6. Gallery skips page backgrounds. **Generate HD** appears only for **decorative** images after classify is ready. Diagrams/labels stay informational (Lanczos only).

State is **in-memory**. A new upload or a backend restart clears extract, topics, explanations, and image caches.

---

## 5. How the pipeline works (so you remember)

1. **Extract (CPU):** PyMuPDF images + RapidOCR/pdfplumber text. Default: RapidOCR, but digital pages can use native `get_text` when `CHAPTERWISE_RAPIDOCR_FORCE=0`. Header/footer detection samples **36 pages** and caches by file fingerprint.
2. **Topics:** Regex headings (`Chapter N` at **line start**, `N.N`, `N.N.N`, `Activity N.N`). Body is kept via page ranges. Set `CHAPTERWISE_TOPIC_BUILDER=llm` only if you want the old Llama/Azure JSON tree (slow/unreliable on 1B).
3. **Explain / Ask:** Azure chat (`TEXT_PROVIDER=azure`) or Kaggle Llama. SSE; cache explanation only when the stream finishes.
4. **Images:** Encode on `/topic` (deferred b64). Classify informational vs decorative; baked text via RapidOCR on the crop. HD uses `gpt-image-2` **`images.edit`** with the source photo.

Test book we used: **NCERT Class 9 Science**, ~301 MB, 1,299 pages. Full local extract was on the order of **3.5 minutes** after defer-b64; regex TOC ~**0.16 s** (~223 topics).

---

## 6. Important endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Providers + health |
| POST | `/upload_pdf` | Save PDF + extract (`?page_start=&page_end=`) |
| POST | `/build_topics` | Build tree |
| GET | `/topic/{id}` | Topic payload (no LLM) |
| GET | `/topic/{id}/explanation/stream` | SSE explanation |
| POST | `/ask` | SSE Q&A |
| POST | `/tts` | MP3 |
| POST | `/topic/{id}/images/{image_id}/generate` | Decorative HD only |
| GET | `/debug/extraction` | Extract summary |
| GET | `/debug/topics` | Full tree |

---

## 7. Optional: Kaggle (GPU Llama / remote extract)

Laptop default should stay `EXTRACT_PROVIDER=local`. Use Kaggle only if you need Llama or offloaded extract.

See **[backend/kaggle/README.md](backend/kaggle/README.md)** for publish + notebook steps.

Short version:

1. Publish dataset: `backend\kaggle\publish_chapterwise_dataset.bat` (`KAGGLE_USERNAME`, `KAGGLE_API_TOKEN`, `KAGGLE_DATASET_SLUG=chapterwise-models`).
2. Notebook: GPU T4 x2, Internet on, attach dataset. Load **Kaggle secrets** named `KAGGLE_API_SECRET` and `HF_TOKEN` (print present/length only).
3. Accept the gated Llama licence on Hugging Face for `meta-llama/Llama-3.2-1B-Instruct`.
4. Copy the new `trycloudflare.com` URL into `KAGGLE_API_BASE_URL` every session (URL dies when the notebook stops).
5. Laptop `KAGGLE_API_SECRET` must **exactly** match the notebook secret.
6. Chat uses async `/chat` + poll (sync completions hit Cloudflare **524**). Busy chat is **409**. Dead tunnel must fail fast so `/topic` still returns source text.

Kaggle is **not** required to run regex TOC or local extract. Do not move extract/TOC to Kaggle just to “use the GPUs.”

---

## 8. Known limits (v1)

- OCR title spill and two-column reading order can scramble heading order.
- Some PDFs still produce extra chapter nodes from repeated lesson text.
- RapidOCR extract stores `overlapWordCount=0`; labelled diagrams use **image-pixel OCR**, not PDF overlap.
- No disk cache, no multi-user auth, no vector DB.
- Free Cloudflare tunnels are ephemeral.

---

## 9. If something is missing after a fresh clone

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError` | Re-run pip `--target backend/vendor` |
| Chat 404 | Chat base URL missing `/openai/v1` |
| Chat `DeploymentNotFound` | Wrong deployment name or chat not deployed on that resource |
| Topic open hangs ~10 minutes | `TEXT_PROVIDER=kaggle` and a dead tunnel — set `TEXT_PROVIDER=azure` or fix the tunnel |
| Frontend cannot reach API | Backend not on 8766, or CORS / `VITE_API_BASE_URL` |
| Black images | Already fixed in code (SMask → Pixmap PNG); pull latest `main` |
| Port in use | Kill the old backend process |

---

## 10. Project layout (high level)

```
chapterwise/
  README.md                 ← this file
  docs/                     ← internship report
  backend/
    main.py                 ← FastAPI, port 8766
    .env                    ← your secrets (gitignored)
    .env.example
    pdf_extract.py
    topic_builder.py
    topic_detail.py
    qa.py
    tts.py
    image_enhance.py
    text_provider.py
    azure_clients.py
    kaggle_client.py
    kaggle/                 ← notebook + dataset publish
    vendor/                 ← pip --target (gitignored)
  frontend/
    src/config/api.js       ← only place for API host/port
```

---

*Last updated for the public repo at https://github.com/eakeswar/chapterwise. Secrets stay in your own `backend/.env`.*
