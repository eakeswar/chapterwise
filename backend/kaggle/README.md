# chapterwise on Kaggle — PDF extraction

Offload slow PDF parsing to a Kaggle notebook so your laptop stays responsive.

## 1. Publish the code bundle (laptop, once per code change)

Add to `backend/.env`:

```env
KAGGLE_USERNAME=your_username
KAGGLE_API_TOKEN=KGAT_...
KAGGLE_DATASET_SLUG=chapterwise-models
```

From repo root:

```powershell
backend\kaggle\publish_chapterwise_dataset.bat
```

This creates `chapterwise-models-bundle/` with:

- `security.py`
- `pdf_extract.py`
- `chapterwise_gpu_server.py`
- `llm_server.py`

No large model files — Llama weights download on first chat request on Kaggle.

## 2. Create / open the Kaggle notebook

1. Upload or create a notebook from `chapterwise_kaggle.ipynb`
2. **Settings:** Accelerator **GPU T4 x2**, **Internet ON**
3. **Add Data** → your dataset **`chapterwise-models`** (latest version)

## 3. Run the notebook

**Cell 1** — copies modules from the dataset into `/kaggle/working`

**Cell 2** — set a strong secret (same value you will use on laptop):

```python
os.environ["KAGGLE_API_SECRET"] = "your-strong-secret-here"
os.environ.setdefault("CHAPTERWISE_EXTRACT_WORKERS", "4")
```

Run the server cell. Copy the printed `trycloudflare.com` URL.

## 4. Configure laptop `backend/.env`

```env
KAGGLE_ENABLED=true
KAGGLE_API_BASE_URL=https://xxxx.trycloudflare.com
KAGGLE_API_SECRET=your-strong-secret-here
EXTRACT_PROVIDER=kaggle
TEXT_PROVIDER=kaggle
CHAPTERWISE_LLM_TIMEOUT=600
KAGGLE_EXTRACT_TIMEOUT=900
```

**Kaggle notebook (cell 2)** — for gated Llama models, set `HF_TOKEN` (Add-ons → Secrets):

```python
os.environ["HF_TOKEN"] = "hf_..."  # or use Kaggle Secrets
os.environ.setdefault("CHAPTERWISE_LLM_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
```

Restart backend: `python backend/main.py`

## 5. Upload from the app

When `EXTRACT_PROVIDER=kaggle`:

1. Browser sends PDF to laptop backend
2. Laptop saves `active_doc.pdf` and forwards PDF to Kaggle
3. Kaggle runs parallel `pdf_extract.py` on its CPUs
4. Extraction JSON returns to laptop → topic build continues as before

Use `EXTRACT_PROVIDER=auto` to fall back to local extraction if Kaggle is offline.

## Endpoints (Kaggle server)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Status + worker count + LLM status |
| POST | `/chat/completions` | Llama chat for topic extraction (`messages`, `temperature`) |
| POST | `/upload_pdf` | Save PDF on Kaggle (small files) |
| POST | `/upload_chunk` | Chunked PDF upload for large textbooks |
| POST | `/extract` | Start async extraction (returns immediately) |
| GET | `/extract/status` | Poll extraction progress |
| GET | `/extract/result` | Fetch full extraction JSON when complete |
| POST | `/upload_and_extract` | Single-shot upload + extract (small files only) |

Auth: `Authorization: Bearer <KAGGLE_API_SECRET>`

## Notes

- Parsing uses **CPU threads** on Kaggle (PyMuPDF/pdfplumber), not the GPU — but Kaggle frees your laptop and has stable I/O.
- Large textbooks (300 MB) use **chunked upload** (`/upload_chunk`, 4 MB parts) so cloudflared tunnels stay stable.
- Extraction runs **async** — laptop polls `/extract/status` then fetches `/extract/result` (avoids Cloudflare 524 timeouts).
- Increase `KAGGLE_EXTRACT_TIMEOUT` if extraction itself times out.
- Stop the Kaggle session when done — GPU hours accrue while the notebook runs.
