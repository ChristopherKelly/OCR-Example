# File: OCR_Example.py
# Description: Standalone single-file document OCR tool.
#              Reads a PDF or DOCX, sends it to an LLM (Gemini Vertex AI) 
#              with a configurable system prompt.
# Usage:
#   Single document
#   python OCR_Example.py <path_to_document>
#   
#   Multiple Documents
#   python OCR_Example.py --multi <folder> [--output CSV]
#
#   Send as a batch, will queue and take longer but you get a 50% discount on costs.
#   python OCR_Example.py --batch <folder> [--output CSV] [--keep-gcs  if the Cloud bucket should keep the data]
# 
#
# Environment variables:
#   VERTEX_PROJECT_ID_DFB_OCR   - GCP project ID (Gemini + Claude via Vertex AI)
#   VERTEX_CREDENTIALS_DFB_OCR  - Path to service account JSON key file
#   VERTEX_BUCKET_DFB_OCR       - ID of the GCP Bucket (used by --batch)



import base64
import csv
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Optional PDF / Word support
try:
    from pypdf import PdfReader
    _PYPDF_AVAILABLE = True
except ImportError:
    _PYPDF_AVAILABLE = False

try:
    import fitz  # PyMuPDF — text extraction + page rendering
    _FITZ_AVAILABLE = True
except ImportError:
    _FITZ_AVAILABLE = False

try:
    from docx import Document as DocxDocument
    _DOCX_AVAILABLE = True
except ImportError:
    _DOCX_AVAILABLE = False


# ============================================================================
# CONFIGURATION — edit these values or set via environment variables
# ============================================================================

VERTEX_CREDENTIALS_DFB_OCR_PATH = os.getenv("VERTEX_CREDENTIALS_DFB_OCR")
VERTEX_PROJECT_ID_DFB_OCR = os.getenv("VERTEX_PROJECT_ID_DFB_OCR")
VERTEX_BUCKET_DFB_OCR = os.getenv("VERTEX_BUCKET_DFB_OCR")

# Maps model name -> Vertex AI endpoint region.
# Each model gets its own genai.Client pointed at the right location.
GEMINI_MODELS = {
    "gemini-2.5-flash-lite": "europe-west1",
    "gemini-2.5-flash": "europe-west1",
    "gemini-2.5-pro":   "europe-west1",
    "gemini-3.1-flash-lite": "eu",
    "gemini-3.5-flash": "eu",
}

# Optional per-model image-input resolution: "low" | "medium" | "high".
# Gemini bills image inputs by tile count, so this is the main cost/accuracy dial
# for document OCR. A model omitted here uses its family DEFAULT.
# Measured on a 14-page scanned form (Blooms):
#   - Gemini 2.5 multi-image default = MEDIUM (~8.9k input tokens). "low" drops it
#     to ~6.3k but loses accuracy (misread a handwritten digit). "high" is REJECTED
#     by 2.5 for multi-image requests (400: HIGH is single-image only) — don't set it.
#   - Gemini 3.x defaults high (~20k input tokens) and reads handwriting/checkboxes
#     best; set "low"/"medium" to claw back cost at some accuracy.
GEMINI_MEDIA_RESOLUTION: dict = {
    # "gemini-3.5-flash": "medium",   # cheaper 3.x
    # "gemini-2.5-flash": "low",      # cheapest 2.5 (accuracy trade)
}

_MEDIA_RESOLUTION_ENUM = {
    "low": "MEDIA_RESOLUTION_LOW",
    "medium": "MEDIA_RESOLUTION_MEDIUM",
    "high": "MEDIA_RESOLUTION_HIGH",
}

# Provider registry: maps provider name -> (key, model_list)
# For Gemini, "key" is the Vertex credentials path (not an API key).
PROVIDERS = {
    "gemini":  {"key": VERTEX_CREDENTIALS_DFB_OCR_PATH, "models": list(GEMINI_MODELS.keys())}
}

# Pricing per 1M tokens (paid tier): {model_name: (input_$/1M, output_$/1M)}
# Batch pricing is 50% of standard (https://ai.google.dev/gemini-api/docs/batch-api)
USD_TO_EUR = 0.92

MODEL_PRICING = {
    # Gemini — standard (interactive) — EU available only
    "gemini-3.5-flash":                      (1.50,    9.00),
    "gemini-3.1-pro-preview":                (2.00,    12.00),  
    "gemini-3.1-flash-image":                (0.50,    3.00),  
    "gemini-3.1-flash-lite":                 (0.075,   0.30),
    "gemini-3-flash-preview":                (0.50,    3.00),  
    "gemini-2.5-pro":                        (1.25,   10.00),
    "gemini-2.5-flash":                      (0.30,    2.50),
    # Gemini — batch (50% discount)
    "gemini-3.5-flash/batch":                (0.75,    4.50),
    "gemini-3.1-pro-preview/batch":          (1.00,    6.00),  
    "gemini-3.1-flash-image/batch":          (0.25,    1.50),  
    "gemini-3.1-flash-lite/batch":           (0.0375,  0.15),
    "gemini-3-flash-preview/batch":          (0.25,    1.50),  
    "gemini-2.5-pro/batch":                  (0.625,   5.00),
    "gemini-2.5-flash/batch":                (0.15,    1.25),
}

# Create a thread-safe cache of Vertex AI Clients
_dfb_vertex_clients: dict = {}   # location -> genai.Client
_dfb_vertex_clients_lock = Lock()


def _get_dfb_vertex_client(location: str):
    with _dfb_vertex_clients_lock:
        if location not in _dfb_vertex_clients:
            from google import genai
            from google.oauth2 import service_account
            creds = service_account.Credentials.from_service_account_file(
                VERTEX_CREDENTIALS_DFB_OCR_PATH,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            _dfb_vertex_clients[location] = genai.Client(
                vertexai=True,
                project=VERTEX_PROJECT_ID_DFB_OCR,
                location=location,
                credentials=creds,
            )
    return _dfb_vertex_clients[location]



# ============================================================================
# SYSTEM PROMPT — replace this placeholder with your own instructions
# ============================================================================

SYSTEM_PROMPT = ("PROMPT GOES HERE")


# ============================================================================
# Document reading
# ============================================================================

# PDF reader tuning knobs (see read_document_parts).
_RENDER_DPI = 260            # every PDF page renders at this DPI (high enough for handwriting)
_TEXT_LAYER_MIN_CHARS = 20   # include the embedded text layer only if it has real content


def _extract_acroform_values(filepath: str) -> str | None:
    """
    Some files have a text layer and form fields.

    This returns a readable dump of *filled* AcroForm field values, or None.

    Authoritative for checkbox/radio state and typed text on digital forms that
    keep live form widgets (class 1). Returns None if pypdf is missing, the PDF
    has no AcroForm, or every field is blank.
    """
    if not _PYPDF_AVAILABLE:
        return None
    try:
        fields = PdfReader(filepath).get_fields()
    except Exception:
        return None
    if not fields:
        return None

    lines = []
    for name, field in fields.items():
        try:
            value = field.value
        except Exception:
            value = None
        if value is None:
            continue
        text = str(value).strip()
        if text in ("", "/Off", "Off"):   # unchecked box / empty radio group
            continue
        lines.append(f"- {name}: {text.lstrip('/')}")  # checkbox states arrive as '/Yes'

    if not lines:
        return None
    return (
        "FORM FIELD VALUES (extracted from the PDF's AcroForm — authoritative for "
        "checkbox/radio state and typed text; reconcile against the rendered pages):\n"
        + "\n".join(lines)
    )


def _render_pages(doc, dpi: int) -> list:
    """Render every page of an open fitz document to a JPEG content part.

    JPEG (not PNG) because these pages are mostly photographic scans, which
    compress ~5-8x smaller as JPEG with no meaningful OCR loss, keeping the
    request under Gemini's inline-data size limit at full DPI.
    """
    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    parts = []
    for page in doc:
        pix = page.get_pixmap(matrix=matrix)
        parts.append({"type": "binary", "mime_type": "image/jpeg",
                      "data": pix.tobytes("jpg", jpg_quality=100)})
    return parts


def read_document_parts(filepath: str) -> list:
    """Read a document and return provider-agnostic content parts.

    Returns:
        List of content dicts:
        - {"type": "binary", "mime_type": "...", "data": b"..."}  (PDF, PNG, ...)
        - {"type": "text",   "text": "..."}
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".pdf":
        if not _FITZ_AVAILABLE:
            # No PyMuPDF — hand the raw PDF to the model and let it ingest natively.
            with open(filepath, "rb") as f:
                return [{"type": "binary", "mime_type": "application/pdf", "data": f.read()}]

        doc = fitz.open(filepath)
        try:
            parts = []

            # Embedded text layer, when the PDF carries real content (digital-native,
            # or a scanner's baked-in OCR). A hint only — the render is authoritative,
            # because scanner OCR mangles handwriting.
            page_texts = [page.get_text() for page in doc]
            if sum(len(t.strip()) for t in page_texts) >= _TEXT_LAYER_MIN_CHARS:
                combined = "\n".join(
                    f"--- PAGE {i + 1} ---\n{t.strip()}"
                    for i, t in enumerate(page_texts) if t.strip()
                )
                parts.append({
                    "type": "text",
                    "text": (
                        "DOCUMENT TEXT LAYER (extracted from the PDF; on scanned forms "
                        "this is imperfect OCR — treat the page images as authoritative "
                        "where they disagree):\n" + combined
                    ),
                })

            # AcroForm field values, when the form keeps live widgets.
            fields_dump = _extract_acroform_values(filepath)
            if fields_dump:
                parts.append({"type": "text", "text": fields_dump})

            # Always render every page at high DPI — the images are the source of truth.
            parts.extend(_render_pages(doc, _RENDER_DPI))
            return parts
        finally:
            doc.close()

    if ext == ".docx":
        if not _DOCX_AVAILABLE:
            print("python-docx is not installed. Install with: pip install python-docx")
            return []
        try:
            doc = DocxDocument(filepath)
            text = "\n".join(p.text for p in doc.paragraphs)
            if not text:
                return []
            return [{"type": "text", "text": text}]
        except Exception as e:
            print(f"Error reading Word document: {e}")
            return []

    print(f"Unsupported file type: {ext}")
    return []


# ============================================================================
# API wrappers
# ============================================================================

def _call_gemini(content_parts: list, system_instruction: str, model: str = None, max_retries: int = 8) -> tuple:
    from google.genai import types as gtypes
    from google.genai import errors as genai_errors
    from google.api_core import exceptions as google_exceptions

    model = model or next(iter(GEMINI_MODELS))
    location = GEMINI_MODELS.get(model)
    if not location:
        raise ValueError(f"Model '{model}' not found in GEMINI_MODELS — add it with a location.")
    client = _get_dfb_vertex_client(location)

    parts = []
    for p in content_parts:
        if p["type"] == "binary":
            parts.append(gtypes.Part.from_bytes(data=p["data"], mime_type=p["mime_type"]))
        elif p["type"] == "text":
            parts.append(gtypes.Part.from_text(text=p["text"]))

    # Optional per-model image resolution override (see GEMINI_MEDIA_RESOLUTION).
    # Omitted models keep the family default, so behaviour is unchanged unless set.
    config_kwargs = dict(system_instruction=system_instruction, temperature=0.0)
    mr_name = GEMINI_MEDIA_RESOLUTION.get(model)
    if mr_name:
        enum_attr = _MEDIA_RESOLUTION_ENUM.get(mr_name.lower())
        if not enum_attr:
            raise ValueError(
                f"Invalid media_resolution {mr_name!r} for {model} — use low|medium|high."
            )
        config_kwargs["media_resolution"] = getattr(gtypes.MediaResolution, enum_attr)

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=parts,
                config=gtypes.GenerateContentConfig(**config_kwargs),
            )
            text = response.text.strip()
            meta = response.usage_metadata
            input_tokens = getattr(meta, "prompt_token_count", 0)
            output_tokens = getattr(meta, "candidates_token_count", 0)
            return text, input_tokens, output_tokens
        except (google_exceptions.ResourceExhausted, google_exceptions.ServiceUnavailable,
                genai_errors.ServerError) as e:
            # Transient (quota exhaustion, 5xx) — back off and retry.
            if attempt < max_retries - 1:
                print(f"Transient API error (attempt {attempt + 1}): {e}")
                time.sleep(2 ** attempt)
            else:
                raise
        except genai_errors.ClientError as e:
            # 4xx. Retry 429 (rate limited); every other client error (400 bad config
            # such as an unsupported media_resolution, 401/403 auth, 404 model) will
            # never improve on retry — raise loudly instead of silently returning an
            # empty result that becomes a blank CSV row.
            if getattr(e, "code", None) == 429 and attempt < max_retries - 1:
                print(f"Rate limited (attempt {attempt + 1}): {e}")
                time.sleep(2 ** attempt)
            else:
                raise
    return "", 0, 0


# ============================================================================
# Multi mode (concurrent — cycles through all configured models per PDF)
# ============================================================================

def _run_multi(pdf_paths: list, csv_path: str) -> None:
    """Process all (PDF, provider, model) combinations concurrently.

    All requests are submitted at the same time via a thread pool.
    Results are written to csv_path as they complete (thread-safe).
    """
    # Build task list: (pdf_path, provider_name, model)
    tasks = []
    for pdf_path in pdf_paths:
        for provider_name, info in PROVIDERS.items():
            if not info["key"]:
                continue
            for model in info["models"]:
                tasks.append((pdf_path, provider_name, model))

    if not tasks:
        print("No tasks to run — check your provider keys.")
        return

    print(f"Submitting {len(tasks)} request(s) concurrently...\n")

    # Pre-warm clients sequentially before threads launch — avoids concurrent
    # auth token acquisition which causes connection errors under the Vertex SDK.
    providers_needed = {t[1] for t in tasks}
    if "gemini" in providers_needed:
        for loc in set(GEMINI_MODELS.values()):
            _get_dfb_vertex_client(loc)

    csv_lock = Lock()

    def _run_one(task):
        pdf_path, provider_name, model = task
        label = f"{os.path.basename(pdf_path)} | {provider_name}/{model}"
        try:
            result, input_tokens, output_tokens, elapsed = analyze_document(
                pdf_path, provider=provider_name, model=model
            )
        except Exception as e:
            print(f"  ERROR  {label}: {e}")
            return

        flat = flatten(result)
        flat["source_file"] = os.path.basename(pdf_path)
        flat["model"] = model

        total_tokens = input_tokens + output_tokens
        pricing = MODEL_PRICING.get(model, (0.0, 0.0))
        cost = (input_tokens * pricing[0] + output_tokens * pricing[1]) / 1_000_000

        flat["input_tokens"] = input_tokens
        flat["output_tokens"] = output_tokens
        flat["total_tokens"] = total_tokens
        flat["cost_euro"] = round(cost * USD_TO_EUR, 6)
        flat["time_seconds"] = round(elapsed, 2)

        with csv_lock:
            file_exists = os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0
            fieldnames = _get_fieldnames(csv_path, flat)
            with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, restval="", extrasaction="ignore")
                if not file_exists:
                    writer.writeheader()
                writer.writerow(flat)

        print(f"  done   {label}  ({elapsed:.1f}s, ${cost:.4f})")

    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        futures = {executor.submit(_run_one, t): t for t in tasks}
        for future in as_completed(futures):
            exc = future.exception()
            if exc:
                task = futures[future]
                print(f"  UNHANDLED {task[0]} | {task[1]}/{task[2]}: {exc}")


# ============================================================================
# Router
# ============================================================================

_CALL_FUNCTIONS = {
    "gemini": _call_gemini,
}

def call_api(content_parts: list, system_instruction: str, provider: str = "gemini", model: str = None, max_retries: int = 8) -> tuple:
    fn = _CALL_FUNCTIONS.get(provider.lower())
    if not fn:
        raise ValueError(f"Unknown provider '{provider}'. Supported: {', '.join(_CALL_FUNCTIONS)}")
    return fn(content_parts, system_instruction, model=model, max_retries=max_retries)


# ============================================================================
# Main
# ============================================================================

def analyze_document(filepath: str, system_prompt: str = None, provider: str = "gemini", model: str = None) -> tuple:
    """Analyse a document and return the LLM response as a dict.

    Args:
        filepath: Path to a PDF or DOCX file.
        system_prompt: Optional override for the system prompt.
        provider: API provider name ("gemini", "claude", "openai").
        model: Specific model name to use.

    Returns:
        Tuple of (parsed_dict, input_tokens, output_tokens, elapsed_seconds).
    """
    prompt = system_prompt or SYSTEM_PROMPT

    content_parts = read_document_parts(filepath)
    if not content_parts:
        return {}, 0, 0, 0.0

    start = time.time()
    raw, input_tokens, output_tokens = call_api(content_parts, prompt, provider=provider, model=model)
    elapsed = time.time() - start

    if not raw:
        return {}, 0, 0, elapsed

    # Strip markdown code fences if the model wraps the response
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]  # remove opening fence line
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]  # remove closing fence
        cleaned = cleaned.strip()

    try:
        return json.loads(cleaned), input_tokens, output_tokens, elapsed
    except json.JSONDecodeError:
        return {"raw_response": raw}, input_tokens, output_tokens, elapsed


def flatten(obj, parent_key="", sep="."):
    """Recursively flatten a nested dict to dot-notation keys."""
    items = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.update(flatten(v, new_key, sep))
    else:
        items[parent_key] = obj
    return items


def _get_fieldnames(csv_path: str, flat: dict) -> list[str]:
    """Return the CSV fieldnames to use for writing.

    If the file already exists and has a header, reuse it exactly so all rows
    stay aligned. Otherwise build the canonical order from the current flat dict.
    """
    prefix = ["model", "source_file"]
    suffix = ["input_tokens", "output_tokens", "total_tokens", "cost_euro", "time_seconds"]
    exclude = set(prefix) | set(suffix)

    if os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0:
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            try:
                existing = next(reader)
                # Only reuse the header if it contains actual form fields, not
                # just metadata — a metadata-only header means a previous run
                # failed before any data was extracted.
                if any(col not in exclude for col in existing):
                    return existing
            except StopIteration:
                pass  # empty file — fall through to build fresh

    field_keys = [k for k in flat.keys() if k not in exclude]
    return prefix + field_keys + suffix


# ============================================================================
# Vertex AI Batch Prediction mode
# ============================================================================

_TERMINAL_JOB_STATES = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"}


def _parts_fingerprint(parts: list) -> str:
    """Stable content hash of a request's parts, used to match a batch *output*
    record back to its source PDF.

    Vertex batch prediction does NOT guarantee output order matches input order
    (and may shard the output across files), so positional matching is unsafe.
    Each output record echoes the original `request`, and the verbatim text and
    inline base64 image data survive the round-trip unchanged — hashing those
    gives a key that's identical on the input and output sides regardless of
    ordering. Tolerates both camelCase (inlineData, as sent) and snake_case
    (inline_data) spellings in case the service normalizes the echoed request.
    """
    h = hashlib.sha256()
    for p in parts:
        if not isinstance(p, dict):
            continue
        inline = p.get("inlineData") or p.get("inline_data")
        if inline and inline.get("data"):
            h.update(b"B:")
            h.update(inline["data"].encode("utf-8"))
        elif p.get("text") is not None:
            h.update(b"T:")
            h.update(p["text"].encode("utf-8"))
    return h.hexdigest()


def _run_batch(pdf_paths: list, csv_path: str, cleanup: bool = True) -> None:
    """Submit Vertex AI Batch Prediction jobs (one per model) for all PDFs.

    Uses the true async batch prediction API -> 50% cheaper than interactive,
    no QPM quota pressure. Requires VERTEX_GCS_BUCKET_DFB_OCR to be set:
    the JSONL input is uploaded there and output is written back to the same bucket.
    """
    try:
        from google.cloud import storage as gcs_lib
    except ImportError:
        print("google-cloud-storage is required for --batch.")
        print("Install with:  pip install google-cloud-storage")
        sys.exit(1)

    from google.genai import types as gtypes
    from google.oauth2 import service_account

    if not VERTEX_BUCKET_DFB_OCR:
        print("VERTEX_BUCKET_DFB_OCR not set — required for --batch.")
        print("Add VERTEX_BUCKET_DFB_OCR=<bucket-name> to your .env file.")
        sys.exit(1)

    models_to_run = list(GEMINI_MODELS.keys())

    creds = service_account.Credentials.from_service_account_file(
        VERTEX_CREDENTIALS_DFB_OCR_PATH,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    gcs_client = gcs_lib.Client(credentials=creds, project=VERTEX_PROJECT_ID_DFB_OCR)
    bucket = gcs_client.bucket(VERTEX_BUCKET_DFB_OCR)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    for model in models_to_run:
        safe_model = model.replace("/", "_").replace(".", "_")
        gcs_prefix = f"dfb_ocr_batch/{timestamp}/{safe_model}"
        input_gcs_path = f"{gcs_prefix}/input.jsonl"
        output_gcs_prefix = f"{gcs_prefix}/output/"

        location = GEMINI_MODELS[model]
        vertex_client = _get_dfb_vertex_client(location)

        print(f"\n{'='*60}")
        print(f"Model: {model}  |  Location: {location}  |  {len(pdf_paths)} PDF(s)")
        print(f"{'='*60}")

        try:
            # Build JSONL — one request line per PDF, inline base64 data
            lines = []
            pdf_by_fingerprint = {}   # content hash -> source PDF (order-independent)
            for pdf_path in pdf_paths:
                content_parts = read_document_parts(pdf_path)
                if not content_parts:
                    print(f"  Skipping unreadable file: {os.path.basename(pdf_path)}")
                    continue

                parts = []
                for p in content_parts:
                    if p["type"] == "binary":
                        parts.append({
                            "inlineData": {
                                "mimeType": p["mime_type"],
                                "data": base64.b64encode(p["data"]).decode("utf-8"),
                            }
                        })
                    elif p["type"] == "text":
                        parts.append({"text": p["text"]})

                fingerprint = _parts_fingerprint(parts)
                if fingerprint in pdf_by_fingerprint:
                    print(f"  Warning: {os.path.basename(pdf_path)} has identical content to "
                          f"{os.path.basename(pdf_by_fingerprint[fingerprint])} — results for these "
                          f"two cannot be told apart and may be labelled with the wrong filename.")
                pdf_by_fingerprint[fingerprint] = pdf_path
                lines.append(json.dumps({
                    "request": {
                        "contents": [{"role": "user", "parts": parts}],
                        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                        "generationConfig": {"temperature": 0.0},
                    }
                }))

            if not lines:
                print(f"  No readable PDFs — skipping {model}.")
                continue

            # Upload input JSONL to GCS
            blob = bucket.blob(input_gcs_path)
            blob.upload_from_string("\n".join(lines).encode("utf-8"), content_type="application/json")
            print(f"  Input uploaded: gs://{VERTEX_BUCKET_DFB_OCR}/{input_gcs_path}")

            # Submit batch prediction job
            try:
                job = vertex_client.batches.create(
                    model=model,
                    src=f"gs://{VERTEX_BUCKET_DFB_OCR}/{input_gcs_path}",
                    config=gtypes.CreateBatchJobConfig(
                        dest=f"gs://{VERTEX_BUCKET_DFB_OCR}/{output_gcs_prefix}",
                    ),
                )
            except Exception as e:
                print(f"  Failed to submit batch job: {e}")
                continue

            print(f"  Job: {job.name}")
            print(f"  Polling every 30 s...")

            # Poll until terminal state
            while True:
                state_name = job.state.name if hasattr(job.state, "name") else str(job.state)
                if state_name in _TERMINAL_JOB_STATES:
                    break
                time.sleep(30)
                job = vertex_client.batches.get(name=job.name)
                state_name = job.state.name if hasattr(job.state, "name") else str(job.state)
                print(f"  State: {state_name}")

            if state_name != "JOB_STATE_SUCCEEDED":
                print(f"  Job ended with state {state_name} — skipping CSV write.")
                continue

            print(f"  Complete. Downloading output...")

            # Download output JSONL files from GCS
            result_lines = []
            for out_blob in bucket.list_blobs(prefix=output_gcs_prefix):
                if not out_blob.name.endswith(".jsonl"):
                    continue
                for raw in out_blob.download_as_text(encoding="utf-8").splitlines():
                    if raw.strip():
                        result_lines.append(raw.strip())

            print(f"  {len(result_lines)} result(s) received.")

            # Parse results and write CSV. Output order is NOT guaranteed to match
            # input order, so match each result back to its source PDF by hashing
            # the request that Vertex echoes into every output record.
            unmatched = 0
            for idx, raw_line in enumerate(result_lines):
                try:
                    result_obj = json.loads(raw_line)
                except json.JSONDecodeError:
                    print(f"  Warning: could not parse result line {idx} — skipping.")
                    unmatched += 1
                    continue

                # Recover the source PDF from the echoed request, not from position.
                echoed_parts = (
                    result_obj.get("request", {})
                    .get("contents", [{}])[0]
                    .get("parts", [])
                )
                pdf_path = pdf_by_fingerprint.get(_parts_fingerprint(echoed_parts))
                if pdf_path is None:
                    print(f"  Warning: result line {idx} matched no submitted PDF "
                          f"(unrecognized request fingerprint) — labelling 'unmatched'.")
                    pdf_path = f"unmatched_{idx}"
                    unmatched += 1

                try:
                    candidate = result_obj.get("response", {}).get("candidates", [{}])[0]
                    response_text = (
                        candidate.get("content", {})
                        .get("parts", [{}])[0]
                        .get("text", "")
                        .strip()
                    )
                    usage = result_obj.get("response", {}).get("usageMetadata", {})
                    input_tokens = usage.get("promptTokenCount", 0)
                    output_tokens = usage.get("candidatesTokenCount", 0)
                except (KeyError, IndexError, TypeError):
                    response_text = ""
                    input_tokens = 0
                    output_tokens = 0

                cleaned = response_text
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("\n", 1)[-1]
                    if cleaned.endswith("```"):
                        cleaned = cleaned.rsplit("```", 1)[0]
                    cleaned = cleaned.strip()

                try:
                    parsed = json.loads(cleaned)
                except json.JSONDecodeError:
                    parsed = {"raw_response": response_text}

                flat = flatten(parsed)
                flat["source_file"] = os.path.basename(pdf_path)
                flat["model"] = model

                # Prefer batch pricing; fall back to standard pricing
                pricing = MODEL_PRICING.get(f"{model}/batch", MODEL_PRICING.get(model, (0.0, 0.0)))
                cost = (input_tokens * pricing[0] + output_tokens * pricing[1]) / 1_000_000

                flat["input_tokens"] = input_tokens
                flat["output_tokens"] = output_tokens
                flat["total_tokens"] = input_tokens + output_tokens
                flat["cost_euro"] = round(cost * USD_TO_EUR, 6)
                flat["time_seconds"] = ""  # not available for batch jobs

                file_exists = os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0
                fieldnames = _get_fieldnames(csv_path, flat)
                with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames, restval="", extrasaction="ignore")
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(flat)

            if unmatched:
                print(f"  WARNING: {unmatched} of {len(result_lines)} result(s) could not be "
                      f"matched to a source PDF — check those rows before trusting them.")
            print(f"  Results written to {csv_path}")
        finally:
            # Always clear this run's input + output from GCS — on success,
            # failure, or error — so applicant data never lingers in the bucket.
            # Pass cleanup=False (--keep-gcs) to retain it for debugging.
            if cleanup:
                deleted = 0
                for blob in bucket.list_blobs(prefix=gcs_prefix):
                    try:
                        blob.delete()
                        deleted += 1
                    except Exception as e:
                        print(f"  Warning: could not delete {blob.name}: {e}")
                if deleted:
                    print(f"  Cleaned up {deleted} GCS object(s) under {gcs_prefix}/")


def process_file(doc_path: str, csv_path: str) -> None:
    """Run all configured providers/models on a single file, appending results to csv_path."""
    for provider_name, info in PROVIDERS.items():
        if not info["key"]:
            print(f"Skipping {provider_name} — no API key configured.")
            continue

        for model in info["models"]:
            print(f"\n{'='*60}")
            print(f"File: {os.path.basename(doc_path)}  |  {provider_name} / {model}")
            print(f"{'='*60}")

            try:
                result, input_tokens, output_tokens, elapsed = analyze_document(
                    doc_path, provider=provider_name, model=model
                )
            except Exception as e:
                print(f"Error with {provider_name}/{model}: {e}")
                continue

            if not result:
                print(f"No result returned for {provider_name}/{model}.")
                continue

            flat = flatten(result)
            flat["source_file"] = os.path.basename(doc_path)

            total_tokens = input_tokens + output_tokens
            pricing = MODEL_PRICING.get(model, (0.0, 0.0))
            cost = (input_tokens * pricing[0] + output_tokens * pricing[1]) / 1_000_000

            file_exists = os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0
            fieldnames = _get_fieldnames(csv_path, flat)
            with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, restval="", extrasaction="ignore")
                if not file_exists:
                    writer.writeheader()
                flat["model"] = model
                flat["input_tokens"] = input_tokens
                flat["output_tokens"] = output_tokens
                flat["total_tokens"] = total_tokens
                flat["cost_euro"] = round(cost * USD_TO_EUR, 6)
                flat["time_seconds"] = round(elapsed, 2)
                writer.writerow(flat)

            print(f"CSV row appended to: {csv_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DFB OCR tool")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("file", nargs="?", help="Path to a single PDF or DOCX file")
    group.add_argument("--multi", metavar="FOLDER", help="Process all PDFs in a folder concurrently via the standard interactive API.")
    group.add_argument("--batch", metavar="FOLDER", help="Process all PDFs via Vertex AI Batch Prediction (async, 50%% cheaper). Requires VERTEX_GCS_BUCKET_DFB_OCR in .env.")
    parser.add_argument("--output", metavar="CSV", help="Output CSV path (batch modes only; defaults to <folder>/results.csv)")
    parser.add_argument("--keep-gcs", action="store_true", help="(--batch only) Keep the input/output JSONL in the bucket instead of deleting it after each run. Default: delete.")
    args = parser.parse_args()

    if args.multi:
        folder = args.multi
        if not os.path.isdir(folder):
            print(f"Folder not found: {folder}")
            sys.exit(1)

        pdfs = sorted(
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(".pdf")
        )
        if not pdfs:
            print(f"No PDF files found in: {folder}")
            sys.exit(1)

        csv_path = args.output or os.path.join(folder, "results.csv")

        print(f"Multi mode: {len(pdfs)} PDF(s) | Output: {csv_path}")
        print()

        _run_multi(pdfs, csv_path)
        print(f"\nMulti complete. Results written to: {csv_path}")

    elif args.batch:
        folder = args.batch
        if not os.path.isdir(folder):
            print(f"Folder not found: {folder}")
            sys.exit(1)

        pdfs = sorted(
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(".pdf")
        )
        if not pdfs:
            print(f"No PDF files found in: {folder}")
            sys.exit(1)

        csv_path = args.output or os.path.join(folder, "results.csv")

        print(f"Batch mode: {len(pdfs)} PDF(s) | Output: {csv_path}")
        print()

        _run_batch(pdfs, csv_path, cleanup=not args.keep_gcs)
        print(f"\nBatch complete. Results written to: {csv_path}")

    else:
        doc_path = args.file
        if not os.path.isfile(doc_path):
            print(f"File not found: {doc_path}")
            sys.exit(1)
        csv_path = args.output or os.path.splitext(doc_path)[0] + "_extracted.csv"
        process_file(doc_path, csv_path)
