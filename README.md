# DFB OCR Tool

Standalone document OCR / data-extraction code. Reads a PDF or DOCX, sends it to
a Gemini model on Vertex AI with a configurable system prompt, and writes the
extracted fields to CSV.

Built for Dublin Fire Brigade forms (Incident Reports, Pre-Incident Plans), but
the system prompt is a plain placeholder you can swap for any extraction schema.

## How it works

For each PDF the tool renders every page to a high-DPI JPEG (the source of
truth for handwriting and checkboxes) and, when present, also passes the
embedded text layer and any AcroForm field values as supporting hints. Gemini
returns one JSON object per document, which is flattened into a CSV row.

## Requirements

- Python 3.10+
- A GCP project with **Vertex AI enabled** and a **service-account JSON key**
- Dependencies:

```bash
pip install -r requirements.txt
```

## Setup

1. Copy `.env.example` to `.env` and fill in your values:

   | Variable | Purpose |
   |----------|---------|
   | `VERTEX_PROJECT_ID_DFB_OCR`  | GCP project ID with Vertex AI enabled |
   | `VERTEX_CREDENTIALS_DFB_OCR` | Path to the service-account JSON key |
   | `VERTEX_BUCKET_DFB_OCR`      | GCS bucket name (only needed for `--batch`) |

2. **Set the system prompt.** `SYSTEM_PROMPT` in `OCR_Example.py` ships as a
   `"PROMPT GOES HERE"` placeholder — replace it with the extraction
   instructions and JSON schema for your form.

3. (Optional) Adjust the model list in `GEMINI_MODELS` near the top of the
   script. Each model is mapped to its Vertex AI region.

## Usage

```bash
# Single document → writes <document>_extracted.csv
python OCR_Example.py path/to/form.pdf

# All PDFs in a folder, concurrently (standard interactive API)
python OCR_Example.py --multi path/to/folder [--output results.csv]

# All PDFs via Vertex AI Batch Prediction — queues and takes longer,
# but ~50% cheaper. Requires VERTEX_BUCKET_DFB_OCR.
python OCR_Example.py --batch path/to/folder [--output results.csv] [--keep-gcs]
```

- `--output` sets the CSV path (defaults to `<folder>/results.csv`).
- `--keep-gcs` (batch only) keeps the input/output JSONL in the bucket instead
  of deleting it after the run. By default the run's GCS objects are removed so
  document data does not linger in the bucket.

## Output

A UTF-8 CSV with one row per (document, model): the flattened extracted fields
plus `model`, `source_file`, `input_tokens`, `output_tokens`, `total_tokens`,
`cost_euro`, and `time_seconds`.

## Notes

- **Costs** — per-model pricing is defined in `MODEL_PRICING`; the reported
  `cost_euro` uses the `USD_TO_EUR` rate in the script. Batch mode uses the
  discounted `/batch` pricing.
