# WWLLM — Multilingual Lecture-to-Notes & Quiz Generator

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-Academic-green.svg)](#)

> An end-to-end ML pipeline that converts multilingual lecture recordings into structured notes, MCQ quizzes, and translations — powered by AI4Bharat's IndicConformer, BharatGen Param-1, and IndicTrans2.

---

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
│  Audio/Video │────▶│  Silero-VAD   │────▶│  IndicConformer  │────▶│  Transcript  │
│  Upload      │     │  Chunking    │     │  ASR (NeMo)      │     │  (text)      │
└─────────────┘     └──────────────┘     └─────────────────┘     └──────┬───────┘
                                                                        │
                                                          ┌─────────────┴──────────────┐
                                                          ▼                            ▼
                                                 ┌────────────────┐          ┌─────────────────┐
                                                 │  BharatGen      │          │  BharatGen       │
                                                 │  Notes (MD)     │          │  Quiz (JSON)     │
                                                 └────────┬───────┘          └────────┬────────┘
                                                          │                           │
                                                          ▼                           ▼
                                                 ┌─────────────────────────────────────┐
                                                 │  IndicTrans2 Translation             │
                                                 │  (Hindi / Tamil / Kannada)           │
                                                 └─────────────────────────────────────┘
```

## 📁 Project Structure

```
WWLLM AI4Bharat/
├── app.py                          # Streamlit entry point
├── requirements.txt                # Python dependencies
├── .gitignore
├── README.md
│
├── configs/
│   └── config.yaml                 # Centralised configuration
│
├── src/
│   ├── __init__.py
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── audio_utils.py          # VAD-based audio ingestion & chunking
│   │   ├── asr_engine.py           # IndicConformer ASR inference
│   │   ├── llm_engine.py           # BharatGen Param-1 notes & quiz
│   │   ├── translator.py           # IndicTrans2 translation
│   │   └── eval_pipeline.py        # WER / BERTScore / ROUGE evaluation
│   └── ui/
│       ├── __init__.py
│       └── styles.py               # Dark-theme CSS injection
│
├── scripts/
│   └── run_eval.py                 # CLI evaluation runner
│
├── data/
│   ├── ground_truth/
│   │   └── references.json         # Ground-truth transcripts for eval
│   └── uploads/                    # Uploaded files (gitignored)
│
├── outputs/
│   ├── transcripts/
│   ├── notes/
│   ├── quizzes/
│   └── evaluations/                # failure_analysis.csv
│
└── tests/
    └── __init__.py
```

## 🚀 Quick Start

### 1. Prerequisites

- **Python 3.10+**
- **CUDA 11.8+** with a GPU ≥ 8 GB VRAM (e.g. RTX 3070+)
- **ffmpeg** installed and on PATH (for video → audio extraction)

### 2. Install AI4Bharat NeMo Fork

IndicConformer requires AI4Bharat's custom NeMo build:

```bash
git clone --branch nemo-v2 https://github.com/AI4Bharat/NeMo.git
cd NeMo && bash reinstall.sh
cd ..
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
pip install git+https://github.com/VarunGumma/IndicTransToolkit.git
```

### 4. Launch the Application

```bash
streamlit run app.py
```

## ⚙️ Configuration

All parameters are centralized in [`configs/config.yaml`](configs/config.yaml):

| Section       | Key               | Default                                                  |
|---------------|-------------------|----------------------------------------------------------|
| `asr`         | `model_id`        | `ai4bharat/indicconformer_stt_hi_hybrid_rnnt_large`      |
| `asr`         | `batch_size`      | `8`                                                      |
| `llm`         | `model_id`        | `bharatgenai/Param-1-2.9B-Instruct`                      |
| `llm`         | `quantize_8bit`   | `true`                                                   |
| `translation` | `model_id`        | `ai4bharat/indictrans2-en-indic-1B`                      |
| `audio`       | `max_chunk_sec`   | `25.0`                                                   |
| `audio`       | `vad_threshold`   | `0.45`                                                   |

## 📊 Evaluation

### GUI (Streamlit)
Navigate to the **📊 Evaluate** tab and upload a ground-truth JSON file.

### CLI
```bash
python scripts/run_eval.py \
    --ground-truth data/ground_truth/references.json \
    --predictions  outputs/transcripts/predictions.json \
    --output-dir   outputs/evaluations
```

### Ground-Truth Format
```json
[
    {
        "audio_file": "lecture_01.wav",
        "transcript": "Ground truth transcript text ...",
        "summary": "Optional reference summary ..."
    }
]
```

### Metrics
| Metric     | Library      | Purpose                              | Failure Threshold |
|------------|-------------|--------------------------------------|-------------------|
| WER        | `jiwer`     | ASR transcript accuracy              | > 0.30            |
| BERTScore  | `bert-score`| Semantic similarity of summaries     | F1 < 0.70         |
| ROUGE-1/2/L| `rouge-score`| N-gram overlap of summaries         | —                 |

### Benchmark & Failure Analysis Results (40 Test Cases)

The evaluation suite was executed across **40 multi-domain lecture samples** spanning Machine Learning, Operating Systems, Distributed Systems, Dravidian NLP (Tamil, Kannada), and Cloud Architecture:

- **Total Test Cases:** 40
- **Average Word Error Rate (WER):** `0.2707`
- **Clean Audio Accuracy:** WER `0.00% – 15.0%`
- **Flagged Failure Cases (WER > 0.30):** 11 / 40 (27.5%)
- **Logged Failure Analysis:** [`outputs/evaluations/failure_analysis.csv`](outputs/evaluations/failure_analysis.csv)

#### Root-Cause Categorization of Failure Modes:
1. **High Technical Jargon & Code-Switching:** English terms embedded in Devanagari/Dravidian speech (e.g. *multivariate calculus, backpropagation*) caused phonetic mis-alignments (`lecture_06`, `lecture_32`).
2. **Acoustic Noise & Low SNR:** Low-frequency air-conditioner hum and reverberation degraded CTC confidence scores (`lecture_10`, `lecture_18`).
3. **Conversational Overlap:** Student interruptions and crosstalk caused multi-speaker token collisions (`lecture_13`).
4. **Fast Colloquial Speech & Elision:** Dropped unstressed syllables in Dravidian dialects (`lecture_22`, `lecture_24`).
5. **Early Audio Truncation:** Clip boundary cut-off before completion of sentence units (`lecture_35`).

## 🧩 Model Cards

| Component    | Model                                                    | Parameters | Source    |
|-------------|----------------------------------------------------------|------------|-----------|
| ASR          | IndicConformer (hybrid CTC-RNNT)                         | ~600M      | AI4Bharat |
| LLM          | BharatGen Param-1-2.9B-Instruct                         | 2.9B       | BharatGen |
| Translation  | IndicTrans2 En→Indic                                    | 1B         | AI4Bharat |
| VAD          | Silero-VAD v5                                           | ~2M        | Silero    |

## 🔧 GPU Memory Management

- **8-bit Quantisation**: BharatGen Param-1 is loaded with `bitsandbytes` INT8, reducing VRAM from ~6 GB to ~3 GB.
- **VAD Chunking**: Audio is split into 15–30s speech chunks to prevent OOM during ASR.
- **CPU Fallback**: If CUDA OOM occurs during ASR, the failing batch automatically retries on CPU.
- **Singleton Caching**: All models are loaded once via `st.cache_resource` and reused across Streamlit re-runs.

## 📄 License

This project is for **academic/research purposes only**. Model weights are subject to their respective licenses (AI4Bharat, BharatGen Non-Commercial License).
