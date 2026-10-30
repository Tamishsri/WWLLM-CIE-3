# WWLLM — Multilingual Lecture Studio

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-Academic-green.svg)](#license)

> Turn a lecture recording into a synced transcript, structured notes, a practice quiz and translations — then navigate all of it by sound, in any Indian script. Powered by AI4Bharat's IndicConformer, BharatGen Param-1 and IndicTrans2.

**Every output links back to the moment it came from.** A note, a quiz answer, a concept or a search result is one click from the exact second the lecturer said it.

---

## Quick start

The studio runs on any laptop. Stages whose models aren't installed fall back to offline engines automatically and say so in the UI.

```bash
pip install -r requirements-web.txt
python server.py --open
```

Open <http://127.0.0.1:8000> and press **Try the sample lecture**.

To run the real models, see [Full ML stack](#full-ml-stack).

## What makes it different

### Polyglot search — `Ctrl K`

Indian lectures are code-mixed: the lecturer says *ग्रेडिएंट*, the student thinks *gradient*. Type either, or even the Tamil *கிரேடியன்ட்*, and land on the same moment in a Hindi lecture.

No model and no dictionary are involved. The nine major Indic Unicode blocks share one layout, so a single table reduces any of those scripts (and Latin spelling, through phonetic rules) to the same consonant skeleton:

```
gradient     →  K R T N T
ग्रेडिएंट      →  K R T N T
கிரேடியன்ட்    →  K R T N T
```

Tamil writes k/g, t/d and p/b with one letter, so voicing is folded together too. Matching is deliberately strict, so everyday English doesn't collide with native words.

Words that were *translated* rather than borrowed have no phonetic link: "slope" is spoken as ढलान. Those are reached through the **notes bridge**: note sections are aligned to the transcript in lecture order, so a match in the notes leads to the right moment in the audio.

### Time-anchored everything

- **Notes** — every section has a chip that plays where that topic starts.
- **Quiz** — after you answer, *Hear it in the lecture* replays the moment that holds the answer.
- **Transcript** — words light up as they're spoken; click any word to jump there.

### Concept map

The ideas in a lecture, laid out left to right along its timeline and linked by where they're discussed together. Labels come from your notes, and each concept resolves to the words actually spoken (*Learning rate* → *लर्निंग रेट*). Pick one to see every mention on the waveform and hear each in turn.

### Lecture Rangoli and Theatre mode

Every lecture gets a generative fingerprint in the mirrored symmetry of a floor rangoli: **one ring per speech chunk, shaped by that chunk's real loudness contour**. No two recordings look alike.

In **Theatre** (`F`) it draws itself as the lecture plays, pulsing with the live audio, with word-by-word captions and concepts appearing as they're spoken. Save the finished pattern as an image.

### Honest about where content came from

Every result carries a badge: *Live model*, *Offline fallback*, *Pre-rendered sample*, *Eval-set replay* or *Your text*. Nothing offline pretends to be model output.

## The pipeline

```
 Audio / video
      │
      ▼
 Silero-VAD ──────── speech chunks ≤ 25 s, with timestamps
      │
      ▼
 IndicConformer ──── per-language ASR, chunk by chunk
      │
      ▼
 Transcript ◄──────── editable; or paste your own
      │
      ├──► BharatGen Param-1 ──► Notes (streamed)  ──┐
      ├──► BharatGen Param-1 ──► Quiz                ├──► IndicTrans2 ──► 10 Indian languages
      │                                              │
      └──► Phonetic index ──► search · anchors · concepts · rangoli
```

| Stage | Real model | Offline fallback |
|---|---|---|
| Speech detection | Silero-VAD | Energy VAD (NumPy, `.wav` only) |
| Transcription | IndicConformer | Sample lecture; replays stored predictions for benchmark clips; otherwise paste a transcript |
| Notes | BharatGen Param-1 | Extractive summariser |
| Quiz | BharatGen Param-1 | Fill-in-the-blank from key terms |
| Translation | IndicTrans2 | Sample lecture only (hand-written Hindi, Tamil, Kannada) |

Search, anchors, concepts, the rangoli and playback run in the browser and never need the ML stack.

## Full ML stack

**Prerequisites:** Python 3.10+, ffmpeg on PATH, and a GPU with ≥ 8 GB VRAM for comfortable speeds.

```bash
# AI4Bharat's NeMo fork (needed for IndicConformer)
git clone --branch nemo-v2 https://github.com/AI4Bharat/NeMo.git
cd NeMo && bash reinstall.sh && cd ..

pip install -r requirements.txt
pip install git+https://github.com/VarunGumma/IndicTransToolkit.git

python server.py
```

Each stage switches to its real model on the next start. Click the engine pill in the top bar to see what's live. `python server.py --demo` forces the offline engines.

A classic Streamlit UI is also kept: `streamlit run app.py`. It needs the full stack to do anything.

## Keyboard

| Key | Action |
|---|---|
| `Ctrl K` or `/` | Search the lecture in any script |
| `Space` | Play / pause |
| `1`–`5` | Jump to a stage |
| `F` | Theatre |
| `Ctrl Enter` | Run the current stage |
| `T` | Toggle theme |
| `?` | All shortcuts |

## Configuration

Everything lives in [`configs/config.yaml`](configs/config.yaml). Override any value with `WWLLM_<SECTION>_<KEY>`, e.g. `WWLLM_ASR_BATCH_SIZE=4`.

| Section | Key | Default |
|---|---|---|
| `asr` | `model_template` | `ai4bharat/indicconformer_stt_{iso}_hybrid_rnnt_large` |
| `asr` | `batch_size` | `8` |
| `llm` | `model_id` | `bharatgenai/Param-1-2.9B-Instruct` |
| `llm` | `quantize_8bit` | `true` (falls back to fp16 without CUDA) |
| `translation` | `model_id` | `ai4bharat/indictrans2-en-indic-1B` |
| `audio` | `max_chunk_sec` | `25.0` |
| `audio` | `vad_threshold` | `0.45` |
| `evaluation` | `simulated` | `true` for the bundled set |

## Evaluation

The **Benchmark** page scores predicted transcripts against references: WER with word-level alignment (substitutions, deletions, insertions), ROUGE, and BERTScore when `bert-score` is installed. Failures (WER > 0.30) expand into a visual diff.

> **The bundled 40-lecture set is simulated, and it is not a measurement of IndicConformer.** [`generate_test_dataset.py`](generate_test_dataset.py) writes the "ASR predictions" by hand, with errors designed to resemble noise, code-switching and truncation, and its audio clips are synthetic sine tones. The figures below (average WER 0.2707, 11 of 40 failures) demonstrate that the evaluation pipeline works. They say nothing about real model accuracy.
>
> To measure the real model, transcribe real recordings, then use **Evaluate your own set** on the Benchmark page or the CLI below. Set `evaluation.simulated: false` once the bundled predictions are replaced.

```bash
python scripts/run_eval.py \
    --ground-truth data/ground_truth/references.json \
    --predictions  outputs/transcripts/eval_inputs.json \
    --output-dir   outputs/evaluations
```

Ground truth is a JSON array:

```json
[{ "audio_file": "lecture_01.wav", "transcript": "…", "summary": "optional", "condition": "optional" }]
```

Predictions are `{ "audio_file": "transcript" }`, or `{ "predictions": …, "summaries_pred": …, "summaries_ref": … }`.

| Metric | Library | Failure threshold |
|---|---|---|
| WER | `jiwer` | > 0.30 |
| BERTScore | `bert-score` | F1 < 0.70 |
| ROUGE-1/2/L | `rouge-score`, or a built-in implementation | — |

## Tests

```bash
python -m unittest discover -s tests -t .
node --test "tests/web/*.test.mjs"
```

The Python suite covers the offline engines, evaluation, the service layer (including live-model branches with fake engines), the HTTP API and the Streamlit app. The JS suite covers phonetic matching, the lecture index, anchors and concept extraction.

## Project structure

```
├── server.py                  # Web studio entry point
├── app.py                     # Classic Streamlit UI
├── requirements.txt           # Full ML stack
├── requirements-web.txt       # Offline studio only
├── configs/config.yaml
├── src/
│   ├── config.py              # Config + WWLLM_* overrides
│   ├── api/app.py             # Starlette routes, NDJSON streaming
│   ├── services/
│   │   ├── engines.py         # PipelineHub: real models + fallbacks as event streams
│   │   ├── capabilities.py    # What can run on this machine
│   │   ├── benchmark.py       # Evaluation reports
│   │   ├── files.py           # Upload store
│   │   └── sample.py          # Sample lecture + synthetic audio
│   └── pipelines/
│       ├── audio_utils.py     # VAD ingestion and chunking
│       ├── asr_engine.py      # IndicConformer
│       ├── llm_engine.py      # Param-1 notes and quiz, token streaming
│       ├── translator.py      # IndicTrans2
│       ├── eval_pipeline.py   # WER, alignment, BERTScore, ROUGE
│       ├── fallback.py        # Offline VAD, notes, quiz
│       ├── quiz_schema.py     # MCQ normalisation
│       └── languages.py
├── web/                       # No build step: plain ES modules
│   ├── index.html
│   └── assets/
│       ├── css/app.css
│       └── js/
│           ├── phonetic.js    # Cross-script consonant skeletons
│           ├── lecture.js     # Index, ranking, ordered alignment
│           ├── anchors.js     # Notes and quiz → moments
│           ├── concepts.js    # Concept extraction and layout
│           ├── rangoli.js     # Generative fingerprint
│           ├── theatre.js     # Immersive playback
│           ├── palette.js     # Ctrl+K search
│           ├── river.js       # Hero animation
│           ├── waveform.js    # Decoding, chunks, scrubbing
│           ├── studio.js      # Studio view
│           └── benchmark.js   # Dashboard
├── data/
│   ├── sample/                # Sample lecture content
│   └── ground_truth/          # Simulated benchmark set
├── scripts/run_eval.py
└── tests/
```

## Limits worth knowing

- **Word timing is estimated.** ASR gives timestamps per chunk; words are interpolated by character position inside it. It is close enough to follow along, but it isn't forced alignment.
- **Phonetic matching finds loanwords.** Native vocabulary needs the notes bridge, and a false link is possible where unrelated words share a skeleton.
- **The sample lecture's audio is a synthetic voice** and its transcript, notes, quiz and translations are hand-written. They are labelled *Pre-rendered sample* wherever a model didn't produce them. The Tamil and Kannada translations have not been reviewed by native speakers.
- **`.wav` only offline.** Decoding MP3 or MP4 needs `torchaudio` or `soundfile`, plus ffmpeg for video.

## Model cards

| Component | Model | Parameters | Source |
|---|---|---|---|
| ASR | IndicConformer (hybrid CTC-RNNT) | ~600M | AI4Bharat |
| LLM | BharatGen Param-1-2.9B-Instruct | 2.9B | BharatGen |
| Translation | IndicTrans2 En→Indic | 1B | AI4Bharat |
| VAD | Silero-VAD v5 | ~2M | Silero |

## GPU memory

- **8-bit quantisation** reduces Param-1's VRAM use when CUDA and `bitsandbytes` are available.
- **VAD chunking** keeps ASR batches small enough to avoid OOM.
- **CPU fallback** retries a failed ASR batch on CPU after a CUDA OOM.
- **One ASR checkpoint at a time**: switching language unloads the previous model.

## License

For **academic and research purposes only**. Model weights are subject to their own licenses (AI4Bharat, BharatGen Non-Commercial License).
