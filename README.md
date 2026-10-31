# WWLLM — Multilingual Lecture Studio

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-Academic-green.svg)](#license)

> Turn a lecture recording into a synced transcript, structured notes, a practice quiz and translations — then navigate all of it by sound, in any Indian script. Real models throughout: Silero-VAD, Whisper or AI4Bharat's IndicConformer, BharatGen Param-1, and NLLB-200 or AI4Bharat's IndicTrans2.

**Every output links back to the moment it came from.** A note, a quiz answer, a concept or a search result is one click from the exact second the lecturer said it.

---

## Quick start

The studio runs on any laptop, with no internet connection once installed (fonts are bundled). The sample lecture is real text-to-speech audio, and its transcript, notes, quiz and translations are stored output from the real models, so it works without them.

```bash
pip install -r requirements-web.txt
python server.py --open
```

Open <http://127.0.0.1:8000> and press **Try the sample lecture**.

For your own recordings, see [Real models](#real-models).

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

Every lecture gets a generative fingerprint in the mirrored symmetry of a floor rangoli: **rings shaped by the real loudness contour of the speech**. No two recordings look alike.

In **Theatre** (`F`) it draws itself as the lecture plays, pulsing with the live audio, with word-by-word captions and concepts appearing as they're spoken. Save the finished pattern as an image.

### Honest about where content came from

Every result carries a badge: *Live model*, *Model output · stored*, *Offline fallback*, *Stored prediction* or *Your text*. Nothing offline pretends to be model output.

## The pipeline

```
 Audio / video
      │
      ▼
 Silero-VAD ──────── one chunk per stretch of speech, with timestamps
      │
      ▼
 Whisper or ───────── per-language ASR, chunk by chunk
 IndicConformer
      │
      ▼
 Transcript ◄──────── editable; or paste your own
      │
      ├──► BharatGen Param-1 ──► Notes (streamed)  ──┐
      ├──► BharatGen Param-1 ──► Quiz (streamed)     ├──► NLLB-200 or IndicTrans2 ──► 10 Indian languages
      │                                              │
      └──► Phonetic index ──► search · anchors · concepts · rangoli
```

Each stage uses the first of these that works, and says which in the UI:

| Stage | 1. Stored | 2. Real model | 3. Offline |
|---|---|---|---|
| Speech detection | Yes | Silero-VAD | Energy VAD (NumPy) |
| Transcription | Yes | IndicConformer, else open Whisper | Paste a transcript |
| Notes | Yes | BharatGen Param-1 | Extractive summariser |
| Quiz | Yes | BharatGen Param-1 | Fill-in-the-blank from key terms |
| Translation | Yes | IndicTrans2, else NLLB-200 | None: a stage that can't be done honestly says so |

Results are stored by content in `data/cache/`, so a lecture never has to be processed twice. **Re-transcribe** and **Regenerate** run the model again. The sample's outputs live in `data/sample/results/`; `python scripts/precompute_sample.py` rebuilds them.

Search, anchors, concepts, the rangoli and playback run in the browser and never need the ML stack.

## Real models

**Prerequisites:** Python 3.10+. ffmpeg on PATH for video and `.m4a`. No GPU or login is required.

```bash
pip install -r requirements.txt
python server.py
```

Models download on first use, about 8 GB in all, and each stage shows its progress. Only one large model stays in memory at a time. For a talk, run `python server.py --offline` so nothing tries to reach the network. `python server.py --demo` forces the offline engines. Click the engine pill in the top bar to see what's live.

| Stage | Model | Licence | Access |
|---|---|---|---|
| VAD | Silero-VAD v5 | MIT | Open |
| ASR (Hindi, Tamil, Kannada, Telugu) | `vasista22/whisper-*-small` | Apache 2.0 | Open |
| ASR (English, others) | `openai/whisper-small` | MIT | Open |
| Notes and quiz | `bharatgenai/Param-1-2.9B-Instruct` | See model card | Open |
| Translation | `facebook/nllb-200-distilled-600M` | CC-BY-NC 4.0 | Open |
| ASR, preferred | `ai4bharat/indicconformer_stt_*` | MIT | **Gated** |
| Translation, preferred | `ai4bharat/indictrans2-en-indic-1B` | MIT | **Gated** |

### Unlocking the AI4Bharat models

IndicConformer and IndicTrans2 are the project's first choice, and the code path for each is in place, but both are gated on Hugging Face, so using them needs your own account:

1. Open each model page while signed in and accept its terms.
2. Run `huggingface-cli login`.
3. Install their toolkits:
   ```bash
   pip install "nemo_toolkit[asr]>=1.23.0"
   pip install git+https://github.com/VarunGumma/IndicTransToolkit.git
   ```
4. Restart the server. The studio prefers them from then on, and falls back to Whisper and NLLB-200 if either fails to load.

A classic Streamlit UI is also kept: `streamlit run app.py`.

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
| `asr` | `batch_size` | `8` (IndicConformer) |
| `llm` | `model_id` | `bharatgenai/Param-1-2.9B-Instruct` |
| `llm` | `quantize_8bit` | `true` (needs CUDA and `bitsandbytes`) |
| `llm` | `cpu_max_new_tokens_notes`, `cpu_max_new_tokens_quiz` | Length ceilings without a GPU |
| `translation` | `model_id` | `ai4bharat/indictrans2-en-indic-1B` |
| `translation` | `open_model_id` | `facebook/nllb-200-distilled-600M` |
| `audio` | `max_chunk_sec` | `25.0` |
| `audio` | `vad_threshold` | `0.45` |

## Evaluation

The **Benchmark** page scores predicted transcripts against references: WER with word-level alignment (substitutions, deletions, insertions), ROUGE, and BERTScore when `bert-score` is installed. Failures (WER > 0.30) expand into a visual diff. Punctuation and case are normalised first, because they aren't word errors.

The bundled 40-lecture set is a real measurement. Meta's MMS text-to-speech voices read each reference aloud, then Silero-VAD and Whisper transcribed the audio:

| Language | Lectures | Model | Average WER |
|---|---|---|---|
| English | 22 | `openai/whisper-small` | 0.257 |
| Hindi | 14 | `vasista22/whisper-hindi-small` | 0.277 |
| Tamil | 2 | `vasista22/whisper-tamil-small` | 0.225 |
| Kannada | 2 | `vasista22/whisper-kannada-small` | 0.167 |
| All | 40 | | 0.258 |

12 of the 40 lectures fail the 0.30 threshold. This is clean, machine-read speech, so real classrooms will score worse, and the Tamil and Kannada rows rest on two clips each. IndicConformer hasn't been measured, because it needs a Hugging Face login.

To reproduce it, or measure another model:

```bash
python scripts/generate_speech.py --benchmark       # read each reference aloud
python scripts/run_benchmark.py                     # transcribe, then score
python scripts/run_benchmark.py --reuse --summaries # add Param-1 summaries for ROUGE
```

To score transcripts from elsewhere, use **Evaluate your own set** on the Benchmark page, or:

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

The Python suite covers the offline engines, evaluation, the result store, the service layer (including live-model branches with fake engines), the sampling loop, the HTTP API and the Streamlit app. The JS suite covers phonetic matching, the lecture index, anchors and concept extraction.

## Project structure

```
├── server.py                  # Web studio entry point
├── app.py                     # Classic Streamlit UI
├── requirements.txt           # Real models
├── requirements-web.txt       # Offline studio only
├── configs/config.yaml
├── src/
│   ├── config.py              # Config + WWLLM_* overrides
│   ├── api/app.py             # Starlette routes, NDJSON streaming
│   ├── services/
│   │   ├── engines.py         # PipelineHub: stored results, models, fallbacks as event streams
│   │   ├── result_cache.py    # Model outputs stored by content
│   │   ├── capabilities.py    # What can run on this machine
│   │   ├── benchmark.py       # Evaluation reports
│   │   ├── files.py           # Upload store
│   │   └── sample.py          # Sample lecture
│   └── pipelines/
│       ├── audio_utils.py     # VAD ingestion and chunking
│       ├── asr_engine.py      # IndicConformer
│       ├── whisper_engine.py  # Open Whisper
│       ├── llm_engine.py      # Param-1 notes and quiz, token streaming
│       ├── translator.py      # IndicTrans2 and NLLB-200
│       ├── eval_pipeline.py   # WER, alignment, BERTScore, ROUGE
│       ├── fallback.py        # Offline VAD, notes, quiz
│       ├── quiz_schema.py     # MCQ normalisation
│       └── languages.py
├── web/                       # No build step: plain ES modules
│   ├── index.html
│   └── assets/
│       ├── css/app.css
│       ├── fonts/             # Bundled so the UI needs no network
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
│   ├── sample/                # Sample lecture, with stored model outputs in results/
│   └── ground_truth/          # 40-lecture benchmark: references and speech
├── scripts/
│   ├── generate_speech.py     # Text-to-speech for the sample and benchmark
│   ├── precompute_sample.py   # Real pipeline over the sample
│   ├── run_benchmark.py       # Real pipeline over the benchmark
│   └── run_eval.py            # Score any predictions
└── tests/
```

## Limits worth knowing

- **Param-1 is slow without a GPU.** The stored sample hides this. A new lecture takes minutes per stage on a laptop, and the UI says so.
- **Technical loanwords trip up Whisper.** "Learning rate" spoken in Hindi can come out as "जर्निंग रेट".
- **Param-1 adds general knowledge** the lecturer never said, so read its notes before trusting them.
- **NLLB-200 garbles technical terms.** "Gradient Descent" came out as "ग्रेडिएंट डाउनसेंस" in Hindi.
- **The quiz needs review.** Param-1's questions are on topic, but answer keys can be weak and some have no explanation.
- **You pick the spoken language.** There's no automatic detection, and the wrong choice produces nonsense.
- **IndicConformer and IndicTrans2 haven't been run here.** Their code paths are in place but need a Hugging Face login to test.
- **Word timing is estimated.** ASR gives timestamps per chunk, and words are interpolated by character position inside it. It is close enough to follow along, but it isn't forced alignment.
- **Phonetic matching finds loanwords.** Native vocabulary needs the notes bridge, and a false link is possible where unrelated words share a skeleton.
- **`.wav` only offline.** Decoding MP3 or MP4 needs `torchaudio` or `soundfile`, plus ffmpeg for video.

## Performance

Measured on the 98-second sample, on an Intel i5-1235U with 16 GB of RAM and no GPU:

| Stage | Model | Time |
|---|---|---|
| Speech detection | Silero-VAD | 7 s |
| Transcription | `vasista22/whisper-hindi-small` | 2.0 min |
| Notes, 371 words | Param-1 | 9.6 min |
| Quiz, 5 questions | Param-1 | 8.2 min |
| Translation, per language | NLLB-200 | 64 s |

Param-1 needs about 6 GB of RAM on CPU. With a GPU, 8-bit quantisation reduces that, and `llm.max_new_tokens_*` allow longer notes.

## License

For **academic and research purposes only**. Model weights are subject to their own licences. Meta's MMS voices and NLLB-200 are CC-BY-NC 4.0, so non-commercial use only.
