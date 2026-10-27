"""
app.py — Streamlit Entry Point
=============================================================================
WWLLM: Multilingual Lecture-to-Notes & Quiz Generator

Launch:  streamlit run app.py

Architecture:
    1. Audio upload → VAD chunking (``AudioProcessor``)
    2. ASR transcription (``ASREngine`` — IndicConformer)
    3. LLM summarisation + quiz generation (``LLMEngine`` — BharatGen Param-1)
    4. Translation into Hindi / Tamil / Kannada (``TranslationEngine`` — IndicTrans2)

Session State Keys (prevent re-running ML models on UI interactions):
    • ``audio_chunks``    — list[torch.Tensor]  from VAD
    • ``transcript``      — str                 from ASR
    • ``notes_md``        — str                 from LLM
    • ``quiz_json``       — list[dict]           from LLM
    • ``translated_notes``— str                 from IndicTrans2
    • ``translated_quiz`` — list[dict]           from IndicTrans2
    • ``models_loaded``   — bool                flag
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
import yaml

# ── Local imports ──────────────────────────────────────────────────────────
from src.ui.styles import inject_css

# ── Page configuration (MUST be the first Streamlit call) ──────────────────
st.set_page_config(
    page_title="WWLLM — Lecture-to-Notes & Quiz",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject custom dark-theme CSS ──────────────────────────────────────────
inject_css()


# ===========================================================================
# Configuration loader
# ===========================================================================
@st.cache_data
def load_config() -> Dict[str, Any]:
    """Load ``configs/config.yaml`` once and cache across sessions."""
    config_path = Path("configs/config.yaml")
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


CFG = load_config()


# ===========================================================================
# Lazy model loading — cached so Streamlit re-runs don't re-load weights
# ===========================================================================
@st.cache_resource(show_spinner="Loading Audio Processor (Silero-VAD)…")
def get_audio_processor():
    """Instantiate AudioProcessor (Silero-VAD). Cached as a singleton."""
    from src.pipelines.audio_utils import AudioProcessor

    audio_cfg = CFG.get("audio", {})
    return AudioProcessor(
        max_chunk_sec=audio_cfg.get("max_chunk_sec", 25.0),
        min_chunk_sec=audio_cfg.get("min_chunk_sec", 2.0),
        vad_threshold=audio_cfg.get("vad_threshold", 0.45),
        min_silence_ms=audio_cfg.get("min_silence_ms", 300),
    )


@st.cache_resource(show_spinner="Loading IndicConformer ASR model…")
def get_asr_engine():
    """Instantiate ASREngine (IndicConformer). Cached as a singleton."""
    from src.pipelines.asr_engine import ASREngine

    asr_cfg = CFG.get("asr", {})
    return ASREngine(
        model_id=asr_cfg.get(
            "model_id", "ai4bharat/indicconformer_stt_hi_hybrid_rnnt_large"
        ),
        device=asr_cfg.get("device"),
        batch_size=asr_cfg.get("batch_size", 8),
    )


@st.cache_resource(show_spinner="Loading BharatGen Param-1 LLM (8-bit)…")
def get_llm_engine():
    """Instantiate LLMEngine (BharatGen Param-1). Cached as a singleton."""
    from src.pipelines.llm_engine import LLMEngine

    llm_cfg = CFG.get("llm", {})
    return LLMEngine(
        model_id=llm_cfg.get("model_id", "bharatgenai/Param-1-2.9B-Instruct"),
        quantize_8bit=llm_cfg.get("quantize_8bit", True),
        temperature=llm_cfg.get("temperature", 0.4),
    )


@st.cache_resource(show_spinner="Loading IndicTrans2 Translation model…")
def get_translation_engine():
    """Instantiate TranslationEngine (IndicTrans2). Cached as a singleton."""
    from src.pipelines.translator import TranslationEngine

    trans_cfg = CFG.get("translation", {})
    return TranslationEngine(
        model_id=trans_cfg.get("model_id", "ai4bharat/indictrans2-en-indic-1B"),
        num_beams=trans_cfg.get("num_beams", 5),
    )


# ===========================================================================
# Session state initialisation
# ===========================================================================
_DEFAULTS: Dict[str, Any] = {
    "audio_chunks": None,
    "transcript": None,
    "notes_md": None,
    "quiz_json": None,
    "translated_notes": None,
    "translated_quiz": None,
    "upload_key": 0,
    "current_file_name": None,
}

for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


def _reset_downstream(from_stage: str = "all") -> None:
    """Clear session state downstream of a given pipeline stage.

    Prevents stale results from being displayed when a new file is
    uploaded or a parameter changes.

    Parameters
    ----------
    from_stage : str
        ``"all"`` | ``"transcript"`` | ``"notes"`` | ``"translation"``
    """
    stages = {
        "all": [
            "audio_chunks",
            "transcript",
            "notes_md",
            "quiz_json",
            "translated_notes",
            "translated_quiz",
        ],
        "transcript": [
            "transcript",
            "notes_md",
            "quiz_json",
            "translated_notes",
            "translated_quiz",
        ],
        "notes": ["notes_md", "quiz_json", "translated_notes", "translated_quiz"],
        "translation": ["translated_notes", "translated_quiz"],
    }
    for key in stages.get(from_stage, stages["all"]):
        st.session_state[key] = None


# ===========================================================================
# Helper — render quiz cards
# ===========================================================================
def render_quiz(quiz: List[Dict[str, Any]], show_answers: bool = True) -> None:
    """Render MCQ cards with styled HTML.

    Parameters
    ----------
    quiz : List[Dict[str, Any]]
        List of MCQ dicts from LLMEngine or TranslationEngine.
    show_answers : bool
        Whether to highlight correct answers.
    """
    for i, q in enumerate(quiz, 1):
        options_html = ""
        for opt in q.get("options", []):
            is_correct = (
                opt.strip() == q.get("correct_answer", "").strip()
                if show_answers
                else False
            )
            cls = "quiz-option correct" if is_correct else "quiz-option"
            marker = " ✅" if is_correct else ""
            options_html += f'<div class="{cls}">{opt}{marker}</div>'

        explanation_html = ""
        if show_answers and q.get("explanation"):
            explanation_html = (
                f'<div class="quiz-explanation">💡 {q["explanation"]}</div>'
            )

        card_html = f"""
        <div class="quiz-card">
            <h4>Q{i}. {q.get("question", "")}</h4>
            {options_html}
            {explanation_html}
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)


# ===========================================================================
# SIDEBAR
# ===========================================================================
with st.sidebar:
    st.markdown("## 🎓 WWLLM")
    st.markdown(
        '<p style="color:#8B949E; font-size:0.85rem;">'
        "Multilingual Lecture-to-Notes & Quiz Generator</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Language selection ─────────────────────────────────────────────
    st.markdown("### 🌐 Target Language")
    target_lang = st.selectbox(
        "Translate output to:",
        options=["Hindi", "Tamil", "Kannada"],
        index=0,
        label_visibility="collapsed",
    )

    # ── ASR language ───────────────────────────────────────────────────
    st.markdown("### 🎙️ Source Language (ASR)")
    asr_lang = st.selectbox(
        "Source audio language:",
        options=["Hindi", "English", "Tamil", "Kannada", "Telugu", "Malayalam"],
        index=0,
        label_visibility="collapsed",
    )

    st.divider()

    # ── Pipeline controls ──────────────────────────────────────────────
    st.markdown("### ⚙️ Pipeline Settings")
    num_questions = st.slider("Number of MCQs", min_value=3, max_value=10, value=5)
    show_answers = st.checkbox("Show correct answers", value=True)

    st.divider()

    # ── System info ────────────────────────────────────────────────────
    st.markdown("### 📊 System")
    import torch

    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "N/A"
    st.markdown(
        f"- **GPU:** {'✅ ' + gpu_name if gpu_available else '❌ CPU only'}\n"
        f"- **CUDA:** {torch.version.cuda if gpu_available else 'N/A'}\n"
        f"- **PyTorch:** {torch.__version__}"
    )


# ===========================================================================
# MAIN CONTENT
# ===========================================================================
st.markdown("# 🎓 WWLLM — Lecture-to-Notes & Quiz Generator")
st.markdown(
    '<p style="color:#8B949E; margin-top:-0.8rem;">'
    "Upload a lecture recording → get structured notes, MCQ quizzes, "
    "and translations powered by AI4Bharat models.</p>",
    unsafe_allow_html=True,
)

# ── File upload ────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload lecture audio/video",
    type=["wav", "mp3", "mp4", "flac", "ogg", "mkv"],
    help="Accepted formats: WAV, MP3, MP4, FLAC, OGG, MKV",
    key=f"uploader_{st.session_state.upload_key}",
)

# Detect new file upload → reset downstream state
if uploaded_file is not None:
    if uploaded_file.name != st.session_state.current_file_name:
        _reset_downstream("all")
        st.session_state.current_file_name = uploaded_file.name

# ── Pipeline tabs ──────────────────────────────────────────────────────────
tab_transcribe, tab_notes, tab_quiz, tab_translate, tab_eval = st.tabs(
    ["📝 Transcribe", "📒 Notes", "❓ Quiz", "🌐 Translate", "📊 Evaluate"]
)

# ===========================================================================
# TAB 1: Transcription (VAD → ASR)
# ===========================================================================
with tab_transcribe:
    st.markdown("## 📝 Automatic Transcription")

    if uploaded_file is None:
        st.info("⬆️  Upload a lecture file to begin.", icon="📂")
    else:
        st.audio(uploaded_file)

        if st.session_state.transcript is not None:
            st.success("Transcription cached — no recomputation needed.", icon="✅")
            st.text_area(
                "Transcript",
                value=st.session_state.transcript,
                height=300,
                disabled=True,
            )
        else:
            if st.button("🚀 Run Transcription", key="btn_transcribe", type="primary"):
                # ── Stage 1: VAD chunking ──────────────────────────────
                with st.status("🔊 Processing audio with VAD…", expanded=True) as status:
                    st.write("Loading Audio Processor…")
                    audio_proc = get_audio_processor()

                    st.write(f"Chunking `{uploaded_file.name}` …")
                    t0 = time.perf_counter()
                    chunks = audio_proc.process_uploaded_bytes(
                        uploaded_file.getvalue(), uploaded_file.name
                    )
                    vad_time = time.perf_counter() - t0
                    st.session_state.audio_chunks = chunks
                    status.update(
                        label=f"✅ VAD complete — {len(chunks)} chunks in {vad_time:.1f}s",
                        state="complete",
                    )

                # ── Stage 2: ASR ───────────────────────────────────────
                with st.status("🎙️ Running IndicConformer ASR…", expanded=True) as status:
                    st.write("Loading ASR model…")
                    asr = get_asr_engine()

                    st.write(f"Transcribing {len(chunks)} chunks…")
                    progress_bar = st.progress(0.0)
                    t0 = time.perf_counter()
                    transcripts = asr.transcribe(chunks)
                    asr_time = time.perf_counter() - t0
                    progress_bar.progress(1.0)

                    full_transcript = " ".join(t for t in transcripts if t.strip())
                    st.session_state.transcript = full_transcript
                    status.update(
                        label=f"✅ ASR complete in {asr_time:.1f}s",
                        state="complete",
                    )

                # ── Display result ─────────────────────────────────────
                col1, col2, col3 = st.columns(3)
                col1.metric("Chunks", len(chunks))
                col2.metric("VAD Time", f"{vad_time:.1f}s")
                col3.metric("ASR Time", f"{asr_time:.1f}s")

                st.text_area(
                    "Transcript",
                    value=full_transcript,
                    height=300,
                    disabled=True,
                )

# ===========================================================================
# TAB 2: Lecture Notes (LLM)
# ===========================================================================
with tab_notes:
    st.markdown("## 📒 Structured Lecture Notes")

    if st.session_state.transcript is None:
        st.info("Run transcription first (Tab 1).", icon="📝")
    elif st.session_state.notes_md is not None:
        st.success("Notes cached.", icon="✅")
        st.markdown(st.session_state.notes_md)
        st.download_button(
            "📥 Download Notes (.md)",
            data=st.session_state.notes_md,
            file_name="lecture_notes.md",
            mime="text/markdown",
        )
    else:
        if st.button("🧠 Generate Notes", key="btn_notes", type="primary"):
            with st.status("🧠 BharatGen Param-1 generating notes…", expanded=True) as status:
                st.write("Loading LLM (8-bit quantised)…")
                llm = get_llm_engine()

                st.write("Generating structured notes…")
                t0 = time.perf_counter()
                notes = llm.generate_notes(st.session_state.transcript)
                gen_time = time.perf_counter() - t0

                st.session_state.notes_md = notes
                status.update(
                    label=f"✅ Notes generated in {gen_time:.1f}s",
                    state="complete",
                )

            st.markdown(notes)
            st.download_button(
                "📥 Download Notes (.md)",
                data=notes,
                file_name="lecture_notes.md",
                mime="text/markdown",
            )

# ===========================================================================
# TAB 3: MCQ Quiz (LLM)
# ===========================================================================
with tab_quiz:
    st.markdown("## ❓ Multiple-Choice Quiz")

    if st.session_state.transcript is None:
        st.info("Run transcription first (Tab 1).", icon="📝")
    elif st.session_state.quiz_json is not None:
        st.success(
            f"Quiz cached — {len(st.session_state.quiz_json)} questions.", icon="✅"
        )
        render_quiz(st.session_state.quiz_json, show_answers=show_answers)
        st.download_button(
            "📥 Download Quiz (.json)",
            data=json.dumps(st.session_state.quiz_json, indent=2, ensure_ascii=False),
            file_name="lecture_quiz.json",
            mime="application/json",
        )
    else:
        if st.button("🎯 Generate Quiz", key="btn_quiz", type="primary"):
            with st.status("🎯 Generating MCQs…", expanded=True) as status:
                st.write("Loading LLM…")
                llm = get_llm_engine()

                st.write(f"Generating {num_questions} MCQs…")
                t0 = time.perf_counter()
                quiz = llm.generate_quiz(
                    st.session_state.transcript, num_questions=num_questions
                )
                gen_time = time.perf_counter() - t0
                st.session_state.quiz_json = quiz
                status.update(
                    label=f"✅ {len(quiz)} MCQs generated in {gen_time:.1f}s",
                    state="complete",
                )

            if quiz:
                render_quiz(quiz, show_answers=show_answers)
                st.download_button(
                    "📥 Download Quiz (.json)",
                    data=json.dumps(quiz, indent=2, ensure_ascii=False),
                    file_name="lecture_quiz.json",
                    mime="application/json",
                )
            else:
                st.warning("LLM did not produce valid MCQ JSON. Try again.", icon="⚠️")

# ===========================================================================
# TAB 4: Translation (IndicTrans2)
# ===========================================================================
with tab_translate:
    st.markdown("## 🌐 Translation")

    has_content = (
        st.session_state.notes_md is not None
        or st.session_state.quiz_json is not None
    )

    if not has_content:
        st.info("Generate notes or quiz first (Tabs 2-3).", icon="📒")
    else:
        from src.pipelines.translator import SUPPORTED_TARGETS

        flores_code = SUPPORTED_TARGETS.get(target_lang, "hin_Deva")
        st.markdown(f"**Target:** {target_lang} (`{flores_code}`)")

        # Check cache
        if st.session_state.translated_notes is not None:
            st.success("Translation cached.", icon="✅")
        else:
            if st.button(
                f"🌐 Translate to {target_lang}",
                key="btn_translate",
                type="primary",
            ):
                with st.status(
                    f"🌐 Translating to {target_lang}…", expanded=True
                ) as status:
                    st.write("Loading IndicTrans2…")
                    translator = get_translation_engine()
                    t0 = time.perf_counter()

                    # Translate notes
                    if st.session_state.notes_md:
                        st.write("Translating notes…")
                        st.session_state.translated_notes = translator.translate(
                            st.session_state.notes_md, target_lang=flores_code
                        )

                    # Translate quiz
                    if st.session_state.quiz_json:
                        st.write("Translating quiz…")
                        st.session_state.translated_quiz = translator.translate_quiz(
                            st.session_state.quiz_json, target_lang=flores_code
                        )

                    trans_time = time.perf_counter() - t0
                    status.update(
                        label=f"✅ Translation complete in {trans_time:.1f}s",
                        state="complete",
                    )

        # Display translated content
        if st.session_state.translated_notes:
            st.markdown("### 📒 Translated Notes")
            st.markdown(st.session_state.translated_notes)
            st.download_button(
                f"📥 Download {target_lang} Notes (.md)",
                data=st.session_state.translated_notes,
                file_name=f"notes_{target_lang.lower()}.md",
                mime="text/markdown",
            )

        if st.session_state.translated_quiz:
            st.markdown("### ❓ Translated Quiz")
            render_quiz(st.session_state.translated_quiz, show_answers=show_answers)

# ===========================================================================
# TAB 5: Evaluation
# ===========================================================================
with tab_eval:
    st.markdown("## 📊 Evaluation Suite")
    st.markdown(
        "Upload a ground-truth JSON file to evaluate ASR accuracy (WER) "
        "and summarisation quality (BERTScore, ROUGE)."
    )

    gt_file = st.file_uploader(
        "Upload ground-truth JSON",
        type=["json"],
        key="gt_uploader",
        help=(
            "JSON array of objects with keys: "
            '"audio_file", "transcript", and optionally "summary".'
        ),
    )

    if gt_file is not None:
        if st.button("📊 Run Evaluation", key="btn_eval", type="primary"):
            from src.pipelines.eval_pipeline import EvalPipeline

            with st.status("📊 Running evaluation…", expanded=True) as status:
                import tempfile

                # Write GT to temp file for the evaluator
                gt_path = Path(tempfile.mktemp(suffix=".json"))
                gt_path.write_bytes(gt_file.getvalue())

                evaluator = EvalPipeline(
                    output_dir=CFG.get("evaluation", {}).get(
                        "output_dir", "outputs/evaluations"
                    )
                )
                results = evaluator.run(gt_path)
                df = evaluator.generate_report_df(results)
                gt_path.unlink(missing_ok=True)

                status.update(label="✅ Evaluation complete", state="complete")

            # Display results
            st.dataframe(df, use_container_width=True, hide_index=True)

            failures = df[df["Failure"] == "FAIL"]
            col1, col2, col3 = st.columns(3)
            col1.metric("Samples", len(df))
            col2.metric("Avg WER", f"{df['WER'].mean():.3f}")
            col3.metric("Failures", len(failures))

            if not failures.empty:
                st.warning(
                    f"{len(failures)} sample(s) flagged (WER > 0.30) — logged to "
                    "`outputs/evaluations/failure_analysis.csv`.",
                    icon="⚠️",
                )

# ===========================================================================
# Footer
# ===========================================================================
st.markdown(
    '<div class="footer-text">'
    "WWLLM v1.0 — Built with AI4Bharat IndicConformer · BharatGen Param-1 · IndicTrans2"
    "</div>",
    unsafe_allow_html=True,
)
