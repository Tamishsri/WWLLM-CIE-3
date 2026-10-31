"""LLMEngine's sampling loop, driven by a stand-in model (needs torch + transformers)."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

try:
    import torch

    from src.pipelines.llm_engine import LLMEngine
except ImportError:
    torch = None

EOS = 9
PROMPT = [1, 2, 3]


class FakeModel:
    """Always picks the next token of *script*, and records what it was fed."""

    device = "cpu"

    def __init__(self, script):
        self.script = iter(script)
        self.calls = []

    def __call__(self, input_ids, past_key_values, use_cache, return_dict):
        self.calls.append((input_ids.shape[1], past_key_values))
        logits = torch.full((1, input_ids.shape[1], 10), -100.0)
        logits[0, -1, next(self.script)] = 100.0
        return SimpleNamespace(logits=logits, past_key_values=("cache",))


class FakeTokenizer:
    eos_token_id = EOS
    pieces = {4: "a", 5: "b", 6: "�", 7: "c"}

    def apply_chat_template(self, conversation, return_tensors, add_generation_prompt):
        return torch.tensor([PROMPT])

    def __call__(self, text, add_special_tokens, return_tensors):
        return SimpleNamespace(input_ids=torch.tensor([[8]]))

    def decode(self, ids, skip_special_tokens):
        text = "".join(self.pieces.get(i, "") for i in ids)
        return text.replace("�c", "é")  # 6 + 7 are one character split in two


def engine(script) -> "LLMEngine":
    llm = LLMEngine.__new__(LLMEngine)
    llm.model = FakeModel(script)
    llm.tokenizer = FakeTokenizer()
    llm.temperature = 0.4
    llm.max_new_tokens_quiz = 100
    return llm


MESSAGES = [{"role": "user", "content": "hi"}]


@unittest.skipIf(torch is None, "the ML stack is not installed")
class Sampling(unittest.TestCase):
    def test_the_prompt_runs_once_then_one_token_at_a_time_with_the_cache(self):
        llm = engine([4, 5, EOS])
        self.assertEqual(llm._generate(MESSAGES, 10), "ab")
        self.assertEqual(llm.model.calls, [(3, None), (1, ("cache",)), (1, ("cache",))])

    def test_stops_at_the_token_limit(self):
        llm = engine([4] * 50)
        self.assertEqual(llm._generate(MESSAGES, 3), "aaa")
        self.assertEqual(len(llm.model.calls), 3)

    def test_a_split_character_waits_for_its_other_half(self):
        pieces = list(engine([4, 6, 7, EOS])._generate_stream(MESSAGES, 10))
        self.assertEqual(pieces, ["a", "é"])

    def test_closing_the_stream_stops_the_model(self):
        llm = engine([4] * 50)
        stream = llm._generate_stream(MESSAGES, 50)
        next(stream)
        stream.close()
        self.assertEqual(len(llm.model.calls), 1)

    def test_rotary_tables_are_rebuilt_after_loading(self):
        class Rotary(torch.nn.Module):
            dim, base, max_seq_len_cached = 4, 10000, 8

            def __init__(self):
                super().__init__()
                self.register_buffer("inv_freq", torch.full((2,), 1e18), persistent=False)

            def _set_cos_sin_cache(self, seq_len, device, dtype):
                self.rebuilt = (seq_len, dtype)

        llm = engine([])
        llm.model = torch.nn.Sequential(Rotary())
        llm._rebuild_rotary_tables()
        rotary = llm.model[0]
        self.assertEqual(rotary.inv_freq[0].item(), 1.0)
        self.assertAlmostEqual(rotary.inv_freq[1].item(), 0.01, places=5)
        self.assertEqual(rotary.rebuilt, (8, torch.float32))

    def test_each_question_is_started_for_the_model_without_rereading_the_prompt(self):
        llm = engine([4, EOS, 5, EOS, 7, EOS, 5, EOS])
        llm._truncate_to_tokens = lambda text, max_tokens: text
        pieces = list(llm.stream_quiz("transcript", 2))
        self.assertEqual(pieces, ["Q1:", "a", "\nWhy:", "b", "\n\nQ2:", "c", "\nWhy:", "b"])
        reads = [length for length, _ in llm.model.calls]
        self.assertEqual(reads[0], len(PROMPT) + 1)  # the prompt and the first opening
        self.assertTrue(all(n == 1 for n in reads[1:]))
        self.assertTrue(all(past == ("cache",) for _, past in llm.model.calls[1:]))

    def test_the_plain_quiz_format_parses_into_the_canonical_schema(self):
        raw = (
            "Q: What sets the step size?\nA) Batch\nB) Learning rate\nC) Loss\nD) Epoch\n"
            "Answer: B\nWhy: It scales each update.\n\n"
            "Q: Which uses one sample?\nA) Batch\nB) Mini-batch\nC) Stochastic\nD) None\n"
            "Answer: C) Stochastic\n"
        )
        quiz = engine([]).parse_quiz(raw)
        self.assertEqual(len(quiz), 2)
        self.assertEqual(quiz[0]["options"], ["Batch", "Learning rate", "Loss", "Epoch"])
        self.assertEqual((quiz[0]["answer_index"], quiz[0]["explanation"]), (1, "It scales each update."))
        self.assertEqual(quiz[1]["answer_index"], 2)

    def test_a_question_cut_off_by_the_token_limit_is_dropped(self):
        raw = "Q: One?\nA) x\nB) y\nAnswer: A\n\nQ: Two?\nA) x\nB"
        self.assertEqual(len(engine([]).parse_quiz(raw)), 1)

    def test_json_replies_still_work(self):
        question = {"question": "Q?", "options": ["A) x", "B) y"], "correct_answer": "B) y"}
        self.assertEqual(engine([]).parse_quiz(json.dumps([question]))[0]["answer_index"], 1)

    def test_questions_are_counted_as_they_start(self):
        self.assertEqual(LLMEngine.count_questions("Q: a\nAnswer: B\n\nQ: b"), 2)


if __name__ == "__main__":
    unittest.main()
