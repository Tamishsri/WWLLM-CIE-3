"""Audio pipeline behaviour that needs the ML stack."""

from __future__ import annotations

import unittest

try:
    import torch
    import silero_vad  # noqa: F401
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "the ML stack is not installed")
class Chunks(unittest.TestCase):
    def test_every_stretch_of_speech_stays_its_own_chunk(self):
        from src.pipelines.audio_utils import AudioProcessor

        waveform = torch.arange(48_000).float()
        stamps = [{"start": 0, "end": 16_000}, {"start": 20_000, "end": 24_000}]
        chunks, spans = AudioProcessor._cut_segments(waveform, stamps)
        self.assertEqual([len(c) for c in chunks], [16_000, 4_000])
        self.assertEqual(spans, [(0.0, 1.0), (1.25, 1.5)])

    def test_no_speech_keeps_the_whole_recording(self):
        from src.pipelines.audio_utils import AudioProcessor

        chunks, spans = AudioProcessor._cut_segments(torch.zeros(16_000), [])
        self.assertEqual(len(chunks), 1)
        self.assertEqual(spans, [(0.0, 1.0)])


@unittest.skipIf(torch is None, "the ML stack is not installed")
class Threads(unittest.TestCase):
    def test_importing_silero_leaves_models_their_threads(self):
        before = torch.get_num_threads()
        torch.set_num_threads(4)
        try:
            import importlib

            import src.pipelines.audio_utils as audio_utils

            importlib.reload(audio_utils)
            self.assertEqual(torch.get_num_threads(), 4)
        finally:
            torch.set_num_threads(before)


if __name__ == "__main__":
    unittest.main()
