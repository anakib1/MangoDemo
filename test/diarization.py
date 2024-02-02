import unittest
import numpy as np
from mango.diarization import DummyDiarizer, DiarizationConfig, WhisperBasedDiarizationConfig, WhisperDiarizer, \
    EENDConfig, EENDDiarizer
from mango.utils.diarization import draw_diarization


class TestDiarization(unittest.TestCase):

    def verifyOutput(self, output, num_speakers):
        self.assertTrue((1500, num_speakers) == output.shape)
        self.assertTrue(isinstance(output, np.ndarray))
        self.assertTrue(np.all(output > 0.0))
        self.assertTrue(np.all(output < 1.0))

        picture = draw_diarization(output)
        self.assertEqual((400, 800, 4), picture.shape)

    def test_dummy(self):
        model = DummyDiarizer(DiarizationConfig(max_num_speakers=3))
        audio, sr = np.random.randn(16_000 * 30), 16_000
        output = model.diarize(audio, sr)

        self.verifyOutput(output, 3)

    def test_whisper(self):
        model = WhisperDiarizer(WhisperBasedDiarizationConfig(whisper_checkpoint='anakib1/whisper-diarization-0.2',
                                                              processor_checkpoint='openai/whisper-tiny',
                                                              max_num_speakers=3))
        audio, sr = np.random.randn(16_000 * 30), 16_000
        output = model.diarize(audio, sr)

        self.verifyOutput(output, 3)

    def test_eend(self):
        model = EENDDiarizer(EENDConfig(hf_api_model_path='anakib1/eend-sa',
                                        hf_api_model_name='model.pt', hf_api_processor_path='openai/whisper-small',
                                        max_num_speakers=2))
        audio, sr = np.random.randn(16_000 * 30), 16_000
        output = model.diarize(audio, sr)

        self.verifyOutput(output, 2)


if __name__ == '__main__':
    unittest.main()
