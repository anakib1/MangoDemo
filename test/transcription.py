import unittest
import numpy as np
from mango.transcription import DummyTranscriptor, WhisperTranscriptionConfig, WhisperTranscriptor


class TestTranscription(unittest.TestCase):

    def test_dummy(self):
        model = DummyTranscriptor()
        audio, sr = np.random.randn(100_000), 16_000
        output = model.transcribe(audio, sr)

        self.assertTrue(isinstance(output, str))

    def test_whisper(self):
        config = WhisperTranscriptionConfig(whisper_checkpoint='anakib1/whisper-small-uk',
                                            processor_checkpoint='openai/whisper-small')
        model = WhisperTranscriptor(config)
        audio, sr = np.random.randn(100_000), 16_000
        output = model.transcribe(audio, sr)

        self.assertTrue(isinstance(output, str))


if __name__ == '__main__':
    unittest.main()
