import unittest

import numpy as np

from mango.classification import BaseClassifierConfig, DummyClassifier, WhisperClassifier, WhisperClassifierConfig


class TestClassification(unittest.TestCase):
    def verifyOutput(self, output: dict, id2label: dict):
        self.assertTrue(isinstance(output, dict))
        self.assertEqual(len(id2label), len(output))

        keys = output.keys()
        for x in keys:
            self.assertTrue(x in id2label.values())

        vals = output.values()
        for x in vals:
            self.assertTrue(isinstance(x, float) or isinstance(x, np.floating))
            self.assertTrue(0 <= x <= 1)

        self.assertTrue(abs(np.sum(list(vals)) - 1.0) < 1e-5)

    def test_dummy(self):
        dummy2label = {0: 'label1', 1: 'label2', 2: 'label3'}
        model = DummyClassifier(BaseClassifierConfig(dummy2label))
        audio, sr = np.random.randn(100_000), 16_000
        output = model.classify(audio, sr)

        self.verifyOutput(output, dummy2label)

    def test_whisper(self):
        config = WhisperClassifierConfig(whisper_checkpoint='anakib1/whisper-tiny-urban',
                                         processor_checkpoint='openai/whisper-tiny')
        model = WhisperClassifier(config)

        self.assertTrue(model.config.id2label is not None)
        audio, sr = np.random.randn(100_000), 16_000
        output = model.classify(audio, sr)

        self.verifyOutput(output, config.id2label)


if __name__ == '__main__':
    unittest.main()
