import unittest
from datasets import load_dataset
from mango.utils.tokenization import retain_cyrillic
from mango.training.DatasetMixer import DatasetMixerConfig, DatasetMixerWrapped
from tqdm.auto import tqdm

class MixetTest(unittest.TestCase):
    def test_mixer_empty(self):
        cv13 = load_dataset('mozilla-foundation/common_voice_13_0', 'uk', trust_remote_code=True)
        urban = load_dataset('danavery/urbansound8K')

        urban = urban['train'].rename_column('class', 'label')
        cv13 = cv13['train'].rename_columns({'sentence': 'transcription', 'client_id': 'speaker_id'})
        cv13 = retain_cyrillic(cv13, 'transcription')

        cv13.set_format('pt')
        urban.set_format('pt')
        mixer_train = DatasetMixerWrapped(
            DatasetMixerConfig(max_speakers=3, utterances_count=2500, beta=5, min_repetitions=3, max_repetitions=6), cv13,
            urban)

        for example in tqdm(mixer_train):
            self.assertTrue(len(example.transcription) > 0)



if __name__ == '__main__':
    unittest.main()
