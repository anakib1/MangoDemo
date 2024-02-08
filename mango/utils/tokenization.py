from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2Processor, PreTrainedTokenizerFast
import json
import datasets
import re
from ..training.DatasetMixer import DatasetMixer
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)


def prepare_for_wav2vec2(repo_id: str, dataset: datasets.Dataset, text_column: str) -> None:
    """
    Prepares processors for wav2vec2 using the text in provided dataset.
    :param repo_id: resulting repo id
    :param dataset: input dataset
    :param text_column: text column name
    :return: None
    """
    vocab = set()
    for trans in dataset[text_column]:
        vocab |= set(trans)
    vocab_dict = {v: k for k, v in enumerate(sorted(list(vocab)))}
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    with open('vocab.json', 'w', encoding='UTF-8') as vocab_file:
        json.dump(vocab_dict, vocab_file)

    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("./", unk_token="[UNK]", pad_token="[PAD]",
                                                     word_delimiter_token=" ")
    tokenizer.push_to_hub(repo_id)

    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                                 do_normalize=True, return_attention_mask=True)

    feature_extractor.push_to_hub(repo_id)

    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    processor.push_to_hub(repo_id)


def retain_cyrillic(dataset: datasets.Dataset, text_column: str) -> datasets.Dataset:
    def remove_special_characters(*args, **kwargs):
        argument = args[0] if len(args) != 0 else next(iter(kwargs.values()))
        return {text_column: ' '.join(re.sub(
            r'[^а-яА-ЯіїєґІЇЄҐ\s-]|—|\d+|_|(?:\s|^)[-]|[-](?:\s|$)|[-«»]|\n', '',
            argument.lower()).split())}

    def filter_out_empty_sequences(*args, **kwargs):
        argument = args[0] if len(args) != 0 else next(iter(kwargs.values()))
        return len(argument) > 10

    return (dataset.map(remove_special_characters, batched=False, input_columns=[text_column])
            .filter(filter_out_empty_sequences, batched=False, input_columns=[text_column]))


def create_tokenizer(mixer: DatasetMixer, num_training_samples=25_000, vocab_size=5_000, max_length=256):
    tokenizer = Tokenizer(models.WordPiece(unk_token='<unk>'))
    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordPieceTrainer(vocab_size=vocab_size, special_tokens=['<pad>', '<unk>', '<sc>'])

    def get_text_iterator():
        for i in range(num_training_samples):
            yield mixer.generate().transcription

    tokenizer.train_from_iterator(get_text_iterator(), trainer=trainer)
    tokenizer.enable_padding(length=max_length)
    tokenizer.enable_truncation(max_length=max_length)
    wrapped_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='<unk>', pad_token='<pad>',
                                                sc_token='<sc>')
    wrapped_tokenizer.push_to_hub()
