from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2Processor
import json
import datasets
import re


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
    def remove_special_characters(batch):
        return {text_column: ' '.join(re.sub(
            r'[^а-яА-ЯіїєґІЇЄҐ\s-]|—|\d+|_|(?:\s|^)[-]|[-](?:\s|$)|[-«»]|\n', '',
            batch[text_column].lower()).split())}

    return dataset.map(remove_special_characters, batched=False)
