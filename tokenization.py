import utils

import pickle
import tokenizers
from tokenizers import Tokenizer
from tokenizers.normalizers import StripAccents
from tokenizers.pre_tokenizers import Sequence, WhitespaceSplit, Punctuation
from tokenizers.processors import TemplateProcessing, ByteLevel
from tokenizers.trainers import WordLevelTrainer, BpeTrainer, WordPieceTrainer
from tokenizers.models import WordLevel, BPE, WordPiece
import numpy as np


def load_tokenizer(filename):
    with open(filename, 'rb') as infile:
        tokenizer = pickle.load(infile)
    return tokenizer


def encodings_normalize_length(encodings, target_length, padding_token='[PAD]'):
    """
    Takes list of tokenizers.Encoding and truncates / right pads it to the desired length.
    In place.
    """
    for index in range(len(encodings)):
        encodings[index].truncate(target_length)
        encodings[index].pad(length=target_length, direction="right", pad_token=padding_token)


def encodings_get_length_greater_than(encodings, percentage):
    """
    Takes encodings list as parameter and
    returns a length of encoding that is bigger than
    than <percentage>% of entries in the list.
    """
    encoding_lengths = list(map(len, encodings))
    percentile_value = np.percentile(encoding_lengths, percentage)

    return int(percentile_value)


if __name__ == '__main__':
    TRAIN_DATA_PATH = "data/train_data.txt"
    TRAIN_DATA_STRATIFIED_PATH = "data/train_data_augmented_1500.txt"

    df_train = utils.load_data(TRAIN_DATA_STRATIFIED_PATH)
    # make everything lower case and remove trailing whitespaces
    df_train['description'] = df_train['description'].apply(lambda x: x.lower().strip())
    df_train['genre'] = df_train['genre'].apply(lambda x: x.lower().strip())



    tokenizerWP = Tokenizer(WordPiece(unk_token='[UNK]'))
    # train tokenizer
    trainer = WordPieceTrainer(special_tokens=['[UNK]', '[CLS]', '[SEP]', '[PAD]'],
                               continuing_subword_prefix='##',
                               vocab_size=10000)
    tokenizerWP.normalizer = StripAccents()  # add method for stripping accents
    tokenizerWP.pre_tokenizer = Sequence([WhitespaceSplit(), Punctuation(behavior='removed')])
    tokenizerWP.post_processor = TemplateProcessing(single='[CLS] $0 [SEP]',
                                                    special_tokens=[('[CLS]', 1), ('[SEP]', 2)])
    tokenizerWP.train_from_iterator(df_train['description'], trainer=trainer, )

    with open("tokenizers/tokenizerWPaugmented.pickle", 'wb') as handle:
        pickle.dump(tokenizerWP, handle, protocol=3)

    tokenizerWP = load_tokenizer("tokenizers/tokenizerWPaugmented.pickle")

    vocabSize = tokenizerWP.get_vocab_size()
    print('size of vocabulary: {}'.format(vocabSize))
    for i in range(10):
        print('vocabulary id: {0}, word: {1}'.format(i, tokenizerWP.id_to_token(i)))
        j = vocabSize - i - 1
        print('vocabulary id: {0}, word: {1}'.format(j, tokenizerWP.id_to_token(j)))

    tokenizerBPE = Tokenizer(BPE(unk_token='[UNK]', dropout=None))
    # train tokenizer
    trainer = BpeTrainer(special_tokens=['[UNK]', '[CLS]', '[SEP]', '[PAD]'],
                         continuing_subword_prefix='##',
                         vocab_size=10000)
    tokenizerBPE.normalizer = StripAccents()  # add method for stripping accents
    tokenizerBPE.pre_tokenizer = Sequence([WhitespaceSplit(), Punctuation(behavior='removed')])
    tokenizerBPE.post_processor = ByteLevel(trim_offsets=True)
    tokenizerBPE.train_from_iterator(df_train['description'], trainer=trainer)

    with open("tokenizers/tokenizerBPEaugmented.pickle", 'wb') as handle:
        pickle.dump(tokenizerBPE, handle, protocol=3)

    tokenizerBPE = load_tokenizer("tokenizers/tokenizerBPEaugmented.pickle")

    vocabSize = tokenizerBPE.get_vocab_size()
    print('size of vocabulary: {}'.format(vocabSize))
    for i in range(10):
        print('vocabulary id: {0}, word: {1}'.format(i, tokenizerBPE.id_to_token(i)))
        j = vocabSize - i - 1
        print('vocabulary id: {0}, word: {1}'.format(j, tokenizerBPE.id_to_token(j)))


