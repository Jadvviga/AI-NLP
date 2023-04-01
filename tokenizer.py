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
    with open(filename, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

if __name__ == '__main__':
    TEST_SOLUTION_DATA_PATH = "data/test_data_solution.txt"
    TRAIN_DATA_PATH = "data/train_data.txt"

    df_train = utils.load_data(TRAIN_DATA_PATH)
    # make everything lower case and remove trailing whitespaces
    df_train['description'] = df_train['description'].apply(lambda x: x.lower().strip())
    df_train['genre'] = df_train['genre'].apply(lambda x: x.lower().strip())

    df_test = utils.load_data(TEST_SOLUTION_DATA_PATH)
    # make everything lower case and remove trailing whitespaces
    df_test['description'] = df_test['description'].apply(lambda x: x.lower().strip())
    df_test['genre'] = df_test['genre'].apply(lambda x: x.lower().strip())


    """ 
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

    with open('tokenizers/tokenizerWP.pickle', 'wb') as handle:
        pickle.dump(tokenizerWP, handle, protocol=3)

    tokenizerWP = load_tokenizer("tokenizers/tokenizerWP.pickle")

    vocabSize = tokenizerWP.get_vocab_size()
    print('size of vocabulary: {}'.format(vocabSize))
    for i in range(10):
        print('vocabulary id: {0}, word: {1}'.format(i, tokenizerWP.id_to_token(i)))
        j = vocabSize - i - 1
        print('vocabulary id: {0}, word: {1}'.format(j, tokenizerWP.id_to_token(j)))

    print(df_test.loc[0, 'description'])
    out = tokenizerWP.encode(df_test.loc[0, 'description'])
    print(out.ids)
    print(out.tokens)
    print(tokenizerWP.decode(out.ids))

    outFull = tokenizerWP.encode_batch(df_test['description'])
    ntokens = 0
    nunk = 0
    for encoded in outFull:
        ntokens += len(encoded.ids)
        nunk += len(encoded.ids) - np.count_nonzero(encoded.ids)
    print('ratio of unknown tokens: {0:.4f}'.format(nunk / ntokens))
    print('total number of tokens: {}'.format(ntokens))
    """
    tokenizerBPE = Tokenizer(BPE(unk_token='[UNK]', dropout=None))
    # train tokenizer
    trainer = BpeTrainer(special_tokens=['[UNK]', '[CLS]', '[SEP]', '[PAD]'], \
                         continuing_subword_prefix='##', \
                         vocab_size=10000)
    tokenizerBPE.normalizer = StripAccents()  # add method for stripping accents
    tokenizerBPE.pre_tokenizer = Sequence([WhitespaceSplit(), Punctuation(behavior='removed')])
    tokenizerBPE.post_processor = ByteLevel(trim_offsets=True)
    tokenizerBPE.train_from_iterator(df_train['description'], trainer=trainer)

    with open('tokenizers/tokenizerBPE.pickle', 'wb') as handle:
        pickle.dump(tokenizerBPE, handle, protocol=3)

    tokenizerBPE = load_tokenizer("tokenizers/tokenizerBPE.pickle")

    vocabSize = tokenizerBPE.get_vocab_size()
    print('size of vocabulary: {}'.format(vocabSize))
    for i in range(10):
        print('vocabulary id: {0}, word: {1}'.format(i, tokenizerBPE.id_to_token(i)))
        j = vocabSize - i - 1
        print('vocabulary id: {0}, word: {1}'.format(j, tokenizerBPE.id_to_token(j)))

    print(df_test.loc[0, 'description'])
    out = tokenizerBPE.encode(df_test.loc[0, 'description'])
    print(out.ids)
    print(out.tokens)
    print(tokenizerBPE.decode(out.ids))

    outFull = tokenizerBPE.encode_batch(df_test['description'])
    ntokens = 0
    nunk = 0
    for encoded in outFull:
        ntokens += len(encoded.ids)
        nunk += len(encoded.ids) - np.count_nonzero(encoded.ids)
    print('ratio of unknown tokens: {0:.4f}'.format(nunk / ntokens))
    print('total number of tokens: {}'.format(ntokens))