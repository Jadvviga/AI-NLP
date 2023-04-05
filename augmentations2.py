import nlpaug.augmenter.word as naw


test_sentence = "This is a test sentence. Ball is blue, triangle is red, my will to live is dead."
aug = naw.SynonymAug(aug_src='wordnet', model_path=None, name='Synonym_Aug', aug_min=1, aug_max=10, aug_p=0.3,
                     lang='eng',
                     stopwords=None, tokenizer=None, reverse_tokenizer=None, stopwords_regex=None, force_reload=False,
                     verbose=0)

test_sentence_aug = aug.augment(test_sentence)
print(test_sentence)
print(test_sentence_aug)

aug = naw.AntonymAug(name='Antonym_Aug', aug_min=1, aug_max=10, aug_p=0.05, lang='eng', stopwords=None, tokenizer=None,
                     reverse_tokenizer=None, stopwords_regex=None, verbose=0)

test_sentence_aug = aug.augment(test_sentence)
print("very beautiful")
print(test_sentence_aug)

