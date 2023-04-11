import nlpaug.augmenter.word as naw
import random
import utils

test_sentence = "This is a test sentence. Ball is blue, triangle is red, my will to live is dead."
aug = naw.SynonymAug(aug_src='wordnet', model_path=None, name='Synonym_Aug', aug_min=1, aug_max=20, aug_p=0.15,
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


def synonym_aug_batch(list_of_strings):
    aug = naw.SynonymAug(aug_src='wordnet', model_path=None, name='Synonym_Aug', aug_min=1, aug_max=10, aug_p=0.1,
                         lang='eng',
                         stopwords=None, tokenizer=None, reverse_tokenizer=None, stopwords_regex=None,
                         force_reload=False,
                         verbose=0)
    return [aug.augment(string)[0] for string in list_of_strings]

def save_to_csv(descriptions_dict, filename):
    lines_to_csv = []
    id_ = 0
    for genre, list_of_descriptions in descriptions_dict.items():
        for description in list_of_descriptions:
            lines_to_csv.append(" ::: ".join([str(id_),
                                              "title",
                                              genre,
                                              description]))
            id_ += 1

    random.shuffle(lines_to_csv)
    with open(filename, "w", encoding='utf-8') as outfile:
        for line in lines_to_csv:
            outfile.write(line)
            outfile.write("\n")



if __name__ == '__main__':

    TRAIN_DATA_PATH = "data/test_4categories.txt"
    TARGET_COUNT_PER_CATEGORY = 5000

    # creating a dict
    df_train = utils.load_data(TRAIN_DATA_PATH)
    dict_of_descriptions = {}
    for i in range(len(df_train["genre"])):
        genre = df_train["genre"][i].strip()
        if genre not in dict_of_descriptions:
            dict_of_descriptions[genre] = []
        dict_of_descriptions[genre].append(df_train["description"][i])

    for genre, list_ in dict_of_descriptions.items():
        print(f'{genre} : {len(list_)}')


    dict_of_descriptions = {key:dict_of_descriptions[key] for key in dict_of_descriptions if key in {"documentary", "drama", "comedy"}}

    save_to_csv(dict_of_descriptions, f"data/test_3categories.txt")

    import sys
    sys.exit()

    for key in dict_of_descriptions:
        # augmenting the categories with not enough entries
        if len(dict_of_descriptions[key]) <= TARGET_COUNT_PER_CATEGORY:
            new_list = list(dict_of_descriptions[key])
            while len(new_list) < TARGET_COUNT_PER_CATEGORY:
                #new_list.extend(synonym_aug_batch(dict_of_descriptions[key]))
                new_list.extend(list(dict_of_descriptions[key]))
            dict_of_descriptions[key] = new_list
        # cutting excess
        dict_of_descriptions[key] = dict_of_descriptions[key][0:TARGET_COUNT_PER_CATEGORY]
    for genre, list_ in dict_of_descriptions.items():
        print(f'{genre} : {len(list_)}')

    save_to_csv(dict_of_descriptions, f"data/train_data_new_{TARGET_COUNT_PER_CATEGORY}.txt")
