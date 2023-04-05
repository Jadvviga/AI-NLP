from transformers import MarianMTModel, MarianTokenizer

import utils

# EN-FR model + tokenizer
en_fr_model_name = 'Helsinki-NLP/opus-mt-en-fr'
en_fr_model_tkn = MarianTokenizer.from_pretrained(en_fr_model_name)
en_fr_model = MarianMTModel.from_pretrained(en_fr_model_name)

# FR-EN model + tokenizer
fr_en_model_name = 'Helsinki-NLP/opus-mt-fr-en'
fr_en_model_tkn = MarianTokenizer.from_pretrained(fr_en_model_name)
fr_en_model = MarianMTModel.from_pretrained(fr_en_model_name)

en_lang, fr_lang = 'en', 'fr'

original_texts = ["This article aims to perform the back translation for text data augmentation",
                  "It is the 25th article by Zoumana on Medium. He loves to give back to the community",
                  "The first model translates from English to French, which is a temporary process",
                  "The second model finally translates back all the temporary french text into English"]


def getTextsWithLangCode(language_code, texts):
    texts_with_code = [">>{}<< {}".format(language_code, text) for text in texts]
    return texts_with_code


def translate(texts, model, tokenizer, language=fr_lang):
    texts_with_code = getTextsWithLangCode(language, texts)
    # translation with model + changing tokens bakc into text
    translated = model.generate(**tokenizer(texts_with_code, return_tensors="pt", padding=True),max_length=10000)
    translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

    return translated_texts


def augmentationBackTranslation(texts):
    fr_translated_batch = translate(texts, en_fr_model, en_fr_model_tkn, fr_lang)
    en_back_translated_batch = translate(fr_translated_batch, fr_en_model, fr_en_model_tkn, en_lang)
    return en_back_translated_batch


if __name__ == '__main__':
    augmented_texts = augmentationBackTranslation(original_texts)
    print(f"Original [{len(original_texts)}]: ")
    print(*original_texts, sep='\n')
    print(f"Augmented (Back Translation) [{len(augmented_texts)}]:")
    print(*augmented_texts, sep='\n')

    TRAIN_DATA_PATH = "data/train_data.txt"
    OUTPUT_DATA_PATH = "data/train_augmented_data.txt"
    TARGET_COUNT_PER_CATEGORY = 1500

    # creating a dict
    df_train = utils.load_data(TRAIN_DATA_PATH)
    dict_of_descriptions = {}
    for i in range(len(df_train["genre"])):
        genre = df_train["genre"][i].strip()
        if genre not in dict_of_descriptions:
            dict_of_descriptions[genre] = []
        dict_of_descriptions[genre].append(df_train["description"][i])

    #print(dict_of_descriptions["war"][0])
    #print(augmentationBackTranslation([dict_of_descriptions['war'][0]]))
    import time
    now  = time.time()
    print(*augmentationBackTranslation(dict_of_descriptions["war"]), sep='\n')
    print(time.time() - now)
    # cutting excessive entries in too big categories
    """
    for key in dict_of_descriptions:
        # augmenting the categories with not enough entries
        if len(dict_of_descriptions[key]) <= TARGET_COUNT_PER_CATEGORY:
            new_list = list(dict_of_descriptions[key])
            while len(new_list) < TARGET_COUNT_PER_CATEGORY:
                new_list.extend(augmentationBackTranslation(dict_of_descriptions[key]))
        
        # cutting excess
        dict_of_descriptions[key] = dict_of_descriptions[key][0:TARGET_COUNT_PER_CATEGORY]
    """
    for genre, list_ in dict_of_descriptions.items():
        print(f'{genre} : {len(list_)}')
