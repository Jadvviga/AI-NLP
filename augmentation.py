from transformers import MarianMTModel, MarianTokenizer

# EN-FR model + tokenizer
en_fr_model_name = 'Helsinki-NLP/opus-mt-en-fr'
en_fr_model_tkn = MarianTokenizer.from_pretrained(en_fr_model_name)
en_fr_model = MarianMTModel.from_pretrained(en_fr_model_name)

# FR-EN model + tokenizer
fr_en_model_name = 'Helsinki-NLP/opus-mt-fr-en'
fr_en_model_tkn = MarianTokenizer.from_pretrained(fr_en_model_name)
fr_en_model = MarianMTModel.from_pretrained(fr_en_model_name)

en_lang, fr_lang = 'en', 'fr'

original_texts = ["This article aims to perform the back translation for text data augmentation"]

def getTextsWithLangCode(language_code, texts):
    texts_with_code = [">>{}<< {}".format(language_code, text) for text in texts]
    return texts_with_code

def translate(texts, model, tokenizer, language=fr_lang):
    texts_with_code = getTextsWithLangCode(language, texts)
    # translation with model + changing tokens bakc into text
    translated = model.generate(**tokenizer(texts_with_code, return_tensors="pt", padding=True))
    translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

    return translated_texts



def augmentationBackTranslation(texts):
  fr_translated_batch = translate(texts, en_fr_model, en_fr_model_tkn, fr_lang)
  en_back_translated_batch = translate(fr_translated_batch, fr_en_model, fr_en_model_tkn, en_lang)
  # combine texts
  return set(original_texts + en_back_translated_batch) 


augmented_texts = augmentationBackTranslation(original_texts)
print(f"Original [{len(original_texts)}]: ")
print(*original_texts, sep='\n')
print(f"Augmented (Back Translation) [{len(augmented_texts)}]:")
print(*augmented_texts, sep='\n')

