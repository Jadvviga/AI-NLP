from transformers import MarianMTModel, MarianTokenizer

# Get the name of the first model
first_model_name = 'Helsinki-NLP/opus-mt-en-fr'
# Get the tokenizer
first_model_tkn = MarianTokenizer.from_pretrained(first_model_name)
# Load the pretrained model based on the name
first_model = MarianMTModel.from_pretrained(first_model_name)

# Get the name of the second model
second_model_name = 'Helsinki-NLP/opus-mt-fr-en'
# Get the tokenizer
second_model_tkn = MarianTokenizer.from_pretrained(second_model_name)
# Load the pretrained model based on the name
second_model = MarianMTModel.from_pretrained(second_model_name)

original_texts = ["This article aims to perform the back translation for text data augmentation",
          "It is the 25th article by Zoumana on Medium. He loves to give back to the community",
          "The first model translates from English to French, which is a temporary process", 
          "The second model finally translates back all the temporary french text into English"]

def format_batch_texts(language_code, batch_texts):
    formated_bach = [">>{}<< {}".format(language_code, text) for text in batch_texts]
    return formated_bach

def perform_translation(batch_texts, model, tokenizer, language="fr"):
    # Prepare the text data into appropriate format for the model
    formated_batch_texts = format_batch_texts(language, batch_texts)
    # Generate translation using model
    translated = model.generate(**tokenizer(formated_batch_texts, return_tensors="pt", padding=True))
    # Convert the generated tokens indices back into text
    translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    return translated_texts

def combine_texts(original_texts, back_translated_batch):
  return set(original_texts + back_translated_batch) 

def perform_back_translation_with_augmentation(batch_texts, original_language="en", temporary_language="fr"):
  # Translate from Original to Temporary Language
  tmp_translated_batch = perform_translation(batch_texts, first_model, first_model_tkn, temporary_language)
  # Translate Back to English
  back_translated_batch = perform_translation(tmp_translated_batch, second_model, second_model_tkn, original_language)
  # Return The Final Result
  return combine_texts(original_texts, back_translated_batch)


final_augmented = perform_back_translation_with_augmentation(original_texts)
print(final_augmented)

