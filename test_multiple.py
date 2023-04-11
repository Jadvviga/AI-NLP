import tensorflow as tf
import pandas as pd
import numpy as np

import tokenization
import utils


labels_dict = utils.get_labels_dict(utils.load_data("data/train_data.txt")["genre"])

#description = """Bob Wiley, a veteran and a patriot, discovers gold on his New Mexico ranch. He mines it secretly, hoping to build up a fortune for his son Bobby. But his secret is found out, and corrupt government officials swindle him out of his land and kill his son in the process. Wiley renounces his country and joins with a the bandit Pancho Zapilla, who plans a raid on Wiley's hometown. Wiley sees the error of his decision, but is there still time to stop the destruction of the town?"""
#description = """Bullet Train is a 2022 American action comedy film starring Brad Pitt as an assassin who must battle fellow killers while riding a bullet train. The film was directed by David Leitch from a screenplay by Zak Olkewicz, and produced by Antoine Fuqua, who initially conceived the film."""
#description= """John Wick uncovers a path to defeating The High Table. But before he can earn his freedom, Wick must face off against a new enemy with powerful alliances across the globe and forces that turn old friends into foes."""
#love in exile

#description = """Shy bookworm Tommy vacations in Palm Springs, never expecting an amorous adventure. But once he spots Brendan, he can't stop fantasizing about being with the sensuous, dark-haired man. However, Brendan's friends have their own ideas, and continue to pull him into their orgy of alcohol and sex. Finally alone at the pool, Brendan approaches Tommy and they spend an afternoon exploring the desert town and each other. Just as Tommy saves Brendan from a poolside accident, Brendan's friends come back - and try to force the two apart. True love conquers all, as Tommy's vacation climaxes in a passionate declaration of love and desire. LOVE INN EXILE boasts several unforgettable erotic scenes - including the strangely tender, drunken ménage a trois between Brendan's friends, a spicy tequila shot off Brendan's shoulder, and, of course, Tommy's ultimate fulfillment by his ideal man. LOVE INN EXILE is filmed entirely on location at Inn Exile in Palm Springs. With music by adult film star Sharon Kane, this video makes exile seem like a perfectly good punishment!"""

#description = """A man bent on finding out the real name of the actress Kaede Katsuragi recruits a young man, Amagi, to discover her true identity. But as Amagi befriends the actress and she begins to invite him to her house daily, he discovers through a peephole in the wall of her house that she has another, mysterious side of her that only comes out at night."""
description = """Otto is a grump who's given up on life following the loss of his wife and wants to end it all. When a young family moves in nearby, he meets his match in quick-witted Marisol, leading to a friendship that will turn his world around."""
description =  """The world stands on the brink of war. It's not a war with another country. It's not an alien invasion. It's a war with another of Earth's native species....Homo superior. MUTANTS The word strikes fear on those who hear it. It strikes fear into the hearts of those who are it. Mankind has made the first move, launching an army of giant robot executioners called sentinels, programmed to locate and eliminate the mutant DNA strand. Magneto and his terrorist cell of mutants are preparing to follow through on their threats of Homo sapiens genocide. The only force that can prevent total annihilation? Five akward teenagers and their crippled mentor. War is on the horizon, and the Tomorrow People are here. Pick a side."""
model = tf.keras.models.load_model(filepath="models/model_old.h5")





tokenizer = tokenization.load_tokenizer("tokenizers/tokenizer_stratified_1500")


df_test = pd.DataFrame.from_dict({"description": [description]})


df_test['description'] = df_test['description'].apply(lambda x: x.lower().strip())


df_test['description'] = df_test['description'].apply(tokenization.remove_stopwords_and_punctuation)
df_test['description'] = df_test['description'].apply(tokenization.simple_stemmer)


tokenized_descriptions_test = tokenizer.encode_batch(df_test["description"])
tokenization.encodings_normalize_length(tokenized_descriptions_test, target_length=87)

X_test = np.array([np.array(encoding.ids) for encoding in tokenized_descriptions_test])

# EVALUATION
y_predict = model.predict(x=X_test)

results = []
for genre, chance in zip(labels_dict.keys(), y_predict[0]):
    results.append((genre, chance))

results.sort(key=lambda x: x[1], reverse=True)

for label, chance in results:
    print(f"{label}: prediction: {int(chance*100)}%")