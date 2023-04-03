import utils
import tokenization

# required modules
import tensorflow as tf
import os

import matplotlib.pyplot as plt
import numpy as np
import json

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib




if __name__ == '__main__':

    TEST_SOLUTION_DATA_PATH = "data/test_data_solution.txt"
    TRAIN_DATA_PATH = "data/train_data.txt"

    TOKENIZERS_PATH = "tokenizers"
    MODELS_PATH = "models"
    REPORTS_PATH = "metrics/reports"
    CONFUSION_MATRIX_PATH = "metrics/confusion_matrix"

    model_filename = utils.choose_file_to_load(MODELS_PATH)
    tokenizer_filename = utils.choose_file_to_load(TOKENIZERS_PATH)

    target_length = utils.model_filename_parse_targetlength(model_filename)


    df_test = utils.load_data(TEST_SOLUTION_DATA_PATH)
    labels_dict = utils.get_labels_dict(df_test["genre"])

    labels_test = np.array([labels_dict[genre] for genre in df_test["genre"]])
    Y_test = tf.keras.utils.to_categorical(labels_test, num_classes=len(labels_dict))


    model = tf.keras.models.load_model(filepath=os.path.join(MODELS_PATH, model_filename))
    tokenizer = tokenization.load_tokenizer(os.path.join(TOKENIZERS_PATH, tokenizer_filename))

    tokenized_descriptions_test = tokenizer.encode_batch(df_test['description'])
    tokenization.encodings_normalize_length(tokenized_descriptions_test, target_length=target_length)

    X_test = np.array([np.array(encoding.ids) for encoding in tokenized_descriptions_test])
    print(X_test[0])


    # EVALUATION
    y_predict = model.predict(x=X_test)
    y_predict_argmax = tf.argmax(input=y_predict, axis=1)

    # classification report
    target_names = list(labels_dict.keys())
    report = classification_report(y_true=labels_test,
                                   y_pred=y_predict_argmax,
                                   target_names=target_names,
                                   output_dict=True)
    # pprint.pprint(report)

    report_filename = "clfreport_" + os.path.splitext(model_filename)[0] + ".json"
    with open(os.path.join(REPORTS_PATH, report_filename), "w") as outfile:
        json.dump(report, outfile, indent=2)

    # confiusion matrix
    cm_filename = "ConfusionMatrix_" + os.path.splitext(model_filename)[0] + ".png"
    cm = confusion_matrix(labels_test, y_predict_argmax)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    fig = matplotlib.pyplot.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    disp.plot(include_values=False, xticks_rotation='vertical', ax=ax)

    ax.set_xticks([x for x in range(len(labels_dict))])
    ax.set_yticks([x for x in range(len(labels_dict))])

    matplotlib.pyplot.savefig(os.path.join(CONFUSION_MATRIX_PATH, cm_filename),dpi=300)

    matplotlib.pyplot.show()