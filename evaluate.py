import utils
import tokenization
import model

# required modules
import tenserflow as tf
import os
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import numpy as np

# optional modules
import nltk
import string

# nltk.download('punkt') # only required for the first time
# nltk.download('averaged_perceptron_tagger') # only required for the first time
# nltk.download('stopwords') # only required for the first time

if __name__ == '__main__':

    TEST_SOLUTION_DATA_PATH = "data/test_data_solution.txt"
    TRAIN_DATA_PATH = "data/train_data.txt"
    TOKENIZER_PATH = "tokenizers/tokenizerWP.pickle"

    MODELS_PATH = "models"
    REPORTS_PATH = "metrics/reports"
    CONFUSION_MATRIX_PATH = "metrics/confusion_matrix"

    model_filename = utils.choose_file_to_load(MODELS_PATH)

    target_length = utils.model_filename_parse_dimension(model_filename)

    #TODO add loading data and labels here


    model = tf.keras.models.load_model(filepath=os.path.join(MODELS_PATH, model_filename))
 
    #TODO evaluation
    y_predict = model.predict(x=data_generator_test)
    y_predict_argmax = tf.argmax(input=y_predict, axis=1)

    # classification report
    target_names = [f"flower_{i}" for i in range(1, 103)]
    report = classification_report(y_true=list(labels_test.values()),
                                   y_pred=y_predict_argmax,
                                   target_names=target_names,
                                   output_dict=True)
    # pprint.pprint(report)

    report_filename = "clfreport_" + os.path.splitext(model_filename)[0] + ".json"
    with open(os.path.join(REPORTS_PATH, report_filename), "w") as outfile:
        json.dump(report, outfile, indent=2)

    # confiusion matrix
    cm_filename = "ConfusionMatrix_" + os.path.splitext(model_filename)[0] + ".png"
    cm = confusion_matrix(list(labels_test.values()), y_predict_argmax)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    fig = matplotlib.pyplot.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    disp.plot(include_values=False, xticks_rotation='vertical', ax=ax)

    ax.set_xticks([x*10 for x in range(10)])
    ax.set_yticks([x*10 for x in range(10)])

    matplotlib.pyplot.savefig(os.path.join(CONFUSION_MATRIX_PATH, cm_filename),dpi=300)

    matplotlib.pyplot.show()