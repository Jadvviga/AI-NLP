import pandas as pd
import matplotlib.pyplot as plt
import os

def load_data(filename):
    """
    Will read the csv files and return pandas dataframe with title, genre and description.
    (Won't work on train_data, cuz it has only 2 columns.
    """
    df_train = pd.read_csv(filename, sep=':::',
                           engine='python', header=None,
                           usecols=[1, 2, 3])
    df_train.rename(columns={0: "id", 1: 'title', 2: 'genre', 3: 'description'}, inplace=True)

    return df_train


def get_labels_dict(genres):
    genres = sorted(genres) # i sort it to arrive always to the same labels dict
    label_dict = {}
    label = 0
    for item in genres:
        if item not in label_dict:
            label_dict[item] = label
            label += 1

    return label_dict


def make_plots_from_history(history, plots_path, model_filename):
    """
    Plots history of a trained model. Show the plots and saves them to pngs with model_filename
    """
    # Accuracy plot
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(('Training Accuracy', 'Validation accuracy'))
    plt.ylabel('Accuracy')
    plt.xlabel('Number of Epochs')

    plot_filename = "plot_accuracy_" + os.path.splitext(model_filename)[0] + ".png"
    plt.savefig(os.path.join(plots_path, plot_filename))
    plt.show()

    # Loss plot
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(('Training Loss', 'Validation Loss'))
    plt.ylabel('Loss')
    plt.xlabel('Number of Epochs')

    plot_filename = "plot_loss_" + os.path.splitext(model_filename)[0] + ".png"
    plt.savefig(os.path.join(plots_path, plot_filename))
    plt.show()
