import pandas as pd

def load_data(filename):
    """
    Will read the csv files and return pandas dataframe with title, genre and description.
    (Won't work on train_data, cuz it has only 2 columns.
    """
    df_train = pd.read_csv(filename, sep=':::',
                           engine='python', header=None,
                           usecols=[1, 2, 3])
    df_train.rename(columns={0: "id",1: 'title', 2: 'genre', 3: 'description'}, inplace=True)

    return df_train
