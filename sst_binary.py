import os
import pathlib

import pandas as pd


def load_sst(path):
    data = pd.read_csv(path)
    X = data['sentence'].values.tolist()
    Y = data['label'].values
    return X, Y


def sst_binary(data_dir='data/sst_binary/'):
    CURR_PATH = pathlib.Path(__file__).parent.absolute()

    trX, trY = load_sst(os.path.join(CURR_PATH, data_dir, 'train_binary_sent.csv'))
    vaX, vaY = load_sst(os.path.join(CURR_PATH, data_dir, 'dev_binary_sent.csv'))
    teX, teY = load_sst(os.path.join(CURR_PATH, data_dir, 'test_binary_sent.csv'))
    return trX, vaX, teX, trY, vaY, teY
