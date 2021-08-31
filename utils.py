import numpy as np
import pandas as pd
import os


DATA_PATH = './data'


def obtain_csv_names(data_path = DATA_PATH):
    return os.listdir(data_path)


def read_sequences(file_name, data_path = DATA_PATH):
    file_path = os.path.join(data_path, file_name)
    df = pd.read_csv(file_path)
    return df['CDR3'].tolist()


def read_energy(file_name, data_path = DATA_PATH):
    file_path = os.path.join(data_path, file_name)
    df = pd.read_csv(file_path)
    return df['Energy'].tolist()



if __name__ == '__main__':
    files = obtain_csv_names()
    name = files[0]
    sequences, energy = read_sequences(name), read_energy(name)
