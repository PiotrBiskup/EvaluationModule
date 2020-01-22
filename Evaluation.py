import pandas as pd
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from os import listdir
from os.path import isfile, join


def get_files_names(dir_path):
    return [file for file in listdir(dir_path) if isfile(join(dir_path, file))]


def get_df_from_files(dir_path, files_names):
    data_frames = []
    for file in files_names:
        path = dir_path + file
        data_frames.append(pd.read_csv(path, sep=" ", header=None))
    return data_frames

def get_df_last_20_percent(data_frame):
