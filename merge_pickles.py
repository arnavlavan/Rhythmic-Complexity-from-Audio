import pickle
import os
import pandas as pd
import numpy as np

file_path_pkl = 'data/pkl/'
file_list = os.listdir(file_path_pkl)
data_list = []

for filename in file_list:
    if not filename.endswith(".pkl"):
        continue
    else:
        data = pickle.load(open(file_path_pkl + filename, "rb"))
        datadict = {}
        for k, v in data.items():
            if not isinstance(v, np.ndarray):
                datadict[k] = v
        df = pd.DataFrame(datadict, index=[datadict['file_id'][:4]])
        data_list.append(df)

all_data = pd.concat(data_list, sort=True)
print(all_data.shape)
print(all_data.describe())
all_data.to_csv(file_path_pkl + 'onsets_data.csv')
