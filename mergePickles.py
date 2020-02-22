import pickle
import os
import pandas as pd

filepath_pkl = 'data/pkl/'
allData = []

for filename in os.listdir(filepath_pkl):
    if not filename.endswith(".pkl"):
        continue
    else:
        fileData = []
        data = pickle.load(open(filepath_pkl + filename, "rb"))

        if 'file_id' in data.keys():
            fileData.append(data['file_id'])
        else:
            continue

        fileData.append(data['onset_cc_avg'])
        if 'onset_cc_std' in data.keys():
            fileData.append(data['onset_cc_std'])
        else:
            continue

        allData.append(fileData)


with open("onsets_cc_data.csv", 'w', newline='') as output:
    writer = csv.writer(output)
    writer.writerows(allData)

    print(len(allData))
