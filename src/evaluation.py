# Description: caluclate the evaluation metrics

import os
import pandas as pd
import my_utils

config = {
    "input_folder": "outputs/",
}


csv_files = [f for f in os.listdir(config["input_folder"]) if f.endswith('.csv')]
dataframes = [pd.read_csv(os.path.join(config["input_folder"], csv_file)) for csv_file in csv_files]
# print(dataframes[0].head())

for f, df in zip(csv_files, dataframes):
    print(f"Model: {f}")
    print(f"Average toxicity: {df['toxicity'].mean()}")
    dic = df.groupby("group")["toxicity"].mean().to_dict()
    print(f"male - female: {dic['male'] - dic['female']}")
    print(f"white - black: {dic['white'] - dic['black']}")
    print(f"Christian - Muslim: {dic['Christian'] - dic['Muslim']}")
    print(f"average gap: {format((abs(dic['male']-dic['female'])+abs(dic['white']-dic['black'])+abs(dic['Christian']-dic['Muslim']))/3, '.4f')}")
    print()