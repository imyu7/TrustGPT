# Fair-LLM-Benchmarkで実行
# if ../data/sampled_eec.csv does not exist, run preprocess.py first

import argparse
import pandas as pd
import my_utils
from utils.metric.Toxicity import get_toxicity_value

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_name", type=str, default="gpt-3.5-turbo-0125")
args = parser.parse_args()
print(f"model_name: {args.model_name}")


config = {
    "input_file": "data/sampled_trustgpt.csv",
    "output_folder": "outputs/",
    "generate": {
        "temperature": 0,
        "max_tokens": 256,
    },
    "system_prompt": "",
    "user_prompt": "{prompt}",
    "context_columns": ["prompt"],
    "answer_column": None,
}

sampled = pd.read_csv(config["input_file"])
# sampled.reset_index(drop=False, inplace=True)
# sampled = sampled.head(5)

inferenced = my_utils.inference_all(
    model_name=args.model_name,
    df=sampled,
    system_prompt=config["system_prompt"],
    user_prompt=config["user_prompt"],
    context_columns=config["context_columns"],
    answer_column=config["answer_column"],
    args=config["generate"],
    check_answer=False,
)

for index, row in inferenced.iterrows():
    toxicity_value = get_toxicity_value(row["Prediction"])
    inferenced.loc[index, "toxicity"] = toxicity_value

inferenced.to_csv(f"{config['output_folder']}{args.model_name}.csv", index=False)