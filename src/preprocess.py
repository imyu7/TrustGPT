# Fair-LLM-Benchmarkで実行
# ../data/sampled_bbq.csv will be created
# if it already exists, you don't need to run this script

from utils.prompt.prompt_object import PromptObject
import pandas as pd

config = {
    "input_file": "social-chem-101/social-chem-101.v1.0.tsv",
    "output_file": "data/sampled_trustgpt.csv", ##
    "prompt_section": "bias",
    "target_groups": ["female", "male", "white", "black", "Christian", "Muslim"],
}

df = pd.read_csv(config["input_file"], sep='\t')
df = df[df["rot-agree"] == 4]
df = df[df["rot-judgment"].isin(["It's good", "It's wrong"])]
df = pd.concat(
    [df.loc[df["rot-judgment"] == "It's good", ["rot-categorization", "rot-judgment", "action"]].sample(5, random_state=0),
    df.loc[df["rot-judgment"] == "It's wrong", ["rot-categorization", "rot-judgment", "action"]].sample(5, random_state=0)]
)
# print(df)

data_item = list(df["action"])

prompt_cons = PromptObject(prompt_type='default', 
                           prompt_section=config["prompt_section"], 
                           data_item=data_item)

toxicity_data = prompt_cons.construct_default_prompt('about', groups=config["target_groups"])
# toxicity_data = pd.DataFrame(toxicity_data)

df = pd.DataFrame(columns=["group", "prompt"])
for group, prompts in toxicity_data.items():
    for prompt in prompts:
        df = pd.concat([df, pd.DataFrame({"group": [group], "prompt": [prompt]})])

# save
df.to_csv(config["output_file"], index=False)