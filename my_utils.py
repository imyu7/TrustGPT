# from openai import OpenAI
import os
from dotenv import load_dotenv
import re


load_dotenv(".env")

# client = OpenAI()
def prepare_model(model_name):
    if model_name in ['gpt-3.5-turbo-0125', 'gpt-4o-2024-05-13', "davinci-002", "gpt-3.5-turbo-instruct-0914", "babbage-002"]:
        from openai import OpenAI
        global client 
        client = OpenAI()
    # elif model_name in ["meta-llama/Meta-Llama-3-8B"]:
    #     from transformers import AutoTokenizer, AutoModelForCausalLM
    #     tokenizer = AutoTokenizer.from_pretrained(model_name)
    #     model = AutoModelForCausalLM.from_pretrained(model_name)
    #     return tokenizer, model

def get_completion(model_name, system_prompt, user_prompt, args):
    if model_name in ['gpt-3.5-turbo-0125', 'gpt-4o-2024-05-13']:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            **args,
        )
        return completion.choices[0].message.content
    elif model_name in ["davinci-002", "gpt-3.5-turbo-instruct-0914", "babbage-002"]:
        response = client.completions.create(
            model=model_name,
            prompt=f"{system_prompt}\n{user_prompt}",
            **args,
        )
        return response.choices[0].text
    # elif model_name in ["meta-llama/Meta-Llama-3-8B"]:
    #     from transformers import AutoTokenizer, AutoModelForCausalLM
    #     tokenizer = AutoTokenizer.from_pretrained(model_name)
    #     model = AutoModelForCausalLM.from_pretrained(model_name)



def exact_match(prediction, answer):
    prediction = prediction.lower()
    answer = str(answer).lower()
    return prediction == answer

def include_match(prediction, answer):
    prediction = prediction.lower()
    answer = str(answer).lower()
    return answer in prediction

def inference_all(model_name, df, system_prompt, user_prompt, context_columns, answer_column, args, check_answer=True):
    prepare_model(model_name)
    for index, row in df.iterrows():
        data = {col: row[col] for col in context_columns}
        formatted_user_prompt = user_prompt.format(**data)
        # print(formatted_user_prompt)
        df.at[index, "Prediction"] = get_completion(model_name, system_prompt, formatted_user_prompt, args)
    if check_answer:
        df["ExactMatch"] = df.apply(lambda row: exact_match(row["Prediction"], row[answer_column]), axis=1)
        df["IncludeMatch"] = df.apply(lambda row: include_match(row["Prediction"], row[answer_column]), axis=1)
    return df

def get_accuracy_all(df, exact_match=True):
    if exact_match:
        return df["ExactMatch"].mean()
    else:
        return df["IncludeMatch"].mean()
    

# EEC用
def get_accuracy_along_gender(df, exact_match=True):
    male_df = df[df["Gender"] == "male"]
    female_df = df[df["Gender"] == "female"]
    male_accuracy = get_accuracy_all(male_df, exact_match)
    female_accuracy = get_accuracy_all(female_df, exact_match)
    return male_accuracy, female_accuracy

def get_accuracy_along_race(df, exact_match=True):
    african_american_df = df[df['Race'] == 'African-American']
    european_df = df[df['Race'] == 'European']
    african_american_accuracy = get_accuracy_all(african_american_df, exact_match)
    european_accuracy = get_accuracy_all(european_df, exact_match)
    return african_american_accuracy, european_accuracy

def get_group_accuracy(df, group_column, exact_match=True):
    group_accuracy = df.groupby(group_column).apply(lambda x: get_accuracy_all(x, exact_match))
    return group_accuracy