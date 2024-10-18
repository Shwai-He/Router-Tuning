import json
import random
import os

task_dict = {'aqua': 'AQuA', 
             'gsm8k': 'grade-school-math', 
             'commonsensqa': 'CommonsenseQA', 
             'addsub': 'AddSub', 
             'multiarith': 'MultiArith', 
             'singleeq': 'SingleEq', 
             'multiarith': 'MultiArith', 
             'strategyqa': 'StrategyQA', 
             'svamp': 'svamp', 
             'bigbench_date': 'Bigbench_Date', 
             'coin_flip': 'coin_flip',
             'last_letters': 'last_letters', 
             }

def load_data(args) -> list:
    """
    read data from dataset file
    Args:
        args:

    Returns:

    """
    file_path = f'dataset/{args.dataset}/test.json'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"can not find dataset file : {file_path}")

    if file_path.endswith("json"):
        json_data = json.load(open(file_path, 'r'))
    else:
        json_data = []
        decoder = json.JSONDecoder()
        with open(file_path) as f:
            lines = f.readlines()
            for line in lines:
                json_line = decoder.raw_decode(line)[0]
                json_data.append(json_line)
    return json_data


# cot data
def get_demonstrations(args, logger):
    
    random.seed(args.seed)
    questions, answers = [], []
    decoder = json.JSONDecoder()
    if args.dataset in task_dict:
        dataset = task_dict[args.dataset]
        dataset_path = f"./data_llm/{dataset}/cot_data.jsonl"
        with open(dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                questions.append(json_res["question"])
                answers.append(json_res["answer"])
        
        indices = random.sample(range(len(questions)), min(args.k, len(questions)))
        demonstrations = "".join([questions[idx] + answers[idx] + '\n' for idx in indices])
        return demonstrations
    else:
        return None