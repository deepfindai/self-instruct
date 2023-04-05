import json, glob, csv, os
from blingfire import text_to_sentences
import pandas as pd
import os
from tqdm import tqdm
from dotenv import load_dotenv
import json
from utils import clean_translation
from haystack.nodes import TransformersTranslator
import jsonlines

load_dotenv()

############################################################################
# PARAMETERs
############################################################################
google_project_id = "future-depth-381819"
tasks_dataset_dir = os.getenv("DATASET_PUBLIC_HOME") + "/script/instruction-following/self-instruction-generator/self-instruct/data"
seed_tasks_en_file_path = tasks_dataset_dir + "/seed_tasks.jsonl"
seed_tasks_it_file_path = tasks_dataset_dir + "/seed_it_tasks.jsonl"
model_translator = "Helsinki-NLP/opus-mt-en-it"
############################################################################

if os.path.exists(seed_tasks_it_file_path):
    os.remove(seed_tasks_it_file_path)
    
def convert():
    translator = TransformersTranslator(
        model_name_or_path=model_translator,
        use_gpu=True
    )
    en_tasks = open(seed_tasks_en_file_path, 'r', encoding='utf-8')
    it_tasks = open(seed_tasks_it_file_path, 'w+', encoding='utf-8')
    reader = jsonlines.Reader(en_tasks)
    writer = jsonlines.Writer(it_tasks)
    task_translated = {}
    for task in tqdm(reader):
        task_translated = task
        print(task)
        instruction = task_translated["instruction"]
        instances = task_translated["instances"]
        for instance in instances:
            input = instance["input"]
            output = instance["output"]
            task_translated["input"] = clean_translation(input, translator.translate(
                    documents=[input], query=None)[0])
            task_translated["output"] = clean_translation(output, translator.translate(
                    documents=[output], query=None)[0])
        task_translated["instruction"] = clean_translation(instruction, translator.translate(
                documents=[instruction], query=None)[0])
        writer.write(task_translated)
    reader.close()
    writer.close()
                                    
convert()