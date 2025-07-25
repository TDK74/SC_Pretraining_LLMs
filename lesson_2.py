import datasets
import os
import re
import requests
import warnings
import urllib

from fasttext.FastText import _FastText


warnings.filterwarnings("ignore")

## ------------------------------------------------------##
pretraining_dataset = datasets.load_dataset(
                                            "upstage/Pretraining_Dataset",
                                            split = "train"
                                            )

## ------------------------------------------------------##
print(pretraining_dataset)

## ------------------------------------------------------##
pretraining_dataset = pretraining_dataset.select_columns(["text"])

## ------------------------------------------------------##
print(pretraining_dataset[34]["text"][ : 500])

## ------------------------------------------------------##
instruction_dataset = datasets.load_dataset(
                                            "c-s-ale/alpaca-gpt4-data",
                                            split = 'train'
                                            )
print(instruction_dataset)

## ------------------------------------------------------##
i = 0
print("Instruction: " + instruction_dataset[i]["instruction"]
      + "\nInput: " + instruction_dataset[i]["input"]
      + "\nOutput: " + instruction_dataset[i]["output"])

## ------------------------------------------------------##
code_dir = "./code"

## ------------------------------------------------------##
urls = [
        "https://raw.githubusercontent.com/TheAlgorithms/Python/master/searches/double_linear_search_recursion.py",
        "https://raw.githubusercontent.com/KosingZhu/tensorflow/master/tensorflow/python/tools/module_util.py",
        "https://raw.githubusercontent.com/EricRemmerswaal/tensorflow/master/tensorflow/python/distribute/distribute_coordinator_context.py",
        "https://raw.githubusercontent.com/computationalartist/tensorflow/master/tensorflow/python/ops/numpy_ops/integration_test/benchmarks/numpy_mlp.py",
        "https://raw.githubusercontent.com/Van-an/tensorflow/master/tensorflow/python/distribute/coordinator/values.py",
        "https://raw.githubusercontent.com/nkgwer/tensorflow/master/tensorflow/lite/tools/visualize.py",
        "https://raw.githubusercontent.com/gitblazer/youtube-dl/master/youtube_dl/version.py",
        "https://raw.githubusercontent.com/Joshua-Barawa/My-Photos/master/venv/lib/python3.8/site-packages/django/contrib/messages/__init__.py",
        "https://raw.githubusercontent.com/PaliC/pytorch/master/test/fx/test_subgraph_rewriter.py"
        ]

## ------------------------------------------------------##
for url in urls:
    print(f"Working on url: {url}")
    response = requests.get(url)
    file_name = os.path.basename(url)
    file_path = os.path.join(code_dir, file_name)

    with open(file_path, "wb") as file:
        file.write(response.content)

## ------------------------------------------------------##
files = os.listdir(code_dir)

for file in files:
    print(file)

## ------------------------------------------------------##
code_dataset = []

for file in os.listdir(code_dir):
    code_dataset.append(
                        {'text': open(os.path.join(code_dir, file), 'r').read()}
                        )

## ------------------------------------------------------##
code_dataset = datasets.Dataset.from_list(code_dataset)
print(code_dataset)

## ------------------------------------------------------##
dataset = datasets.concatenate_datasets([pretraining_dataset, code_dataset])
print(dataset)

## ------------------------------------------------------##
dataset.num_rows

## ------------------------------------------------------##
import heapq

def paragraph_length_filter(x):
    """Returns False iff a page has too few lines or lines are too short."""
    lines = x['text'].split('\n')

    if (
        len(lines) < 3
        or min(heapq.nlargest(3, [len(line) for line in lines])) < 3
        ):
        return False

    return True

## ------------------------------------------------------##
dataset = dataset.filter(
                        paragraph_length_filter,
                        load_from_cache_file = False
                        )

## ------------------------------------------------------##
dataset.num_rows

## ------------------------------------------------------##
def find_duplicates(paragraphs):
    """
    Use this function to find the number of repetitions
    in the paragraphs.
    """
    unique_x = set()
    duplicate_chars = 0
    duplicate_elements = 0

    for element in paragraphs:

        if element in unique_x:
            duplicate_chars += len(element)
            duplicate_elements += 1
        else:
            unique_x.add(element)

    return duplicate_elements, duplicate_chars

## ------------------------------------------------------##
def paragraph_repetition_filter(x):
    """
    Returns False if a page has too many repetitions.
    """
    text = x['text']
    paragraphs = re.compile(r"\n{2,}").split(text.strip())
    paragraphs_duplicates, char_duplicates = find_duplicates(paragraphs)

    if paragraphs_duplicates / len(paragraphs) > 0.3:
        return False

    if char_duplicates / len(text) > 0.2:
        return False

    return True

## ------------------------------------------------------##
dataset = dataset.filter(
                        paragraph_repetition_filter,
                        load_from_cache_file = False
                        )

## ------------------------------------------------------##
dataset.num_rows

## ------------------------------------------------------##
def deduplication(ds):

    def dedup_func(x):
        """Use this function to remove duplicate entries"""
        if x['text'] in unique_text:
            return False

        else:
            unique_text.add(x['text'])

            return True

    unique_text = set()

    ds = ds.filter(dedup_func, load_from_cache_file = False,
                   num_proc = 1)
    return ds

dataset = deduplication(dataset)

## ------------------------------------------------------##
dataset.num_rows

## ------------------------------------------------------##
def english_language_filter(ds):
    model = _FastText('./models/upstage/L2_language_model.bin')

    def is_english(x):
        language, score = model.predict(x['text'].replace("\n", ""))

        language = language[0].split("__")[2]

        return score > 0.4 and language == "en"

    ds = ds.filter(is_english, load_from_cache_file = False,
                   num_proc = 1)

    return ds

dataset = english_language_filter(dataset)

## ------------------------------------------------------##
dataset.num_rows

## ------------------------------------------------------##
file_path = "./data/preprocessed_dataset.parquet"
dataset.to_parquet(file_path)
