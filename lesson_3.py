import datasets
import numpy as np

from transformers import AutoTokenizer


dataset = datasets.load_dataset(
                                "parquet",
                                data_files  = "./data/preprocessed_dataset.parquet",
                                split = "train"
                                )
print(dataset)

## ------------------------------------------------------##
dataset = dataset.shard(num_shards = 10, index =  0)
print(dataset)

## ------------------------------------------------------##
model_path_or_name = "./models/SOLAR-10.7B-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_path_or_name, use_fast =  False)

## ------------------------------------------------------##
tokenizer.tokenize("I'm a short sentence")

## ------------------------------------------------------##
def tokenization(example):
    tokens = tokenizer.tokenize(example["text"])

    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    token_ids = [tokenizer.bos_token_id] + token_ids + [tokenizer.eos_token_id]
    example["input_ids"] = token_ids

    example["num_tokens"] = len(token_ids)

    return example

## ------------------------------------------------------##
dataset = dataset.map(tokenization, load_from_cache_file = False)
print(dataset)

## ------------------------------------------------------##
sample = dataset[34]

print("text", sample["text"][ : 30])
print("\ninput_ids", sample["input_ids"][ : 30])
print("\nnum_tokens", sample["num_tokens"])

## ------------------------------------------------------##
np.sum(dataset["num_tokens"])

## ------------------------------------------------------##
input_ids = np.concatenate(dataset["input_ids"])
print(len(input_ids))

## ------------------------------------------------------##
max_seq_length = 32

## ------------------------------------------------------##
total_length = len(input_ids) - len(input_ids) % max_seq_length
print(total_length)

## ------------------------------------------------------##
input_ids = input_ids[ : total_length]
print(input_ids.shape)

## ------------------------------------------------------##
input_ids_reshaped = input_ids.reshape(-1, max_seq_length).astype(np.int32)
input_ids_reshaped.shape

## ------------------------------------------------------##
type(input_ids_reshaped)

## ------------------------------------------------------##
input_ids_list = input_ids_reshaped.tolist()
packaged_pretrain_dataset = datasets.Dataset.from_dict({"input_ids": input_ids_list})
print(packaged_pretrain_dataset)

## ------------------------------------------------------##
packaged_pretrain_dataset.to_parquet("./data/packaged_pretrain_dataset.parquet")
