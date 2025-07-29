import datasets
import torch
import transformers
import warnings

from dataclasses import dataclass, field
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM
from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers import AutoTokenizer, TextStreamer


warnings.filterwarnings('ignore')

## ------------------------------------------------------##
pretrained_model = AutoModelForCausalLM.from_pretrained(
                                                        "./models/TinySolar-308m-4k-init",
                                                        device_map = "cpu",
                                                        torch_dtype = torch.bfloat16,
                                                        use_cache = False,
                                                        )

## ------------------------------------------------------##
pretrained_model

## ------------------------------------------------------##
class CustomDataset(Dataset):
    def __init__(self, args, split = "train"):
        """
        Initializes the custom dataset object.
        """
        self.args = args
        self.dataset = datasets.load_dataset(
                                            "parquet",
                                            data_files = args.dataset_name,
                                            split = split
                                            )


    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.dataset)


    def __getitem__(self, idx):
        """
        Retrieves a single data sample from the dataset
        at the specified index
        """
        input_ids = torch.LongTensor(self.dataset[idx]["input_ids"])
        labels = torch.LongTensor(self.dataset[idx]["input_ids"])

        return {"input_ids": input_ids, "labels": labels}

## ------------------------------------------------------##
@dataclass
class CustomArguments(transformers.TrainingArguments):
    dataset_name: str = field(default = "./parquet/packaged_pretrain_dataset.parquet")
    num_proc: int = field(default = 1)
    max_seq_length: int = field(default = 32)

    seed: int = field(default  = 0)
    optim: str = field(default = "adamw_torch")
    max_steps: int = field(default = 30)

    learning_rate: float = field(default = 5e-5)
    weight_decay: float = field(default = 0)
    warmup_steps: int = field(default = 10)
    lr_scheduler_type: str = field(default = "linear")
    gradient_checkpointing: bool = field(default = True)
    dataloader_num_workers: int = field(default = 2)
    bf16: bool = field(default = True)
    gradient_accumulation_steps: int = field(default = 1)

    logging_steps: int = field(default = 3)
    report_to: str = field(default = "none")

    # save_strategy: str = field(default = "steps")
    # save_steps: int = field(default = 3)
    # save_total_limit: int = field(default = 2)

## ------------------------------------------------------##
parser = transformers.HfArgumentParser(CustomArguments)
args, = parser.parse_args_into_dataclasses(args = ["--output_dir", "output"])

## ------------------------------------------------------##
train_dataset = CustomDataset(args = args)

## ------------------------------------------------------##
print("Input shape: ", train_dataset[0]['input_ids'].shape)

## ------------------------------------------------------##
class LossLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs = None, **kwargs):
        if logs is not None:
            self.logs.append(logs)

    def __init__(self):
        self.logs = []


loss_logging_callback = LossLoggingCallback()

## ------------------------------------------------------##
trainer = Trainer(
                model = pretrained_model,
                args = args,
                train_dataset = train_dataset,
                eval_dataset = None,
                callbacks = [loss_logging_callback]
                )

trainer.train()

## ------------------------------------------------------##
save_strategy: str = field(default = "steps")
save_steps: int = field(default = 3)
save_total_limit: int = field(default = 2)

## ------------------------------------------------------##
model_name_or_path = "./models/TinySolar-248m-4k"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

## ------------------------------------------------------##
model_name_or_path = "./models/upstage/output/checkpoint-10000"
model2 = AutoModelForCausalLM.from_pretrained(
                                            model_name_or_path,
                                            device_map = "auto",
                                            torch_dtype = torch.bfloat16,
                                            )

## ------------------------------------------------------##
prompt = "I am an engineer. I love"

inputs = tokenizer(prompt, return_tensors = "pt").to(model2.device)

streamer = TextStreamer(
                        tokenizer,
                        skip_prompt = True,
                        skip_special_tokens = True
                        )

outputs = model2.generate(
                        **inputs,
                        streamer = streamer,
                        use_cache = True,
                        max_new_tokens = 64,
                        do_sample = True,
                        temperature = 1.0,
                        )
