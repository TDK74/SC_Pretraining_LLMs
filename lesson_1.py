import torch
import warnings

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import TextStreamer


warnings.filterwarnings('ignore')

## ------------------------------------------------------##
def fix_torch_seed(seed = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

fix_torch_seed()

## ------------------------------------------------------##
model_path_or_name = "./models/TinySolar-248m-4k"

## ------------------------------------------------------##
tiny_general_model = AutoModelForCausalLM.from_pretrained(
                                                        model_path_or_name,
                                                        device_map = "cpu",
                                                        torch_dtype = torch.bfloat16
                                                        )

## ------------------------------------------------------##
tiny_general_tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)

## ------------------------------------------------------##
prompt = "I am an engineer. I love"

## ------------------------------------------------------##
inputs = tiny_general_tokenizer(prompt, return_tensors = "pt")

## ------------------------------------------------------##
streamer = TextStreamer(
                        tiny_general_tokenizer,
                        skip_prompt = True,
                        skip_special_tokens = True
                        )

## ------------------------------------------------------##
outputs = tiny_general_model.generate(
                                    **inputs,
                                    streamer = streamer,
                                    use_cache = True,
                                    max_new_tokens = 128,
                                    do_sample = False,
                                    temperature = 0.0,
                                    repetition_penalty = 1.1
                                    )

## ------------------------------------------------------##
prompt =  "def find_max(numbers):"

## ------------------------------------------------------##
inputs = tiny_general_tokenizer(prompt, return_tensors = "pt").to(tiny_general_model.device)

streamer = TextStreamer(
                        tiny_general_tokenizer,
                        skip_prompt = True,
                        skip_special_tokens = True
                        )

## ------------------------------------------------------##
outputs = tiny_general_model.generate(
                                    **inputs,
                                    streamer = streamer,
                                    use_cache = True,
                                    max_new_tokens = 128,
                                    do_sample = False,
                                    temperature = 0.0,
                                    repetition_penalty = 1.1
                                    )

## ------------------------------------------------------##
model_path_or_name = "./models/TinySolar-248m-4k-code-instruct"

## ------------------------------------------------------##
tiny_finetuned_model = AutoModelForCausalLM.from_pretrained(
                                                            model_path_or_name,
                                                            device_map = "cpu",
                                                            torch_dtype = torch.bfloat16,
                                                            )

tiny_finetuned_tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)

## ------------------------------------------------------##
prompt =  "def find_max(numbers):"

inputs = tiny_finetuned_tokenizer(prompt, return_tensors =  "pt").to(tiny_finetuned_model.device)

streamer = TextStreamer(
                        tiny_finetuned_tokenizer,
                        skip_prompt = True,
                        skip_special_tokens = True
                        )

outputs = tiny_finetuned_model.generate(
                                        **inputs,
                                        streamer = streamer,
                                        use_cache = True,
                                        max_new_tokens = 128,
                                        do_sample = False,
                                        temperature = 0.0,
                                        repetition_penalty = 1.1
                                        )

## ------------------------------------------------------##
model_path_or_name = "./models/TinySolar-248m-4k-py"

## ------------------------------------------------------##
tiny_custom_model = AutoModelForCausalLM.from_pretrained(
                                                        model_path_or_name,
                                                        device_map = "cpu",
                                                        torch_dtype = torch.bfloat16,
                                                        )

tiny_custom_tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)

## ------------------------------------------------------##
prompt = "def find_max(numbers):"

inputs = tiny_custom_tokenizer(prompt, return_tensors = "pt").to(tiny_custom_model.device)

streamer = TextStreamer(
                        tiny_custom_tokenizer,
                        skip_prompt = True,
                        skip_special_tokens = True
                        )

outputs = tiny_custom_model.generate(
                                    **inputs,
                                    streamer = streamer,
                                    use_cache = True,
                                    max_new_tokens = 128,
                                    do_sample = False,
                                    repetition_penalty = 1.1
                                    )

## ------------------------------------------------------##
def find_max(numbers):
    max = 0

    for num in numbers:
        if num > max:
            max = num

    return max

## ------------------------------------------------------##
find_max([1, 3, 5, 1, 6, 7, 2])
