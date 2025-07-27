import gc
import torch
import warnings

from transformers import LlamaConfig
from transformers import LlamaForCausalLM
from transformers import LlamaTokenizer
from transformers import TextStreamer
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, AutoConfig
from copy import deepcopy


warnings.filterwarnings('ignore')

def fix_torch_seed(seed = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

fix_torch_seed()

## ------------------------------------------------------##
config = LlamaConfig()
print(config)

## ------------------------------------------------------##
config.num_hidden_layers = 12
config.hidden_size = 1024
config.intermediate_size = 4096
config.num_key_value_heads = 8
config.torch_dtype = "bfloat16"
config.use_cache = False
print(config)

## ------------------------------------------------------##
model = LlamaForCausalLM(config)
print(model)

## ------------------------------------------------------##
def print_nparams(model):
    """
    Calculate the total number of model parameters
    """
    nparams = sum(p.numel() for p in model.parameters())
    print(f"The total number of parameters is: {nparams}")

print_nparams(model)

## ------------------------------------------------------##
layer_name = "model.layers.0.self_attn.q_proj.weight"

for name, param in model.named_parameters():
    if name == layer_name:
        print(f"First 30 weights of layer '{layer_name}':")
        print(param.data.view(-1)[ : 30])
        break

## ------------------------------------------------------##
model_dir = "./models/SOLAR-10.7B-v1.0"
tokenizer = LlamaTokenizer.from_pretrained(model_dir)

prompt = "I am an engineer. I love"

inputs = tokenizer(prompt, return_tensors = "pt").to(model.device)

streamer = TextStreamer(
                        tokenizer,
                        skip_prompt = True,
                        skip_special_tokens = True
                        )

outputs = model.generate(
                        **inputs,
                        streamer = streamer,
                        use_cache = True,
                        max_new_tokens = 128,
                        do_sample = False
                        )

## ------------------------------------------------------##
del model
del streamer
del outputs

gc.collect()

## ------------------------------------------------------##
model_name_or_path = "./models/TinySolar-248m-4k"
model = AutoModelForCausalLM.from_pretrained(
                                                        model_name_or_path,
                                                        device_map = "cpu",
                                                        torch_dtype = torch.bfloat16,
                                                        )

## ------------------------------------------------------##
del model

gc.collect()

## ------------------------------------------------------##
model_name_or_path = "./models/TinySolar-248m-4k"
model = AutoModelForCausalLM.from_pretrained(
                                                        model_name_or_path,
                                                        device_map = "cpu",
                                                        torch_dtype = torch.bfloat16,
                                                        )
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

## ------------------------------------------------------##
print(model)

## ------------------------------------------------------##
print_nparams(model)

## ------------------------------------------------------##
layers = model.model.layers
model.model.layers = layers[ : 5] + layers[-5 : ]

config = AutoConfig.from_pretrained(
                                    model_name_or_path,
                                    num_hidden_layers = len(model.model.layers),
                                    )
model.config = config

print_nparams(model)

## ------------------------------------------------------##
del model

gc.collect()

## ------------------------------------------------------##
config = LlamaConfig(
                    num_hidden_layers = 16,
                    hidden_size = 1024,
                    intermediate_size = 4096,
                    num_attention_heads = 32,
                    num_key_value_heads = 8,
                    torch_dtype = "bfloat16",
                    use_cache = False
                    )
print(config)

## ------------------------------------------------------##
model = LlamaForCausalLM(config)
model = model.to(dtype = torch.bfloat16)
print_nparams(model)

## ------------------------------------------------------##
model_name_or_path = "upstage/TinySolar-248m-4k"
pretrained_model = AutoModelForCausalLM.from_pretrained(
                                                        model_name_or_path,
                                                        device_map = "cpu",
                                                        torch_dtype = torch.bfloat16,
                                                        )
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

print_nparams(pretrained_model)

## ------------------------------------------------------##
model.model.layers = deepcopy(pretrained_model.model.layers[ : -4]) \
                    + deepcopy(pretrained_model.model.layers[4 : ])

model.model.embed_tokens = deepcopy(pretrained_model.model.embed_tokens)

model.lm_head = deepcopy(pretrained_model.lm_head)

print(model.config)

## ------------------------------------------------------##
print_nparams(model)

## ------------------------------------------------------##
prompt = "I am an engineer. I love"

inputs = tokenizer(prompt, return_tensors = "pt").to(model.device)

streamer = TextStreamer(
                        tokenizer,
                        skip_prompt = True,
                        skip_special_tokens = True
                        )

outputs = model.generate(
                        **inputs,
                        streamer = streamer,
                        use_cache = True,
                        max_new_tokens = 128,
                        do_sample = False
                        )

## ------------------------------------------------------##
model.save_pretrained('./data/TinySolar-308m-4k-init')
