import torch
from datasets import load_dataset
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import transformers
from trl import SFTTrainer
from data import dataloader
from dataloader import Tokenizer_preprocess
from prompt import generate_completion
import argparse

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

ref2 = args.ref2
model_id = "google-t5/t5-small"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})
tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj","o_proj","gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

peft_model = get_peft_model(model, lora_config)
print_trainable_parameters(peft_model)
model.add_adapter(lora_config, adapter_name="adapter")

#Here I reload the model and specify it should be loaded in a single GPU to avoid errors" Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! when resuming training"
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})

tokenizer.pad_token = tokenizer.eos_token
torch.cuda.empty_cache()

model.load_state_dict(torch.load("./paraphrase.pt"))
                     
data_path = args.file_name if args.use_file else 'data/quoraquestionpair10k.json'
train_df, test_df, train_data, test_data = Tokenizer_preprcoess_nosplit(data_path, tokenizer, model, ref2)

##deleting unwanted memory
del train_df
del test_df

for input in train_data:
  result = generate_completion(input, model, tokenizer, ref2)
  print(f'Paraphrase : {result}\n\n')
  

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--ref2", type=bool, help=f"Are there 2 references?" default = False)
  parser.add_argument("--use-file", type=bool, deafault = False)
  parser.add_argument("--file-name", type=str, help=f"Name of the file to take inputs from")
  args = parser.parse_args()
