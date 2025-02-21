from datasets import Dataset
from peft import LoraConfig
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.tokenization_utils_base import BatchEncoding
import os
import torch
import argparse

# Disable wandb for offline training
import wandb
os.environ["WANDB_DISABLED"]="true"
wandb.init(mode="disabled")


parser = argparse.ArgumentParser(
  prog="CausalLM Unsupervised Fine-Tune using LoRA",
  description="Loads dataset, defines training arguments, and runs huggingface training loop for CausalLM. This was set up using a .tsv dataset of text documents, as well as the Llama3.2-3B model."
)

parser.add_argument('model_id')
parser.add_argument('dataset')
parser.add_argument('output_dir')
parser.add_argument('--seed', nargs='?', const=42, type=int)

args = parser.parse_args()


#%% Dataset Loading

dataset = Dataset.load_from_disk(args.dataset)

# Split into train and eval
train_test_dataset = dataset.train_test_split(0.2, seed=args.seed)
train_dataset = train_test_dataset['train']
eval_dataset = train_test_dataset['test']

#%% Model Setup

# Specify how to quantize the model
quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
)

device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None

# Set up model parameters and pass in the quantization and device configs
model_kwargs = dict(
    # attn_implementation="flash_attention_2", # set this to True if your GPU supports it (Flash Attention drastically speeds up model computations)
    torch_dtype="auto",
    use_cache=False,
    quantization_config=quantization_config.to_dict(),
    device_map=device_map
)

# Load the model
model = AutoModelForCausalLM.from_pretrained(args.model_id, model_kwargs)

# Set up parameter efficient fine-tuning (LoRA in this case)
# These parameters were taken as recommendation from the LoRA paper: https://doi.org/10.48550/arXiv.2106.09685
peft_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

# Apply peft (adapter) settings
model.add_adapter(peft_config)

# Set up training arguments
training_args = TrainingArguments(
  args.output_dir,
  learning_rate=2e-5,
  weight_decay=0.01,
  push_to_hub=False,
	fp16=True, # specify bf16=True instead when training on GPUs that support bf16
	do_eval=True,
	evaluation_strategy="epoch",
	gradient_accumulation_steps=128,
	gradient_checkpointing=True,
	gradient_checkpointing_kwargs={"use_reentrant": False},
	log_level="info",
	logging_steps=5,
	logging_strategy="steps",
	lr_scheduler_type="cosine",
	max_steps=-1,
	num_train_epochs=1,
	per_device_eval_batch_size=8,
	per_device_train_batch_size=8,
	report_to="none",
	save_strategy="no",
	save_total_limit=None,
	seed=args.seed,
)

#%% Training

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

train_result = trainer.train()
trainer.save_model(args.output_dir)
