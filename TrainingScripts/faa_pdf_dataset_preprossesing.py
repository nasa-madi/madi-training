from datasets import Dataset
from peft import LoraConfig
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.tokenization_utils_base import BatchEncoding
import os
import torch
import argparse


parser = argparse.ArgumentParser(
  prog="FAA PDF Dataset Preprocessing",
  description="Converts faa pdf .tsv dataset into HuggingFace dataset with single 'text' field"
)

parser.add_argument('input_filename')
parser.add_argument('output_filename')
parser.add_argument('model_id')
parser.add_argument('--chunk_size', default=1800, type=int)
parser.add_argument('--chunk_overlap', default=200, type=int)
parser.add_argument('--tokenizer_max_length', default=2048, type=int)

args = parser.parse_args()

#%% Load tokenizer 

tokenizer = AutoTokenizer.from_pretrained(args.model_id)

# set pad_token_id equal to the eos_token_id if not set
if tokenizer.pad_token_id is None:
  tokenizer.pad_token_id = tokenizer.eos_token_id

tokenizer.model_max_length = args.tokenizer_max_length

#%% 

def load_faa_dataset(dataset_name: str) -> Dataset:
  """Custom function for loading in faa data as a huggingface Dataset"""

  import pandas as pd
  
  data = pd.read_csv(dataset_name, sep="\t")

  data = data.dropna()
  # Clean some of the FAA pdf data to remove newline characters and improper encoding characters
  data['text'] = data['text'].apply(lambda x: x.encode().decode('unicode_escape', errors='replace'))

  # Convert pandas DataFrame to huggingface Dataset
  dataset = Dataset.from_pandas(data)
  dataset = dataset.remove_columns('Unnamed: 0')
  dataset = dataset.remove_columns('name')
  dataset = dataset.remove_columns('__index_level_0__')

  tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

  # Chunk the data to reduce the size of each document. This will be based off the chunk size and overlap
  tokenized_chunked_dataset = tokenized_dataset.map(group_texts, batched=True)
    
  # Create plain text data back from tokenized ids
  text_chunked_dataset = tokenized_chunked_dataset.map(decode_function, batched=True)
  # texts = [tokenizer.decode(x) for x in tokenized_chunked_dataset['input_ids']]
  # text_chunked_dataset = Dataset.from_dict({'text': texts})
  text_chunked_dataset = text_chunked_dataset.remove_columns('input_ids')
  text_chunked_dataset = text_chunked_dataset.remove_columns('attention_mask')

  return text_chunked_dataset

def tokenize_function(examples: dict) -> BatchEncoding:
  """Apply tokenization to dataset column of col_name when using Dataset.map"""
  result = tokenizer(examples["text"])
  return result

def decode_function(examples: dict) -> BatchEncoding:
  """Decode to dataset column of input_ids"""
  examples["text"] = tokenizer.decode(examples["input_ids"])
  return examples

def group_texts(examples: dict) -> BatchEncoding:
  """Chunk a dataset into smaller documents of specific size and overlap"""
  concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
  total_length = len(concatenated_examples[list(examples.keys())[0]])
  total_length = (total_length // args.chunk_size) * args.chunk_size
  result = {
      k: [t[i - args.chunk_overlap: i + args.chunk_size + args.chunk_overlap] + [0] * (args.chunk_size + (args.chunk_overlap * 2) - len(t[i - args.chunk_overlap: i + args.chunk_size + args.chunk_overlap])) for i in range(0, total_length, args.chunk_size)]
      for k, t in concatenated_examples.items()
  }

  return result

#%%

# Loading a huggingface dataset
dataset = load_faa_dataset(args.input_filename)

dataset.save_to_disk(args.output_filename)