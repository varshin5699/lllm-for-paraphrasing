from datasets import load_dataset
from prompt import generate_prompt, generate_prompt_2
import pandas
import os

data_path= "./quoraquestionpair10k.json"
def Tokenizer_preprocess(data_path, tokenizer, model, ref2 = False):
  train_ds, test_ds =load_dataset("json",data_files=datapath, split=[train,test])
  
  train_df= train_ds.to_pandas()
  test_df = test_ds.to_pandas()
  text_column = [generate_prompt(data_point) if not ref2 else generate_prompt_2(data_point) for data_point in train_ds]
  train_ds = train_ds.add_column("prompt", text_column)
  
  text_column = [generate_prompt(data_point) if not ref2 else generate_prompt_2(data_point) for data_point in test_ds]
  test_ds = test_ds.add_column("prompt", text_column) 
  
  train_ds = train_ds.shuffle(seed=1234)
  data_train = train_ds.map(lambda samples: tokenizer(samples["prompt"])  , batched = True)
  
  test_ds = test_ds.shuffle(seed=1234)
  data_test = test_ds.map(lambda samples: tokenizer(samples["prompt"])  , batched = True)

  return train_df, test_df, data_train, data_test

def Tokenizer_preprocess_2(data_path, tokenizer, model, ref2 = False):
  
  train_ds=load_dataset("json",data_files=data_path, split="train")
  text_column = [generate_prompt(data_point) if not ref2 else generate_prompt_2(data_point) for data_point in train_ds]
  train_ds = train_ds.add_column("prompt", text_column)
  train_df = train_ds.shuffle(seed=1234)
  data_train = train_ds.map(lambda samples: tokenizer(samples["prompt"])  , batched = True)
  return data_train

