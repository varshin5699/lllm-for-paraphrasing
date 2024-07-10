from datasets import load_dataset
import pandas
import os

data_path= "./quoraquestionpair10k.json"
def Tokenizer_preprocess(data_path, tokeniser, model):
  train_ds, test_ds =load_dataset("json",data_files=datapath, split=[train,test])
  
  train_df= train_ds.to_pandas()
  test_df = test_ds.to_pandas()
  
  train_df = train_df.shuffle(seed=1234)
  data_train = train_df.map(lambda samples: tokenizer(samples["prompt"])  , batched = True)
  
  test_df = test_df.shuffle(seed=1234)
  data_test = test_df.map(lambda samples: tokenizer(samples["prompt"])  , batched = True)

  return data_train,data_test
  



