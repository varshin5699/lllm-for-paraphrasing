def generate_prompt(data_point, return_out = True):
  text = 'Given Input in the form of questions, Generate a Paraphrase such that it conveys the same meaning as Input.\n\n'
  text += f'### Input:\n{data_point["qid1"]})\n\n'
  text += f'### Paraphrase:\n{data_point["qid2"] if return_out else ""}'

  return text

def generate_prompt_2(data_point, return_out = True):
  text = if return_out 'Given Input in the form of questions, and an Example Paraphrase, Generate a Paraphrase such that it conveys the same meaning as Input.\n\n' else 'Given Input in the form of questions, Generate a Paraphrase such that it conveys the same meaning as Input.\n\n' 
  text += f'### Input:\n{data_point["qid1"]})\n\n'
  text += if return_out f'### Example:\n{data_point["qid1"]})\n\n' else f''
  text += f'### Paraphrase:\n{data_point["qid2"] if return_out else ""}'

  return text

def generate_completion(query: str, model, tokenizer, use_cuda = True, ref2 = False) -> str:
  device = if use_cuda "cuda:0" else "cpu"
  prompt = if not ref2 generate_prompt(query, return_out=False) else generate_prompt_2(query, return_out=False)
  encode = tokenizer(prompt, return_tensors = "pt")
  model_inputs = encode.to(device)

  generation_=model.generate(**model_inputs, max_new_token = 1000, do_sample = True)
  decode = tokenizer.batch_decode(generation_)
  return (decode[0])
