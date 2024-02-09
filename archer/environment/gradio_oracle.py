from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers 
import torch
import gradio as gr
model_lm = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_lm)
model = AutoModelForCausalLM.from_pretrained(model_lm, torch_dtype = torch.bfloat16).to("cuda:0")
prompt_template = """<s>[INST]{user_message}[/INST]
"""

def predict(text):
    text = text.split(",,,,,")
    input_ids = tokenizer([prompt_template.format(user_message=t) for t in text], return_tensors = "pt").to("cuda:0")
    context_len = input_ids['attention_mask'].size(1)
    output = model.generate(**input_ids, max_new_tokens=16, 
                            do_sample = False, pad_token_id = tokenizer.eos_token_id)
    return ",,,,,".join(tokenizer.batch_decode(output[:, context_len:], skip_special_tokens=True))
demo = gr.Interface(fn=predict, inputs="textbox", outputs="textbox")
    
demo.launch(share=True) 