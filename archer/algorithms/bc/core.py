import numpy as np
import torch
softmax = torch.nn.Softmax(dim=-1)
def bc_loss(model, llama_model, tokenizer, observation, action, noise_std = 0, **kwargs):
    """
    obs_ids: the dict from tokenizer output of the state
    action_ids: the dict from tokenizer output of the state
    """
    prompt = model.get_action(observation).to(llama_model.device)
    action_ids = tokenizer(action, return_tensors='pt', padding=True).to(llama_model.device)
    obs_ids = tokenizer(observation, return_tensors='pt', padding=True).to(llama_model.device)
    noise = torch.normal(torch.zeros(prompt.size()), torch.ones(prompt.size())*noise_std).to(prompt.device)


    action_embeds = llama_model.get_input_embeddings()(action_ids["input_ids"]).detach()
    obs_embeds = llama_model.get_input_embeddings()(obs_ids["input_ids"]).detach()
    input_embeds = torch.cat([obs_embeds, prompt+noise, action_embeds], dim = 1)
    attention_mask = torch.cat([obs_ids["attention_mask"], \
                                torch.ones(prompt.size(0), prompt.size(1), device = prompt.device).long() ,action_ids["attention_mask"]],\
                              dim = 1)


    outputs = llama_model(inputs_embeds = input_embeds, attention_mask = attention_mask)
    prediction_probs = softmax(outputs.logits)
    selected_prediction_probs = torch.take_along_dim(prediction_probs[:, obs_ids["attention_mask"].size(1)+prompt.size(1)-1:-1], action_ids["input_ids"].unsqueeze(2), dim=2).squeeze(2)
    logsum_probs = torch.sum(torch.log(selected_prediction_probs)*action_ids["attention_mask"], dim = 1)
    return  - logsum_probs.mean()


def plain_bc_loss(model, tokenizer, observation, action, **kwargs):
    """
    obs_ids: the dict from tokenizer output of the state
    action_ids: the dict from tokenizer output of the state
    """
    action_ids = tokenizer(action, return_tensors='pt', padding=True).to(model.device)
    obs_ids = tokenizer(observation, return_tensors='pt', padding=True).to(model.device)

    # action_embeds = model.get_input_embeddings()(action_ids["input_ids"]).detach()
    # obs_embeds = model.get_input_embeddings()(obs_ids["input_ids"]).detach()
    # input_embeds = torch.cat([obs_embeds, action_embeds], dim = 1)
    input_ids = torch.cat([obs_ids["input_ids"], action_ids["input_ids"]], dim = 1)
    attention_mask = torch.cat([obs_ids["attention_mask"] ,action_ids["attention_mask"]],\
                              dim = 1)
    outputs = model(input_ids=input_ids, attention_mask = attention_mask)
    prediction_probs = softmax(outputs.logits)
    selected_prediction_probs = torch.take_along_dim(prediction_probs[:, obs_ids["attention_mask"].size(1)-1:-1], action_ids["input_ids"].unsqueeze(2), dim=2).squeeze(2)
    logsum_probs = torch.sum(torch.log(selected_prediction_probs)*action_ids["attention_mask"], dim = 1)
    return  - logsum_probs.mean()