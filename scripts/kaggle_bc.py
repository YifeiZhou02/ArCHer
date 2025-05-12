import torch
import transformers
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn
import numpy as np 
from torch.utils.data import Dataset, DataLoader
import json
import os
import random
from torch.nn.functional import softmax
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Simple dataset wrapper
class DummyDataset(Dataset):
    def __init__(self, buffer):
        self.buffer = buffer

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer[idx]

# Behavioral cloning loss function
def plain_bc_loss(model, tokenizer, observation, action, **kwargs):
    action_ids = tokenizer(action, return_tensors='pt', padding=True).to(model.device)
    obs_ids = tokenizer(observation, return_tensors='pt', padding=True).to(model.device)
    input_ids = torch.cat([obs_ids["input_ids"], action_ids["input_ids"]], dim=1)
    attention_mask = torch.cat([obs_ids["attention_mask"], action_ids["attention_mask"]], dim=1)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    prediction_probs = softmax(outputs.logits, dim=-1)
    selected_prediction_probs = torch.take_along_dim(
        prediction_probs[:, obs_ids["attention_mask"].size(1)-1:-1], 
        action_ids["input_ids"].unsqueeze(2), 
        dim=2
    ).squeeze(2)
    logsum_probs = torch.sum(torch.log(selected_prediction_probs)*action_ids["attention_mask"], dim=1)
    return -logsum_probs.mean()

def process_conversation(lines):
    """Process a conversation into observation-action pairs."""
    pairs = []
    current_context = "Doctor-Patient Conversation:\n"
    
    for i, line in enumerate(lines):
        if "Based on your symptoms" in line:
            # Skip the final diagnosis as it's not a question
            continue
            
        # Split the line into question and answer
        parts = line.split("?")
        if len(parts) != 2:
            continue
            
        question = parts[0].strip()
        answer = parts[1].strip()
        
        # Create the observation (context + previous Q&A)
        observation = current_context
        
        # Create the action (next question)
        action = f"{question}?\n"
        
        pairs.append({
            'observation': observation,
            'action': action
        })
        
        # Update context for next iteration
        current_context += f"Doctor: {question}?\nPatient: {answer}\n"
    
    return pairs

def main():
    # Configuration
    config = {
        'data_path': '/kaggle/input/twenty-questions/twenty_questions.json',
        'checkpoint_path': '/kaggle/working/gpt2_bc_20q.pt',
        'batch_size': 8,
        'iterations': 100,
        'grad_accum_steps': 4,
        'learning_rate': 5e-5,  # Reduced learning rate
        'patience': 5,
        'val_split': 0.1,
        'dropout_rate': 0.1,  # Added dropout
        'weight_decay': 0.01  # Added L2 regularization
    }

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained('gpt2').to(device)
    
    # Add dropout to the model
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = config['dropout_rate']
    
    # Add weight decay to optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # More aggressive learning rate scheduling
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        patience=2, 
        factor=0.2,  # More aggressive reduction
        min_lr=1e-6  # Minimum learning rate
    )
    
    tokenizer = AutoTokenizer.from_pretrained('gpt2', trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load and prepare data
    with open(config['data_path'], "r") as fb:
        raw_data = json.load(fb)
    
    data = []
    for conversation in raw_data:
        pairs = process_conversation(conversation["lines"])
        data.extend(pairs)
    
    random.shuffle(data)
    
    # Split into train and validation
    val_size = int(len(data) * config['val_split'])
    train_data = data[val_size:]
    val_data = data[:val_size]
    
    train_dataloader = DataLoader(DummyDataset(train_data), batch_size=config['batch_size'])
    val_dataloader = DataLoader(DummyDataset(val_data), batch_size=config['batch_size'])

    # Training loop
    print_step = 0
    grad_step = 0
    best_val_loss = float('inf')
    patience_counter = 0
    
    for i in range(config['iterations']):
        # Training phase
        model.train()
        train_losses = []
        for batch in tqdm(train_dataloader, desc=f"Training Iteration {i}"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            loss = plain_bc_loss(model, tokenizer, **batch)
            train_losses.append(loss.item())
            grad_step += 1
            loss.backward()
            
            if grad_step % config['grad_accum_steps'] == 0:
                print_step += 1
                optimizer.step()
                optimizer.zero_grad()
                
                if print_step % 100 == 0:
                    avg_loss = sum(train_losses[-100:]) / min(100, len(train_losses))
                    print(f"Iteration {i}, Step {print_step}, Loss: {avg_loss:.4f}")
                    print_step = 0
        
        # Validation phase
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                loss = plain_bc_loss(model, tokenizer, **batch)
                val_losses.append(loss.item())
        
        avg_val_loss = sum(val_losses) / len(val_losses)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'model_state_dict': model.cpu().state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, config['checkpoint_path'])
            model = model.to(device)
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"Early stopping triggered after {i+1} iterations")
                break

    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main() 