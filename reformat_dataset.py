import json

def reformat_dataset(input_file='results.json', output_file='Mddial.json'):
    # Read the input JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Process each conversation
    for conversation in data:
        # Get the diagnosis from the word field, with error handling
        if 'word' in conversation and conversation['word'] and len(conversation['word']) > 0:
            diagnosis = conversation['word'][0]
            # Add the diagnosis as the last line in the conversation
            conversation['lines'].append(f"Based on your symptoms, I diagnose you with {diagnosis}.")
        else:
            print(f"Warning: Conversation missing diagnosis word: {conversation}")
    
    # Write the updated data to the new output file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    reformat_dataset()
    print("Dataset reformatting completed successfully! New file created: Mddial.json") 