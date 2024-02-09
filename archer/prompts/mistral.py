MISTRAL_TWENTY_QUESTIONS_TEMPLATE = """<s>[INST]Please play Twenty Questions with me. The possible hidden words are:
football, dog, banana, truck, pants, computer, piano, chair, pen, scissors.
Some examples are following:
Questions:
Is the object alive? Yes.
Is the object a mammal? No.
Is the object a plant? Yes.
Is the object edible? Yes.
Is the object a fruit? Yes.
Is the object a tropical fruit? Yes.
Is the object a banana? Yes.
You guessed the correct word! You win!

Please continue this conversation by completing the next question. 
{obs}
Please answer in the following format:
{
"Question": "Your Question",
}
The possible hidden words are:
football, dog, banana, truck, pants, computer, piano, chair, pen, scissors.[/INST]
"""

def mistral_twenty_questions_decode_actions(output):
    """
    Decode the actions from the output of the model.
    """
    actions = []
    for a in output:
        action = a.split('"Question":')[-1]
        action = action.split("?")[0] + "?"
        action = action.strip().replace('"', '')
        actions.append(action)
    return actions