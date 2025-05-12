MISTRAL_TWENTY_QUESTIONS_TEMPLATE = """<s>[INST]Please play Twenty Questions with me. The possible hidden words are:
Esophagitis, Enteritis, Asthma, Coronary heart disease, Pneumonia, Rhinitis, Thyroiditis, Traumatic brain injury, Dermatitis, External otitis, Conjunctivitis, Mastitis.
Some examples are following:
Questions:
Do you experience Cough? Yes.
Do you experience Chest tightness? No.
Do you experience Chest tightness and shortness of breath? No.
Do you experience Pain behind the breastbone? No.
Do you experience Chest tightness? No.
Do you experience Hemoptysis? Yes.
Do you experience Expectoration? Yes.
You guessed the correct word! You win!

Please continue this conversation by completing the next question. 
{obs}
Please answer in the following format:
{
"Question": "Your Question",
}
The possible hidden words are:
Esophagitis, Enteritis, Asthma, Coronary heart disease, Pneumonia, Rhinitis, Thyroiditis, Traumatic brain injury, Dermatitis, External otitis, Conjunctivitis, Mastitis.[/INST]
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