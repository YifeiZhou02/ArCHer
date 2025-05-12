import random
from typing import Optional, List, Tuple
import time
import logging
logging.getLogger().setLevel(logging.CRITICAL)
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import concurrent.futures

# Danh sách các triệu chứng cho bệnh mà bạn đã cung cấp
DISEASE_SYMPTOMS = {
    "Esophagitis": ["Acid reflux", "Anorexia", "Bitter", "Bloating", "Blood in the tears", "Burning sensation behind the breastbone", "Chest tightness", "Constipation", "Cough", "Diarrhea", "Difficulty breathing", "Dizziness", "Edema", "Expectoration", "Feel sick and vomit", "Frequent urination", "Hard to swallow", "Hemoptysis", "Hiccough", "Hiccup", "Increased stool frequency", "Loss of appetite", "Nausea", "Pain behind the breastbone", "Palpitations", "Pharynx discomfort", "Runny nose", "Shortness of breath", "Stomach ache", "Stuffy nose", "Thin", "Thin white moss", "Thirst", "Vomiting"],
    "Enteritis": ["Abdominal pain and diarrhea", "Anorexia", "Bloating", "Bloody stools", "Body aches", "Constipation", "Cough", "Decreased urine output", "Defecate", "Diarrhea", "Dizziness", "Edema", "Expectoration", "Fatigue", "Fear of cold", "Feel sick and vomit", "Fever", "First degree swelling of bilateral tonsils", "Frequent urination", "Headache", "Increased stool frequency", "Loss of appetite", "Nausea", "Palpitations", "Pharynx discomfort", "Poor spirits", "Runny nose", "Sneeze", "Stomach ache", "Stuffy nose", "Thirst", "Vomiting"],
    "Asthma": ["Acid reflux", "Body aches", "Chest tightness", "Chest tightness and shortness of breath", "Chills", "Chills and fever", "Cough", "Difficulty breathing", "Dizziness", "Dizzy", "Expectoration", "Fatigue", "Fear of cold", "Fever", "First degree swelling of bilateral tonsils", "Itching", "Joint pain", "Loss of appetite", "Nasal congestion", "Pain behind the breastbone", "Palpitations", "Pharynx discomfort", "Poor sleep", "Poor spirits", "Rash", "Runny nose", "Shortness of breath", "Sneeze", "Stomach ache", "Stuffy nose", "Sweating"],
    "Coronary heart disease": ["Acid reflux", "Backache", "Bloating", "Burning sensation behind the breastbone", "Chest tightness", "Chest tightness and shortness of breath", "Chills", "Chills and fever", "Cough", "Diarrhea", "Difficulty breathing", "Dizziness", "Dizzy", "Edema", "Expectoration", "Fatigue", "Fear of cold", "Feel sick and vomit", "Fever", "Frequent urination", "Hazy", "Headache", "Hemoptysis", "Hiccup", "Nausea", "Night sweats", "Pain behind the breastbone", "Pain in front of neck", "Palpitations", "Pharynx discomfort", "Runny nose", "Shortness of breath", "Stomach ache", "Stuffy nose", "Sweating", "Syncope", "Tinnitus", "Urgency", "Vomiting", "Waist pain"],
    "Pneumonia": ["Acid reflux", "Anorexia", "Bitter", "Chest tightness", "Chest tightness and shortness of breath", "Chills", "Chills and fever", "Consciousness disorder", "Cough", "Diarrhea", "Difficulty breathing", "Dizziness", "Expectoration", "Fatigue", "Fever", "First degree swelling of bilateral tonsils", "Headache", "Loss of appetite", "Nausea", "Night sweats", "Oliguria", "Pain behind the breastbone", "Palpitations", "Pharynx discomfort", "Poor sleep", "Poor spirits", "Runny nose", "Shortness of breath", "Sneeze", "Stomach ache", "Stuffy nose", "Vomiting"],
    "Rhinitis": ["	Blood in the tears", "Body aches", "Chest tightness", "Cough", "Difficulty breathing", "Diplopia", "Dizziness", "Dizzy", "Edema", "Expectoration", "Fatigue", "Fever", "First degree swelling of bilateral tonsils", "Headache", "Hemoptysis", "Itchy eyes", "Nasal congestion", "Nasal mucosal congestion", "Nose bleeding", "Pain behind the breastbone", "Palpitations", "Pharynx discomfort", "Runny nose", "Shortness of breath", "Sleep disorder", "Sneeze", "Snore", "Stuffy nose", "Swelling of both nasal mucosa", "Thin", "Thin white moss", "Thirst"],
    "Thyroiditis": ["Acid reflux", "Afraid of cold", "Afraid of heat", "Anorexia", "Backache", "Bloating", "Chest tightness", "Chest tightness and shortness of breath", "Constipation", "Cough", "Cry", "Dizziness", "Dizzy", "Edema", "Eye swelling", "Fatigue", "Fever", "Frequent urination", "Hard to swallow", "Headache", "Hiccough", "Hoarse", "Loss of appetite", "Mild thyroid enlargement", "Pain behind the breastbone", "Pain in front of neck", "Palpitations", "Pharynx discomfort", "Poor sleep", "Poor spirits", "Right earache", "Runny nose", "Shaking hands", "Sleep disorder", "Sneeze", "Stomach ache", "Stuffy nose", "Sweating", "Thin", "Thirst", "Vision loss"],
    "Traumatic brain injury": ["Chest tightness", "Consciousness disorder", "Cough", "Cry", "Difficulty breathing", "Dizziness", "Dizzy", "Earache", "Fatigue", "Fear of cold", "Feel sick and vomit", "Fever", "Head trauma pain", "Headache", "Headache and dizziness", "Nausea", "Pain in front of neck", "Palpitations", "Poor sleep", "Poor spirits", "Stomach ache", "Tinnitus", "Unconsciousness", "Vertigo", "Vomiting", "Waist pain"],
    "Dermatitis": ["Acid reflux", "Bloating", "Congestion of skin and mucous membranes", "Cough", "Dizziness", "Fever", "Itching", "Itchy and uncomfortable eyes", "Jealous", "Loss of appetite", "Nausea", "Papule", "Pharynx discomfort", "Poor sleep", "Rash", "Redness", "Runny nose", "Sneeze", "Snore", "Stomach ache", "Stuffy nose", "Suppuration", "Vomiting", "Waist pain"],
    "External otitis": ["Cough", "Cry", "Dizziness", "Ear itching", "Earache", "Expectoration", "Fever", "Headache", "Hearing loss", "Itchy eyes", "Nausea", "Pharynx discomfort", "Poor sleep", "Redness", "Right earache", "Runny nose", "Sleep disorder", "Stomach ache", "Stuffy nose", "Tinnitus", "Vertigo", "Vomiting"],
    "Conjunctivitis": ["Cough", "Cry", "Edema", "Eye pain", "Eye swelling", "Fever", "Headache", "Itchy and uncomfortable eyes", "Itchy eyes", "Jealous", "Loss of appetite", "Nausea", "Pharynx discomfort", "Photophobia", "Redness", "Runny nose", "Stomach ache", "Stuffy nose", "Vision loss", "Vomiting"],
    "Mastitis": ["Bloating", "Body aches", "Breast tenderness", "Chills", "Chills and fever", "Cough", "Diarrhea", "Dizziness", "Dizzy", "Expectoration", "Fatigue", "Fear of cold", "Feel sick and vomit", "Fever", "Headache", "Loss of appetite", "Lumbago", "Nausea", "Pharynx discomfort", "Redness", "Runny nose", "Stomach ache", "Stuffy nose", "Thin", "Thin white moss"]
}

DEFAULT_DISEASE_LIST = list(DISEASE_SYMPTOMS.keys())
INITIAL_PATIENT_STATEMENT = "Patient: Hello doctor, I'm not feeling well."

class MDDialEnv():
    def __init__(
        self,
        disease_list: Optional[List[str]] = None,
        disease_symptoms: Optional[dict] = None,
        max_conversation_length: int = 20,
    ):
        # Sử dụng các biến toàn cục nếu không được cung cấp
        self.disease_list = [disease.lower() for disease in disease_list] if disease_list is not None else [disease.lower() for disease in DEFAULT_DISEASE_LIST]
        self.disease_symptoms = {k.lower(): [s.lower() for s in v] for k, v in disease_symptoms.items()} if disease_symptoms is not None else {k.lower(): [s.lower() for s in v] for k, v in DISEASE_SYMPTOMS.items()}
        self.max_conversation_length = max_conversation_length
        self.random = random.Random(None)
        self.count = 0
        self.curr_disease = None
        self.history = ''
        self.done = True

    def reset(self, idx: Optional[int] = None) -> str:
        self.count = 0
        if idx is not None:
            if 0 <= idx < len(self.disease_list):
                self.curr_disease = self.disease_list[idx]
            else:
                raise ValueError(f"Index {idx} is out of the disease list bounds.")
        else:
            if self.disease_list:
                self.curr_disease = self.random.choice(self.disease_list)
            else:
                self.curr_disease = None

        self.history = INITIAL_PATIENT_STATEMENT + "\n"
        self.done = False
        return self.history

    def _get_answer_to_question(self, question: str) -> str:
        """
        Simulates the patient's 'Yes'/'No' answer based on the symptoms of the current disease.
        The answer will be in English.
        """
        if self.curr_disease and self.disease_symptoms:
            question_lower = question.lower()
            question_words = set(question_lower.replace('?', '').replace('.', '').split())
            current_symptoms = set(self.disease_symptoms.get(self.curr_disease, []))

            for symptom in current_symptoms:
                symptom_words = set(symptom.split())
                if symptom_words.issubset(question_words):
                    return "Yes."
            return "No."
        else:
            return random.choice(["Yes.", "No."])

    def step(self, question: str) -> Tuple[str, int, bool]:
        """
        Performs a dialogue turn (doctor asks a question).

        Args:
            question: The doctor's question.

        Returns:
            Tuple[str, int, bool]:
                - history: The updated dialogue history.
                - reward: The reward for the current step (always 0 here).
                - done: The environment's done state (False until diagnosis or max turns).
        """
        if self.done:
            return self.history, 0, True

        answer = self._get_answer_to_question(question)
        self.history += f"Doctor: {question}\nPatient: {answer}\n"
        self.count += 1

        if self.count >= self.max_conversation_length:
            self.done = True

        return self.history, 0, self.done

    def diagnose(self, diagnosis_attempt: str) -> Tuple[str, int, bool]:
        """
        The doctor makes a final diagnosis.

        Args:
            diagnosis_attempt: The disease diagnosis made by the agent.

        Returns:
            Tuple[str, int, bool]:
                - history: The updated dialogue history with the diagnosis.
                - reward: The reward for the diagnosis.
                - done: The environment's done state (always True after diagnosis).
        """
        if self.done:
            return self.history, 0, True

        is_correct = False
        if self.curr_disease:
            if diagnosis_attempt.strip().lower() == self.curr_disease.lower():
                is_correct = True

        reward = 0
        if not is_correct:
            reward = -1

        self.done = True
        self.history += f"Doctor makes a diagnosis: {diagnosis_attempt}\n"

        return self.history, reward, self.done

    def copy(self):
        return MDDialEnv(
            disease_list=self.disease_list,
            disease_symptoms=self.disease_symptoms,
            max_conversation_length=self.max_conversation_length,
        )

class BatchedMDDialEnv():
    def __init__(
        self,
        env_load_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        device=None,
        max_conversation_length: int = 20,
        bsize: int = 32,
    ):
        self.bsize = bsize
        # Sử dụng trực tiếp các biến toàn cục
        self.env_list = [MDDialEnv(max_conversation_length=max_conversation_length) for _ in range(bsize)]

        # Initialize the language model for generating 'Yes'/'No' answers
        if env_load_path and cache_dir and device is not None:
            self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
            self.model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small", cache_dir=cache_dir).to(device)
            if env_load_path:
                try:
                    self.model.load_state_dict(torch.load(env_load_path, map_location=device)['model_state_dict'])
                    print(f"Loaded model weights from: {env_load_path}")
                except FileNotFoundError:
                    print(f"Warning: Model file not found at {env_load_path}. The model will use default weights.")
        else:
            self.tokenizer = None
            self.model = None
            print("Note: No model path, cache_dir, or device provided. `generate_answers` will use the symptom-based logic.")

    def generate_answers(self, questions: List[str]) -> List[str]:
        """
        Uses the language model or symptom-based logic to generate 'Yes'/'No' answers for a batch of questions.
        The answers will be in English.
        """
        if self.tokenizer and self.model:
            inputs = []
            for i, question in enumerate(questions):
                curr_disease = self.env_list[i].curr_disease
                if curr_disease:
                    # Create a prompt for the model.
                    # Note: The model needs to be finetuned on relevant data to provide accurate medical dialogue responses.
                    prompt = f"The patient's disease is {curr_disease}. Question: {question} Answer with 'Yes.' or 'No.'."
                else:
                    prompt = f"Question: {question} Answer with 'Yes.' or 'No.'."
                inputs.append(prompt)

            encoder_ids = self.tokenizer(inputs, padding=True, return_tensors='pt').to(self.model.device)
            output_ids = self.model.generate(
                input_ids=encoder_ids['input_ids'],
                attention_mask=encoder_ids['attention_mask'],
                max_new_tokens=16,
                do_sample=False
            )
            answers = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            # Process the model's output to ensure it's 'Yes.' or 'No.'
            processed_answers = []
            for i, ans in enumerate(answers):
                ans_lower = ans.strip().lower()
                if 'yes' in ans_lower:
                    processed_answers.append("Yes.")
                elif 'no' in ans_lower:
                    processed_answers.append("No.")
                else:
                    # Fallback: use the symptom-based logic if the model's answer is unclear
                    processed_answers.append(self.env_list[i]._get_answer_to_question(questions[i]))
            return processed_answers
        else:
            # Fallback: use the symptom-based logic
            return [env._get_answer_to_question(q) for env, q in zip(self.env_list, questions)]

    def reset(self, idx: Optional[int] = None) -> List[str]:
        return [env.reset(idx) for env in self.env_list]

    def step(self, questions: List[str]) -> List[Tuple[str, int, bool]]:
        """
        Performs a dialogue turn for a batch of environments (doctor asks questions).

        Args:
            questions: A list of questions (one for each environment in the batch).

        Returns:
            List[Tuple[str, int, bool]]: A list of results from each environment (history, reward, done).
        """
        if len(questions) != self.bsize:
            raise ValueError(f"The size of 'questions' ({len(questions)}) must be equal to 'bsize' ({self.bsize}).")

        answers = self.generate_answers(questions)

        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            jobs = [executor.submit(env.step, q) for env, q in zip(self.env_list, questions)]
            # Update the history with the generated answers
            for i, (history, reward, done) in enumerate([job.result() for job in jobs]):
                # The answer is already added in `env.step`, we just need to return the result
                results.append((self.env_list[i].history, reward, done))
        return results

    def diagnose_batch(self, diagnosis_attempts: List[str]) -> List[Tuple[str, int, bool]]:
        """
        The doctor makes a final diagnosis for a batch of environments.

        Args:
            diagnosis_attempts: A list of diagnoses (one for each environment in the batch).

        Returns:
            List[Tuple[str, int, bool]]: A list of results from each environment (history, reward, done).
        """
        if len(diagnosis_attempts) != self.bsize:
            raise ValueError(f"The size of 'diagnosis_attempts' ({len(diagnosis_attempts)}) must be equal to 'bsize' ({self.bsize}).")

        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            jobs = [executor.submit(env.diagnose, diag) for env, diag in zip(self.env_list, diagnosis_attempts)]
            results = [job.result() for job in jobs]
        return results