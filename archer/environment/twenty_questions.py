import random
from typing import Optional, Dict
import time
import logging
logging.getLogger().setLevel(logging.CRITICAL)
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import concurrent.futures
# openai.util.logger.setLevel(logging.WARNING)
PROMPT_TEMPLATE = 'You are playing a game called twenty questions with me. The rule of twenty question is that you are given a hidden word, and I am guessing what the word is within twenty questions. For every question, if it is an invalid question, you should answer "Invalid Question.". For any valid question, you should answer either "Yes." or "No.". Now the hidden word given to you is "{word}", and the question for the current round is "{question}". Your response is:'
DEFAULT_OBJECT_DICT = {
  "Esophagitis": ["Acid reflux", "Anorexia", "Bitter", "Bloating", "Blood in the tears", "Burning sensation behind the breastbone", "Chest tightness", "Constipation", "Cough", "Diarrhea", "Difficulty breathing", "Dizziness", "Edema", "Expectoration", "Feel sick and vomit", "Frequent urination", "Hard to swallow", "Hemoptysis", "Hiccough", "Hiccup", "Increased stool frequency", "Loss of appetite", "Nausea", "Pain behind the breastbone", "Palpitations", "Pharynx discomfort", "Runny nose", "Shortness of breath", "Stomach ache", "Stuffy nose", "Thin", "Thin white moss", "Thirst", "Vomiting"],
  "Enteritis": ["Abdominal pain and diarrhea", "Anorexia", "Bloating", "Bloody stools", "Body aches", "Constipation", "Cough", "Decreased urine output", "Defecate", "Diarrhea", "Dizziness", "Edema", "Expectoration", "Fatigue", "Fear of cold", "Feel sick and vomit", "Fever", "First degree swelling of bilateral tonsils", "Frequent urination", "Headache", "Increased stool frequency", "Loss of appetite", "Nausea", "Palpitations", "Pharynx discomfort", "Poor spirits", "Runny nose", "Sneeze", "Stomach ache", "Stuffy nose", "Thirst", "Vomiting"],
  "Asthma": ["Acid reflux", "Body aches", "Chest tightness", "Chest tightness and shortness of breath", "Chills", "Chills and fever", "Cough", "Difficulty breathing", "Dizziness", "Dizzy", "Expectoration", "Fatigue", "Fear of cold", "Fever", "First degree swelling of bilateral tonsils", "Itching", "Joint pain", "Loss of appetite", "Nasal congestion", "Pain behind the breastbone", "Palpitations", "Pharynx discomfort", "Poor sleep", "Poor spirits", "Rash", "Runny nose", "Shortness of breath", "Sneeze", "Stomach ache", "Stuffy nose", "Sweating"],
  "Coronary heart disease": ["Acid reflux", "Backache", "Bloating", "Burning sensation behind the breastbone", "Chest tightness", "Chest tightness and shortness of breath", "Chills", "Chills and fever", "Cough", "Diarrhea", "Difficulty breathing", "Dizziness", "Dizzy", "Edema", "Expectoration", "Fatigue", "Fear of cold", "Feel sick and vomit", "Fever", "Frequent urination", "Hazy", "Headache", "Hemoptysis", "Hiccup", "Nausea", "Night sweats", "Pain behind the breastbone", "Pain in front of neck", "Palpitations", "Pharynx discomfort", "Runny nose", "Shortness of breath", "Stomach ache", "Stuffy nose", "Sweating", "Syncope", "Tinnitus", "Urgency", "Vomiting", "Waist pain"],
  "Pneumonia": ["Acid reflux", "Anorexia", "Bitter", "Chest tightness", "Chest tightness and shortness of breath", "Chills", "Chills and fever", "Consciousness disorder", "Cough", "Diarrhea", "Difficulty breathing", "Dizziness", "Expectoration", "Fatigue", "Fever", "First degree swelling of bilateral tonsils", "Headache", "Loss of appetite", "Nausea", "Night sweats", "Oliguria", "Pain behind the breastbone", "Palpitations", "Pharynx discomfort", "Poor sleep", "Poor spirits", "Runny nose", "Shortness of breath", "Sneeze", "Stomach ache", "Stuffy nose", "Vomiting"],
  "Rhinitis": ["Blood in the tears", "Body aches", "Chest tightness", "Cough", "Difficulty breathing", "Diplopia", "Dizziness", "Dizzy", "Edema", "Expectoration", "Fatigue", "Fever", "First degree swelling of bilateral tonsils", "Headache", "Hemoptysis", "Itchy eyes", "Nasal congestion", "Nasal mucosal congestion", "Nose bleeding", "Pain behind the breastbone", "Palpitations", "Pharynx discomfort", "Runny nose", "Shortness of breath", "Sleep disorder", "Sneeze", "Snore", "Stuffy nose", "Swelling of both nasal mucosa", "Thin", "Thin white moss", "Thirst"],
  "Thyroiditis": ["Acid reflux", "Afraid of cold", "Afraid of heat", "Anorexia", "Backache", "Bloating", "Chest tightness", "Chest tightness and shortness of breath", "Constipation", "Cough", "Cry", "Dizziness", "Dizzy", "Edema", "Expectoration", "Eye swelling", "Fatigue", "Fever", "Frequent urination", "Hard to swallow", "Headache", "Hiccough", "Hoarse", "Loss of appetite", "Mild thyroid enlargement", "Pain behind the breastbone", "Pain in front of neck", "Palpitations", "Pharynx discomfort", "Poor sleep", "Poor spirits", "Right earache", "Runny nose", "Shaking hands", "Sleep disorder", "Sneeze", "Stomach ache", "Stuffy nose", "Sweating", "Thin", "Thirst", "Vision loss"],
  "Traumatic brain injury": ["Chest tightness", "Consciousness disorder", "Cough", "Cry", "Difficulty breathing", "Dizziness", "Dizzy", "Earache", "Fatigue", "Fear of cold", "Feel sick and vomit", "Fever", "Head trauma pain", "Headache", "Headache and dizziness", "Nausea", "Pain in front of neck", "Palpitations", "Poor sleep", "Poor spirits", "Stomach ache", "Tinnitus", "Unconsciousness", "Vertigo", "Vomiting", "Waist pain"],
  "Dermatitis": ["Acid reflux", "Bloating", "Chest tightness", "Congestion of skin and mucous membranes", "Cough", "Dizziness", "Fever", "Itching", "Itchy and uncomfortable eyes", "Jealous", "Loss of appetite", "Nausea", "Papule", "Pharynx discomfort", "Poor sleep", "Rash", "Redness", "Runny nose", "Sneeze", "Snore", "Stomach ache", "Stuffy nose", "Suppuration", "Vomiting", "Waist pain"],
  "External otitis": ["Cough", "Cry", "Dizziness", "Ear itching", "Earache", "Expectoration", "Fever", "Headache", "Hearing loss", "Itchy eyes", "Nausea", "Pharynx discomfort", "Poor sleep", "Redness", "Right earache", "Runny nose", "Sleep disorder", "Stomach ache", "Stuffy nose", "Tinnitus", "Vertigo", "Vomiting"],
  "Conjunctivitis": ["Cough", "Cry", "Edema", "Eye pain", "Eye swelling", "Fever", "Headache", "Itchy and uncomfortable eyes", "Itchy eyes", "Jealous", "Loss of appetite", "Nausea", "Pharynx discomfort", "Photophobia", "Redness", "Runny nose", "Stomach ache", "Stuffy nose", "Vision loss", "Vomiting"],
  "Mastitis": ["Bloating", "Body aches", "Breast tenderness", "Chills", "Chills and fever", "Cough", "Diarrhea", "Dizziness", "Dizzy", "Expectoration", "Fatigue", "Fear of cold", "Feel sick and vomit", "Fever", "Headache", "Loss of appetite", "Lumbago", "Nausea", "Pharynx discomfort", "Redness", "Runny nose", "Stomach ache", "Stuffy nose", "Thin", "Thin white moss"]
}

DEFAULT_OBJECT_LIST = sum([d for d in DEFAULT_OBJECT_DICT.values()], [])
# random.seed(42)
# DEFAULT_OBJECT_LIST = random.sample(DEFAULT_OBJECT_LIST, k=5)
# DEFAULT_OBJECT_LIST = [DEFAULT_OBJECT_LIST[i] for i in [1,11,21,31,41,51,61,71,81,91]]
INITIAL_STR = "Questions:\n"

class TwentyQuestionsEnv():
    def __init__(
        self, 
        # word_list,  
        max_conversation_length: int=20,
    ):
        self.word_list = DEFAULT_OBJECT_LIST
        self.word_list =[ list(map(lambda x: x.lower(), word.split(";"))) for word in self.word_list]
        self.max_conversation_length = max_conversation_length
        self.random = random.Random(None)
        self.count = 0
        self.curr_word = None
        self.history = ''
        self.done = True

    def is_correct(self, question):
        #check for the last word
        # cut out punctuations at the end
        while len(question) > 0 and not question[-1].isalpha():
            question = question[:-1]

        if len(question) == 0:
            return False
        guess = question.split(" ")[-1].lower()
        return guess in self.curr_word
    
    def _step(self, question, answer):
        if 'yes' in answer.strip().lower():
            answer = 'Yes.'
        elif 'no' in answer.strip().lower():
            answer = 'No.'
        else:
            # print("question:  " + question)
            # print('answer: '+ answer)
            # import IPython; IPython.embed()
            answer = 'Invalid Question.'
        if self.done:
            return None
        self.count+=1
        self.history += question + ' ' + answer + '\n'

        # trajectory = create_trajectory_from_history(self.curr_word, text_history + (answer_text,), self.max_conversation_length)
        # if self.count == self.max_conversation_length:
        #     print("The word was", self.curr_word[0])
        done = (answer.replace('.', '').lower() == 'yes') and self.is_correct(question)
        reward = -1
        if done:
            reward = 0
        self.done = done or self.count == self.max_conversation_length
        return  self.history, reward, self.done

    def reset(self, idx : Optional[int]=None):
        self.count = 0 
        # if self.curr_word is not None: 
        #     print("The word was ", self.curr_word)
        #     print("Next word...")
        if idx is not None:
            self.curr_word = self.word_list[idx]
        else:
            self.curr_word = self.random.choice(self.word_list)
        self.history = INITIAL_STR 
        self.done = False
        return INITIAL_STR
        # return (Text(INITIAL_STR, is_action=False),)

    def copy(self):
        return TwentyQuestionsEnv(
            oracle=self.oracle,
            word_list=self.word_list,
            max_conversation_length=self.max_conversation_length,
        )

class BatchedTwentyQuestionsEnv():
    def __init__(
        self, 
        env_load_path: str,
        cache_dir: str,
        device,
        max_conversation_length: int=20,
        bsize: int=32,
    ):
        self.env_list = [TwentyQuestionsEnv(max_conversation_length) for _ in range(bsize)]
        self.bsize = bsize
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
        self.model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small", cache_dir=cache_dir).to(device)
        self.model.load_state_dict(torch.load(env_load_path)['model_state_dict'])
        # self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        # self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-base").to(device)
        # self.model.load_state_dict(torch.load('/home/yifei/llm_rl/20q_oracle/20q_bart_oracle.pt')['model_state_dict'])

    def generate_answers(self, questions):
        curr_words = [env.curr_word[0].lower() for env in self.env_list]
        inputs = [f"The object is {curr_word}." + question for  curr_word, question in zip(curr_words, questions)]
        encoder_ids = self.tokenizer(inputs ,padding=True, return_tensors='pt').to(self.model.device)
        return self.tokenizer.batch_decode(self.model.generate(input_ids=encoder_ids['input_ids'], attention_mask=encoder_ids['attention_mask'],\
                                                                max_new_tokens=16, do_sample = False), skip_special_tokens= True)

    def reset(self, idx: Optional[int] = None):
        return [env.reset(idx) for env in self.env_list]
    
    def step(self, questions):
        answers = self.generate_answers(questions)
        # print("Step once!")
        with concurrent.futures.ThreadPoolExecutor() as executor: 
            jobs = [executor.submit(env._step, q, a) for env, q, a in zip(self.env_list, questions, answers)]
            results = [job.result() for job in jobs]
        return results

# class BatchedTwentyQuestionsEnv():
#     def __init__(
#         self, 
#         max_conversation_length: int=20,
#         bsize: int=32,
#     ):
#         self.env_list = [TwentyQuestionsEnv(max_conversation_length) for _ in range(bsize)]
#         self.bsize = bsize
    
#     def reset(self, idx: Optional[int] = None):
#         return [env.reset(idx) for env in self.env_list]
    
#     def step(self, questions):
#         with concurrent.futures.ThreadPoolExecutor() as executor: 
#             jobs = [executor.submit(env.step, q) for env, q in zip(self.env_list, questions)]
#             results = [job.result() for job in jobs]
#         return results
