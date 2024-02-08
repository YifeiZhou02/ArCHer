import random
import logging
logging.getLogger().setLevel(logging.CRITICAL)
import concurrent.futures

from typing import List, Optional
import requests
from bs4 import BeautifulSoup, Comment
# WEBSHOP_URL = "http://127.0.0.1:3000"
ACTION_TO_TEMPLATE = {
    'Description': 'description_page.html',
    'Features': 'features_page.html',
    'Reviews': 'review_page.html',
    'Attributes': 'attributes_page.html',
}

def clean_str(p):
  return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")

def tag_visible(element):
    ignore = {'style', 'script', 'head', 'title', 'meta', '[document]'}
    return (
        element.parent.name not in ignore and not isinstance(element, Comment)
    )


def webshop_text(session, page_type, query_string='', page_num=1, asin='', options={}, subpage='', webshop_url=None, **kwargs):
    assert webshop_url is not None
    WEBSHOP_URL = webshop_url
    if page_type == 'init':
      url = (
          f'{WEBSHOP_URL}/{session}'
      )
    if page_type == 'search':
      url = (
          f'{WEBSHOP_URL}/search_results/{session}/'
          f'{query_string}/{page_num}'
      )
    elif page_type == 'item':
      url = (
          f'{WEBSHOP_URL}/item_page/{session}/'
          f'{asin}/{query_string}/{page_num}/{options}'
      )
    elif page_type == 'item_sub':
      url = (
          f'{WEBSHOP_URL}/item_sub_page/{session}/'
          f'{asin}/{query_string}/{page_num}/{subpage}/{options}'
      )
    elif page_type == 'end':
      url = (
          f'{WEBSHOP_URL}/done/{session}/'
          f'{asin}/{options}'
      )
    # print(url)
    html = requests.get(url).text
    html_obj = BeautifulSoup(html, 'html.parser')
    texts = html_obj.findAll(text=True)
    visible_texts = list(filter(tag_visible, texts))
    # visible_texts = [str(text).strip().strip('\\n') for text in visible_texts]
    # if page_type == 'end': import pdb; pdb.set_trace()
    if False:
        # For `simple` mode, return just [SEP] separators
        return ' [SEP] '.join(t.strip() for t in visible_texts if t != '\n')
    else:
        # Otherwise, return an observation with tags mapped to specific, unique separators
        observation = ''
        option_type = ''
        options = {}
        asins = []
        cnt = 0
        prod_cnt = 0
        just_prod = 0
        for t in visible_texts:
            if t == '\n': continue
            if t.replace('\n', '').replace('\\n', '').replace(' ', '') == '': continue
            # if t.startswith('Instruction:') and page_type != 'init': continue
            # print(t.parent.name, t)
            if t.parent.name == 'button':  # button
                processed_t = f'\n[{t}] '
            elif t.parent.name == 'label':  # options
                if f"'{t}'" in url:
                    processed_t = f'[[{t}]]'
                    # observation = f'You have clicked {t}.\n' + observation
                else:
                    processed_t = f'[{t}]'
                options[str(t)] = option_type
                # options[option_type] = options.get(option_type, []) + [str(t)]
            elif t.parent.get('class') == ["product-link"]: # product asins
                processed_t = f'\n[{t}] '
                if prod_cnt >= 3:
                  processed_t = ''
                prod_cnt += 1
                asins.append(str(t))
                just_prod = 0
            else: # regular, unclickable text
                processed_t =  '\n' + str(t) + ' '
                if cnt < 2 and page_type != 'init': processed_t = ''
                if just_prod <= 2 and prod_cnt >= 4: processed_t = ''
                option_type = str(t)
                cnt += 1
            just_prod += 1
            observation += processed_t
        info = {}
        if options:
          info['option_types'] = options
        if asins:
          info['asins'] = asins
        if 'Your score (min 0.0, max 1.0)' in visible_texts:
          idx = visible_texts.index('Your score (min 0.0, max 1.0)')
          info['reward'] = float(visible_texts[idx + 1])
          observation = 'Your score (min 0.0, max 1.0): ' + (visible_texts[idx + 1])
        return clean_str(observation), info

class WebShopEnv:
  def __init__(self, session_id=None, url=None):
    assert url is not None
    self.url = url
    if session_id is None:
      session_id = random.randint(0, 10000)
    self.reset(session_id)
  
  def step(self, action):
    session = self.session_id

    done = False
    observation_ = None
    if action == 'reset':
      self.sessions[session] = {'session': session, 'page_type': 'init'}
    elif action.startswith('think['):
      observation_ = 'OK.'
    elif action.startswith('search['):
      assert self.sessions[session]['page_type'] == 'init'
      query = action[7:-1]
      self.sessions[session] = {'session': session, 'page_type': 'search',
                                'query_string': query, 'page_num': 1}
    elif action.startswith('click['):
      button = action[6:-1]
      if button == 'Buy Now':
        assert self.sessions[session]['page_type'] == 'item'
        self.sessions[session]['page_type'] = 'end'
        done = True
      elif button == 'Back to Search':
        assert self.sessions[session]['page_type'] in ['search', 'item_sub', 'item']
        self.sessions[session] = {'session': session, 'page_type': 'init'}
      elif button == 'Next >':
        assert False # ad hoc page limitation
        assert self.sessions[session]['page_type'] == 'search'
        self.sessions[session]['page_num'] += 1
      elif button == '< Prev':
        assert self.sessions[session]['page_type'] in ['search', 'item_sub', 'item']
        if self.sessions[session]['page_type'] == 'search':
          assert False
          self.sessions[session]['page_num'] -= 1
        elif self.sessions[session]['page_type'] == 'item_sub':
          self.sessions[session]['page_type'] = 'item'
        elif self.sessions[session]['page_type'] == 'item':
          self.sessions[session]['page_type'] = 'search'
          self.sessions[session]['options'] = {}
      elif button in ACTION_TO_TEMPLATE:
        assert self.sessions[session]['page_type'] == 'item'
        self.sessions[session]['page_type'] = 'item_sub'
        self.sessions[session]['subpage'] = button
      else:
        if self.sessions[session]['page_type'] == 'search':
          assert button in self.sessions[session].get('asins', [])  # must be asins
          self.sessions[session]['page_type'] = 'item'
          self.sessions[session]['asin'] = button
        elif self.sessions[session]['page_type'] == 'item':
          assert 'option_types' in self.sessions[session]
          assert button in self.sessions[session]['option_types'], (button, self.sessions[session]['option_types'])  # must be options
          option_type = self.sessions[session]['option_types'][button]
          if not 'options' in self.sessions[session]:
            self.sessions[session]['options'] = {}
          self.sessions[session]['options'][option_type] = button
          observation_ = f'You have clicked {button}.'
    else:
      assert False
    observation, info = webshop_text(**self.sessions[session], webshop_url=self.url)
    if observation_:
      observation = observation_
    self.sessions[session].update(info)
    reward = info.get('reward', 0.0)
    return observation, reward, done
  
  def reset(self, session):
     self.sessions = {}
     self.session_id = session

def get_instruction(init_obs: str):
  start = "Instruction:  \n"
  instruction_start = init_obs.find(start) + len(start)
  instruction_end = init_obs.rfind("\n[Search]")
  instruction = init_obs[instruction_start:instruction_end]
  return instruction


class WebShopFullObsEnv:
  def __init__(self, session_id=None, url=None):
    assert url is not None
    self.env = WebShopEnv(session_id, url=url)
    
  def step(self, action):
    if self.has_finished:
       return None
    if action[-1] == "\n":
      action = action[:-1]
    if self.current_step >= 10:
       print("Time out! Max Step of 10 reached")
       return self.full_obs, 0.0, True
    if action == "reset":
      raise NotImplementedError("Reset should be called on the environment, not the agent.")
    try:
      observation, reward, done = self.env.step(action)
    except AssertionError:
      observation = "Invalid action!"
      reward = 0.0
      done = False
    self.full_obs += "\nAction:\n" + action
    self.full_obs += "\nObservation:\n" + observation
    self.current_step += 1
    if done:
      self.has_finished = True
    return self._get_obs(), reward, done
  
  def _get_obs(self):
    """We """
    return self.full_obs + "\nInstruction:\n" + self.instruction

  def reset(self, session):
    self.env.reset(session)
    self.current_step = 0
    observation, reward, done = self.env.step("reset")
    self.instruction = get_instruction(observation)
    self.full_obs = ""
    self.full_obs += "\nObservation:\n" + observation
    self.has_finished = False
    return self._get_obs()

class BatchedWebShopEnv():
    def __init__(
        self, 
        env_load_path: str,
        lower: int,
        upper: int,
        bsize: int=32,
    ):

        # self.urls = [
        #    f"http://127.0.0.1:{i}" for i in range(3000, 3002)
        # ]
        self.urls = [
           env_load_path,
          #  f"http://127.0.0.1:3001"
        ]
        self.bsize = bsize
        assert self.bsize % len(self.urls) == 0, "Batch size must be a multiple of the number of urls"
        self.url_per_dp = [self.urls[i % len(self.urls)] for i in range(self.bsize)]
        self.concurrent_group = []
        self.lower = lower
        self.upper = upper
        group_size = len(self.urls)
        for i in range(0, self.bsize, group_size):
            self.concurrent_group.append(list(range(i, i + group_size)))

    def reset(self, idx: Optional[List[int]] = None):
        self.env_list = [WebShopFullObsEnv(url=url) for url in self.url_per_dp]
        if idx is None:
            idx = random.choices(range(self.lower,self.upper), k=self.bsize)
        session_ids = [f"fixed_{i}" for i in idx]
        return [env.reset(id) for env, id in zip(self.env_list, session_ids)]
        # results = []
        # for group in self.concurrent_group:
        #     with concurrent.futures.ThreadPoolExecutor() as executor: 
        #         jobs = [executor.submit(env.reset, sessin_id) for env, session_id in zip([self.env_list[i] for i in group], [session_ids[i] for i in group])]
        #         results += [job.result() for job in jobs]
        # return results
    
    def step(self, actions: List[str]):
        results = []
        for group in self.concurrent_group:
            with concurrent.futures.ThreadPoolExecutor() as executor: 
                jobs = [executor.submit(env.step, action) for env, action in zip([self.env_list[i] for i in group], [actions[i] for i in group])]
                results += [job.result() for job in jobs]
        # with concurrent.futures.ThreadPoolExecutor() as executor: 
        #     jobs = [executor.submit(env.step, action) for env, action in zip(self.env_list, actions)]
        #     results = [job.result() for job in jobs]
        return results
        # return [env.step(action) for env, action in zip(self.env_list, actions)]