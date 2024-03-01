# ArCHer
Research Code for ["ArCHer: Training Language Model Agents via Hierarchical Multi-Turn RL"](https://arxiv.org/abs/2402.19446)

[website](https://yifeizhou02.github.io/archer.io/)

[Yifei Zhou](https://yifeizhou02.github.io/), [Andrea Zanette](https://azanette.com/), [Jiayi Pan](https://www.jiayipan.me/), [Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/), [Aviral Kumar](https://aviralkumar2907.github.io/)


![archer_diagram 001](https://github.com/YifeiZhou02/ArCHer/assets/83000332/b874432a-d330-49a5-906c-bba37e17f831)


This repo supports the following online methods and offline ArCHer implementation can be found [in this repo](https://github.com/andreazanette/OfflineArcher):

- [ArCHer][1]
- [Online CHAI][2]
- Online Filtered BC

[1]: https://arxiv.org/abs/2402.19446
[2]: https://arxiv.org/abs/2204.08426


And the following environments
- [Detective Game][3]
- [Twenty Questions][4]
- [Guess My City][4]
- [Webshop][5]

[3]: https://arxiv.org/abs/1909.05398
[4]: https://lmrl-gym.github.io/
[5]: https://webshop-pnlp.github.io/


## Quick Start
### 1. Install Dependencies
```bash
conda create -n archer python==3.10
conda activate archer

git clone https://github.com/YifeiZhou02/ArCHer
cd ArCHer
python -m pip install -e .
python3 -m spacy download en_core_web_sm
```
### 2. Download Datasets and Checkpoints
Offline datasets and SFT checkpoints used in the paper can be found [here](https://drive.google.com/drive/folders/1pRocQI0Jv479G4vNMtQn1JOq8Shf2B6U?usp=sharing).
### 3. Modify Paths
Change the ```huggingface_token``` and ```wandb_token``` in ```scripts/config/default.yaml``` .

**Guess My City**, **Twenty Questions**, **Detective Game** are directly usable by changing ```env_load_path``` (data to use for each environment), ```checkpoint_path``` (the SFT checkpoint to start with as provided), ```save_path``` (required, the path to save checkpoint and replay buffer) in corresponding configurations in ```scripts/config``` such as ```scripts/config/archer_20q.yaml```. For **Webshop**, additional installation is required in addition to modifying paths in the corresponding configuration.

### 4. Run Experiments
You can directly run experiments with the following commands:
```bash
cd scripts
python run.py --config-name archer_20q
```
Different environments and method can be run with corresponding configurations.

## Webshop Env Installation (Optional)
To use the webshop env, you need to do the following setups in addition. This step can be skipped if you do not plan to use Webshop.

Go to [WebShop's Github](https://github.com/princeton-nlp/WebShop) and follow the instructions to install the Webshop env

```
git clone https://github.com/princeton-nlp/webshop.git webshop
cd webshop
./setup.sh -d all
```

It turns out the provided installation guide is already outdates, so we need to do the following modifications:
```
pip install Werkzeug==2.2.2 pip install pydantic==1.10.11 pip install pip install --force-reinstall typing-extensions==4.5.0 beautifulsoup4
conda install mkl=2021
python -m spacy download en_core_web_lg
```

By default the WebShop only loads 1,000 products for a faster environment preview. To load all products, change web_agent_site/utils.py:
```
# DEFAULT_ATTR_PATH = join(BASE_DIR, '../data/items_ins_v2_1000.json')
# DEFAULT_FILE_PATH = join(BASE_DIR, '../data/items_shuffle_1000.json')
DEFAULT_ATTR_PATH = join(BASE_DIR, '../data/items_ins_v2.json')
DEFAULT_FILE_PATH = join(BASE_DIR, '../data/items_shuffle.json')
```

Then start the server at `128.0.0.1:3000`
```
python -m web_agent_site.app --log --attrs
```
### Run WebShop Experiments
An additional steps is required for running experiments on Webshop.
```bash
python -m PATH_TO_WEBSHOP/web_agent_site.app --log --attrs &
cd scripts
python run.py --config-name archer_webshop
```

## Distributed Data Parallel with [Accelerate](https://huggingface.co/docs/accelerate/en/index)
Experiments on single GPU can be slow (e.g. ArCHer on Twenty Questions can take a week), so this codebase supports Distributed Data Parallel.

First, you will need to set up the config for accelerate by changing the accelerate config file ```scripts/config/accelerate_config/default_config.yaml``` . Then change to run command to:
```bash
cd scripts
accelerate launch --config_file accelerate_config/default_config.yaml run.py --config-name archer_20q
```
## Specification for the Configuration
```cache_dir```: The cache dir for huggingface transformers (for saving pre-trained model weights etc).

```huggingface_token```: (Optional) Huggingface token for logging in (access some private models such as llama2).

```wandb_key```: This repo uses Weight and Biases for logging, put your wandb key here.

```policy_lm```: The model name (from huggingface) for the policy language model. The main results in the paper use ```gpt2```.

```critic_lm```: The model name (from huggingface) for the critic language model. The main results in the paper use ```roberta-base```.

```agent_type```: The algorithm to use, currently supports ```archer```, ```chai```, and ```online_filteredbc```.

```use_baseline```: Whether or not to train a separate model as token-level baseline. Will be added soon.

```use_lora```: Whether or not to use lora for policy language model. 

```max_new_tokens```: Maximum number of tokens to generate at each turn from the policy language model.

```save_freq```: Number of iterations to save all models and optimizers weights.

```eval_freq```: Number of times to do deterministic evaluations.

```capacity```: Number of utterance-level interaction tuples (s,a,r,s') that can be saved in the replay buffer.

```rollout_size```: Number of trajectories to collect for each iteration.

```eval_size```: Number of trajectories to evaluate on for each evaluation.

```batch_size```: Training batch size (same both for the actor and critic).

```iterations```: Number of total iterations.

```epochs```: The number of critic gradient steps for each iteration.

```actor_epochs```: The number of actor gradient steps for each iteration.

```warmup_iter```: Number of warming up iterations only updating the critic (i.e. the actor is not updated).

```grad_accum_steps```: Number of gradient accumulation steps. Note that gradients are not normalized with respect to gradient accumulation steps, so the effective learning rate is ```learning_rate*grad_accum_steps```.

```do_sample```: Whether or not sampling is used for rolling out trajectories to collect data.

```temperature```: The temperature when sampling from the policy language model.

```critic_lr```: The (unnormalized) learning rate for the critic, please also see ```grad_accum_steps```.

```lm_lr```: The (unnormalized) learning rate for the actor, please also see ```grad_accum_steps```.

```gamma```: The discount factor.

```tau```: Polyak constant for soft updating the target network.

```max_grad_norm```: Maximum gradient norm clipping threshold after gradient accumulation.

```use_wandb```: Whether or not to use Weights and Biases.

```checkpoint_path```: The path to the SFT checkpoint to start with, as provided above.

```save_path```: (Required) The path to save replay buffer and training checkpoints.

```env_name```: Which environment to use, currently supporting: ```twenty_questions```, ```guess_my_city```, ```adventure``` (can be used for Detective Game), and ```webshop```.

```env_load_path```: The path where the data for environment comes from (different for each environment, see example configs  for each environment).

```project_name```: Weights and Biases project name.

## Support for LLM
 Our default configuration runs with [GPT2](https://huggingface.co/openai-community/gpt2), but it also supports running state-of-the-art LLMs such as [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2). If you have a machine with larger RAM, simply try:
```bash
cd scripts
python run.py --config-name archer_llm_20q
```

## Citing ArCHer
```
@misc{zhou2024archer,
      title={ArCHer: Training Language Model Agents via Hierarchical Multi-Turn RL}, 
      author={Yifei Zhou and Andrea Zanette and Jiayi Pan and Sergey Levine and Aviral Kumar},
      year={2024},
      eprint={2402.19446},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
