# ArCHer
Research Code for "ArCHer: Training Language Model Agents via Hierarchical Multi-Turn RL" (Yifei Zhou, Andrea Zanette, Jiayi Pan, Aviral Kumar, Sergey Levine)
This repo supports the following methods:

- [ArCHer][1]
- [Online CHAI][2]
- Online Filtered BC

[1]: https://github.com/YifeiZhou02/ArCHer
[2]: https://arxiv.org/abs/2204.08426


And the following environments
- [Detective Game][1]
- [Twenty Questions][2]
- [Guess My City][2]
- [Webshop][3]

[1]: https://arxiv.org/abs/1909.05398
[2]: https://lmrl-gym.github.io/
[3]: https://webshop-pnlp.github.io/


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

**Guess My CITY**, **Twenty Questions**, **Detective Game** are directly usable by changing ```env_load_path``` (data to use for each environment), ```checkpoint_path``` (the SFT checkpoint to start with as provided), ```save_path``` (required, the path to save checkpoint and replay buffer) in corresponding configurations in ```scripts/config``` such as ```scripts/config/archer_20q.yaml```. For **Webshop**, additional installation is required in addition to modifying paths in the corresponding configuration.

### 4. Run Experiments
You can directly run experiments with the following commands:
```bash
cd scripts
python run.py --config-name archer_20q
```

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
