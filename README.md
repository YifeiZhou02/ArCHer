# ArCHer
Research Code for "ArCHer: Training Language Model Agents via Hierarchical Multi-Turn RL" (Yifei Zhou, Andrea Zanette, Jiayi Pan, Aviral Kumar, Sergey Levine)

## Quick Start
### Install Dependencies
```bash
conda create -n archer python==3.10
conda activate archer

git clone https://github.com/YifeiZhou02/ArCHer
cd ArCHer
python -m pip install -e .
python3 -m spacy download en_core_web_sm
```
### Download Datasets and Checkpoints
Offline datasets and SFT checkpoints used in the paper can be found [here](https://drive.google.com/drive/folders/1pRocQI0Jv479G4vNMtQn1JOq8Shf2B6U?usp=sharing).
### Modify Paths
Change the ```huggingface_token``` and ```wandb_token``` in ```scripts/config/default.yaml``` .

**Guess My CITY**, **Twenty Questions**, **Detective Game** are directly usable by changing ```env_load_path```, ```checkpoint_path```, ```save_path``` in corresponding configurations in ```scripts/config``` such as ```scripts/config/archer_20q.yaml```.


