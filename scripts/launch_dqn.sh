#! /bin/bash
# This is a job script that uses array jobs and GNU parallel to launch a hyperparameter
# sweep. We use Parallel to launch M array tasks, with each job running N processes in parallel.
# By using some tricks, we can evenly distribution our overall hyperparameter configurations
# to these M x N total parallel processes.

# Job configurations. Note in the last line that we are launching an array job of 4 tasks,
# all tasks will execute this script. We u#se the environment variable SLURM_ARRAY_TASK_ID
# to determine which array task it is.

#SBATCH --job-name=dqn
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -o /global/scratch/users/yifeizhou/20q/logs/%j.log  # Name of stdout output file (%j expands to jobId)
#SBATCH -e /global/scratch/users/yifeizhou/20q/logs/%j.err  # Name of stderr output file
#SBATCH --time=14:00:00
#SBATCH --account=co_rail
#SBATCH --qos=rail_gpu4_normal
#SBATCH --partition=savio4_gpu
#SBATCH --mem=128G
#SBATCH --gres=gpu:A5000:4

# Exit the script if it is not launched from slurm.
if [ -z "$SLURM_JOB_ID" ]; then
    echo "This script is not launched with slurm, exiting!"
    exit 1
fi
export TOKENIZER_PARALLELISM=false
export NCCL_P2P_DISABLE=1

source /global/scratch/users/yifeizhou/miniconda3/bin/activate
conda activate llm_rl
cd /global/home/users/yifeizhou/llm_rl/ArCHer/scripts
# bash /global/home/users/yifeizhou/llm_rl/LLM_rep_RL/scripts/serve_webshop.sh
# accelerate launch --main_process_port 12727 run_dqn.py --config-name dqn_webshop
# # accelerate launch --main_process_port 12327 run_filteredbc.py --config-name filteredbc_webshop
accelerate launch --main_process_port 21427 run.py --config-name archer_20q
# accelerate launch --config_file  /global/home/users/yifeizhou/.cache/huggingface/accelerate/llama_config.yaml run_dqn.py --config-name dqn_20q_llama1b
# python run_dqn.py --config-name chai_20q


