#!/usr/bin/env bash

#SBATCH --job-name=deci_war
#SBATCH --output=/home/grandee/grandee/projects/crypto-bert/war_and_peace_expt_lstm/slurm.log
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=8000MB
#SBATCH -n 4

/home/grandee/anaconda3/envs/transformer37/bin/python war_and_peace_lstm.py 10 3 512 10
/home/grandee/anaconda3/envs/transformer37/bin/python war_and_peace_lstm.py 20 3 512 10
/home/grandee/anaconda3/envs/transformer37/bin/python war_and_peace_lstm.py 30 3 512 10
/home/grandee/anaconda3/envs/transformer37/bin/python war_and_peace_lstm.py 50 3 512 10
/home/grandee/anaconda3/envs/transformer37/bin/python war_and_peace_lstm.py 70 3 512 10
/home/grandee/anaconda3/envs/transformer37/bin/python war_and_peace_lstm.py 100 3 512 10