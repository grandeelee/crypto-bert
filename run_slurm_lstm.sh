#!/usr/bin/env bash

#SBATCH --job-name=deci_war
#SBATCH --output=/home/grandee/grandee/projects/crypto-bert/war_and_peace_expt_lstm/slurm_2.log
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=8000MB
#SBATCH -n 4

/home/grandee/anaconda3/envs/transformer37/bin/python war_and_peace_lstm.py 10 2 512 10
/home/grandee/anaconda3/envs/transformer37/bin/python war_and_peace_lstm.py 20 2 512 10
/home/grandee/anaconda3/envs/transformer37/bin/python war_and_peace_lstm.py 30 2 512 10
/home/grandee/anaconda3/envs/transformer37/bin/python war_and_peace_lstm.py 50 2 512 10
/home/grandee/anaconda3/envs/transformer37/bin/python war_and_peace_lstm.py 70 2 512 10
/home/grandee/anaconda3/envs/transformer37/bin/python war_and_peace_lstm.py 100 2 512 10