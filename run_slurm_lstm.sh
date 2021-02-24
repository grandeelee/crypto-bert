#!/usr/bin/env bash

#SBATCH --job-name=deci_war
#SBATCH --output=/home/grandee/grandee/projects/crypto-bert/war_and_peace_expt_lstm/slurm.log
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=8000MB
#SBATCH -n 4

/home/grandee/anaconda3/envs/transformer37/bin/python war_and_peace_lstm.py 10 2 512 10
#/home/grandee/anaconda3/envs/transformer37/bin/python war_and_peace_exp.py 40 2 500 10
#/home/grandee/anaconda3/envs/transformer37/bin/python war_and_peace_exp.py 50 2 400 10
#/home/grandee/anaconda3/envs/transformer37/bin/python war_and_peace_exp.py 70 12 300 10
#/home/grandee/anaconda3/envs/transformer37/bin/python war_and_peace_exp.py 100 12 200 10
#/home/grandee/anaconda3/envs/transformer37/bin/python war_and_peace_exp.py 150 12 100 10