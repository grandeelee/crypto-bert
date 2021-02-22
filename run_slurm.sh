#!/usr/bin/env bash

#SBATCH --job-name=deci_war
#SBATCH --output=/home/grandee/grandee/projects/crypto-bert/war_and_peace_expt/slurm.log
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=8000MB
#SBATCH -n 4

/home/grandee/anaconda3/envs/transformer37/bin/python war_and_peace_exp.py 30 12 512 10
/home/grandee/anaconda3/envs/transformer37/bin/python war_and_peace_exp.py 40 12 500 10
/home/grandee/anaconda3/envs/transformer37/bin/python war_and_peace_exp.py 50 12 400 10
/home/grandee/anaconda3/envs/transformer37/bin/python war_and_peace_exp.py 70 12 300 10
/home/grandee/anaconda3/envs/transformer37/bin/python war_and_peace_exp.py 100 12 200 10
/home/grandee/anaconda3/envs/transformer37/bin/python war_and_peace_exp.py 150 12 100 10