#!/bin/bash
#SBATCH --mail-user=jaulicino@uchicago.edu
#SBATCH --job-name=villin_4dim
#SBATCH --output=villin_train.out
#SBATCH --partition=fela
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8

source /home/jaulicino/anaconda2/bin/activate lss
python villin_weights.py
