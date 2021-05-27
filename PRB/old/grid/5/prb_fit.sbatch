#!/bin/bash
#SBATCH --mail-user=jaulicino@uchicago.edu
#SBATCH --job-name=prb_initial
#SBATCH --output=prb_train.out
#SBATCH --partition=fela
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4000

source /home/jaulicino/anaconda2/bin/activate lss
python prb_weights.py
