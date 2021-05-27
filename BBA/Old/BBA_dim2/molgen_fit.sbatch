#!/bin/bash
#SBATCH --mail-user=jaulicino@uchicago.edu
#SBATCH --job-name=bba_train
#SBATCH --output=bba_train.out
#SBATCH --partition=fela
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8

source /home/jaulicino/anaconda2/bin/activate lss
python molgen_weights_generator.py
