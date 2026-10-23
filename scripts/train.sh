#! /bin/bash
#PBS -N train
#PBS -o logs/train.log
#PBS -e logs/train_err.log
#PBS -l ncpus=3
#PBS -q gpu

module load compiler/anaconda3
cd /home/kuber.shahi_asp22/project
source aml/bin/activate
rm -f logs/train.log
rm -f logs/train_err.log
python scripts/train.py
