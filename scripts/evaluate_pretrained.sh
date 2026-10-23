#! /bin/bash
#PBS -N evaluate_pretrained
#PBS -o logs/evaluate_pretrained.log
#PBS -e logs/evaluate_pretrained_err.log
#PBS -l ncpus=2
#PBS -q cpu

module load compiler/anaconda3
cd /home/kuber.shahi_asp22/project
source aml/bin/activate
rm -f logs/evaluate_pretrained.log
rm -f logs/evaluate_pretrained_err.log
python scripts/evaluate_pretrained.py
