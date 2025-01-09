#! /bin/bash

#SBATCH --job-name=ct2rep_baseline
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=1g
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --error=/data/houbb/.logs/biowulf/ct2rep_baseline.err
#SBATCH --output=/data/houbb/.logs/biowulf/ct2rep_baseline.out
#SBATCH --time=10-00:00:00

source /data/houbb/_venv/python39/bin/activate
TOKENIZERS_PARALLELISM=false python main.py
