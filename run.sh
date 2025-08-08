#!/bin/bash
#SBATCH --job-name=bfa
#SBATCH --partition=gpu
#SBATCH --mail-user=zafaryab.haider@maine.edu  
#SBATCH --mem=200gb   
#SBATCH --output=log/bfa_%j.log   
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL      



module load apptainer
echo "Running Script"


apptainer exec --nv comp1.sif run.py
