#!/bin/bash
#SBATCH --job-name=bfa
#SBATCH --partition=gpu
# SBATCH --mail-user=zafaryab.haider@maine.edu  
#SBATCH --mem=200gb   
#SBATCH --output=log/yelp5_%j.log   
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL      



module load apptainer

# Only install packages if INSTALL_DEPS is set to 1
if [ "${INSTALL_DEPS:-0}" = "1" ]; then
    echo "Installing dependencies..."
    apptainer exec --nv comp1.sif python -m pip install --upgrade pip > /dev/null 2>&1
    apptainer exec --nv comp1.sif python -m pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118 > /dev/null 2>&1
    apptainer exec --nv comp1.sif python -m pip install transformers==4.35.2 accelerate==0.24.1 datasets==2.14.5 sentencepiece==0.1.99 > /dev/null 2>&1
    apptainer exec --nv comp1.sif python -m pip install "numpy<2.0" > /dev/null 2>&1
    echo "Dependencies installed."
fi

export APPTAINER_SILENT=1
export SINGULARITY_SILENT=1

echo "Starting Python script..."
# Redirect apptainer's stderr to /dev/null while keeping Python's output
{
    apptainer exec --nv comp1.sif bash -c "
    export TRANSFORMERS_VERBOSITY=error
    export DATASETS_VERBOSITY=error  
    export TOKENIZERS_PARALLELISM=false
    python -u run.py --dataset=yelp_review_full --split=test --model_name=rttl-ai/bert-base-uncased-yelp-reviews --max_examples=512
    "
} 2> >(grep -v "gocryptfs\|not a valid test operator" >&2)
