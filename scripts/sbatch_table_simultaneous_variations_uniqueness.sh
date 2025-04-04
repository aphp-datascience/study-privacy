#!/bin/bash
#SBATCH --job-name=table_simultaneous_variations_uniqueness
#SBATCH -t 24:00:00
#SBATCH -N1-1
#SBATCH	--cpus-per-task=16
#SBATCH --mem=80000
#SBATCH --container-image /scratch/images/sparkhadoop.sqsh  --container-mounts=/export/home/acohen/privacy:/export/home/acohen/privacy --container-mount-home --container-writable   --container-workdir=/
#SBATCH --wait
#SBATCH --output=./logs/output-%j.out

# source $HOME/.user_conda/miniconda/etc/profile.d/conda.sh

## your code here
source $HOME/privacy/.venv/bin/activate 

 
python $HOME/privacy/scripts/table_simultaneous_variations_uniqueness.py --config $HOME/privacy/configs/config_base.cfg
