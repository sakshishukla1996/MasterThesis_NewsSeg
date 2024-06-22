#!/bin/bash

#SBATCH --job-name=test_job
#SBATCH --output=/home/mturan/gitlab/SLURM-diarization/test_output_%j.out
#SBATCH --error=/home/mturan/gitlab/SLURM-diarization/test_error_%j.out
#SBATCH --ntasks=1
#SBATCH --time=00:05:00  # 5 minute

export CONDA_ENVS_PATH="/home/mturan/conda/miniconda/envs"
eval "$(conda shell.bash hook)"
conda activate SLURM-test
conda info --envs
python -V
echo ""

cd "/home/mturan/gitlab/SLURM-diarization"
start_time="$(date -u +%s)"

python main.py --input-file test/tagesschau02092019.wav --output-dir ./exp

end_time="$(date -u +%s)"
elapsed="$(($end_time-$start_time))"

echo "Total Elapsed Time -- `date -d@"$elapsed_EN" -u +%H:%M:%S`"
