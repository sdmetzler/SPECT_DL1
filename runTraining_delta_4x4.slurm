#!/bin/bash
#SBATCH --mem=16g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=4    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA100x4      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bcps-delta-gpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH --job-name=4x4
#SBATCH --time=10:00:00      # hh:mm:ss for the job
#SBATCH --constraint="scratch"
### GPU options ###
#SBATCH --gpus-per-node=2
#SBATCH --gpu-bind=none     # <- or closest
#SBATCH --mail-user=metzler@upenn.edu
#SBATCH --mail-type="BEGIN,END"


#module reset # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)
             # $WORK and $SCRATCH are now set
module load python  # ... or any appropriate modules
module load anaconda3_gpu/23.9.0
module list  # job documentation and metadata
echo "job is starting on `hostname`"
srun python3 -u /projects/bcps/smetzler/SPECT_DL1/SPECT_training_channels3.py -v -e 120 -n 500 -b 5000 -l 1.0E-3 -o Models/channelized_model_4x4_new.tar -r /projects/bcps/smetzler/SPECT_DL1_save/good_model_4x4_4.tar
#-r channelized_model_4x4.tar.3499354
##-r Models/channelized_model_4x4_best.tar


