#!/bin/bash
#SBATCH --job-name=gao_remesh
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=120:00:00
#SBATCH --mem=64G
#SBATCH --partition=tue.default.q

module purge
module load Umbrella/2024
module load Python/3.10.13-GCCcore-11.3.0
module load imkl/2025.1.0
module load libGLU/9.0.2-GCCcore-11.3.0
source ~/myenv/bin/activate

cd /scratch-shared/20236379/gao_remeshed

python -c "import nutils; print(nutils.__version__)"

export NUTILS_MATRIX=mkl

export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

export NUTILS_NPROCS=1


srun python -u gao_ext_remesh.py main
