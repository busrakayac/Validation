#!/bin/bash
#SBATCH --job-name=optimized
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

cd /scratch-shared/20236379/Gao

python -c "import nutils; print(nutils.__version__)"

export NUTILS_MATRIX=mkl

export MKL_NUM_THREADS=16
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16

export NUTILS_NPROCS=16              # CHANGED: was 1 — tell Nutils to use 16 processes

echo "Running from: $(pwd)"
git rev-parse --short HEAD
readlink -f gao_ext_remesh.py

srun python -u gao_optimized.py main
