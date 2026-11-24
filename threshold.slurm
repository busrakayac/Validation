#!/bin/bash
#SBATCH --job-name=gao_remesh
#SBATCH --output=output_%A_%a.txt
#SBATCH --error=error_%A_%a.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=120:00:00
#SBATCH --mem=64G
#SBATCH --partition=tue.default.q
#SBATCH --array=0-3%4

module purge
module load Umbrella/2024
module load Python/3.10.13-GCCcore-11.3.0
module load imkl/2025.1.0
module load libGLU/9.0.2-GCCcore-11.3.0
source ~/myenv/bin/activate

BASE=/scratch-shared/20236379/Gao

THRESHOLDS=(0.4 0.5 0.6 0.7)
THRESHOLD=${THRESHOLDS[$SLURM_ARRAY_TASK_ID]}

export REMESH_J_THRESHOLD=$THRESHOLD

# Nutils / MKL
export NUTILS_MATRIX=mkl
export NUTILS_NPROCS=16

export MKL_NUM_THREADS=16
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16

# Give every simulation its own directory.
# This is essential because fast.csv, mesh.jpg, remesh .geo/.msh files,
# and diagnostic figures otherwise overwrite each other.
RUN_DIR=${BASE}/runs/remesh_${THRESHOLD}_job${SLURM_ARRAY_JOB_ID}

mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

echo "======================================"
echo "Array job      : $SLURM_ARRAY_JOB_ID"
echo "Array task     : $SLURM_ARRAY_TASK_ID"
echo "Threshold      : $THRESHOLD"
echo "Working dir    : $RUN_DIR"
echo "CPUs           : $SLURM_CPUS_PER_TASK"
echo "======================================"

python -c "import nutils; print('Nutils:', nutils.__version__)"

cd "$BASE"
git rev-parse --short HEAD
readlink -f gao_neohookean.py

cd "$RUN_DIR"

srun python -u "$BASE/gao_neohookean.py" main
