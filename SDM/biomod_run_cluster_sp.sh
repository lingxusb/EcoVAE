#!/bin/bash
#
#SBATCH -n 8                 # Number of cores
#SBATCH -N 1                 # Number of nodes for the cores
#SBATCH -t 0-04:10           # Runtime in D-HH:MM format
#SBATCH -p shared    # Partition to submit to
#SBATCH --mem 60G            # Memory pool for all CPUs

module load python
conda activate r_biomod_env2
export R_LIBS_USER=$HOME/apps/R_4.4.3:$R_LIBS_USER


sp="$1"
group="$2"

R CMD BATCH --quiet --no-restore --no-save "--args $sp $group" biomod_single_sp_by_grid.R output_${sp}_${group}.out