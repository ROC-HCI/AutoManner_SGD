#!/bin/bash
#SBATCH -p standard
#SBATCH -J tanveer_sisc
#SBATCH -o /scratch/mtanveer/sisc_out/tanveer_sisc_%j_%a
#SBATCH --mem=8gb
#SBATCH -t 4-23:00:00
#SBATCH -c 4
#SBATCH -a 0-293

M_array=(8 16 32 64 128 256)
D_array=(5 10 30 50 100 200 500)
Beta_array=(10 20 50 100 200 500 1000)

let "z = $SLURM_ARRAY_TASK_ID % 6"; 
let "y = $SLURM_ARRAY_TASK_ID / 6 % 7"; 
let "x = $SLURM_ARRAY_TASK_ID / 42 % 7"; 

module load anaconda
echo M ${M_array[$z]} D ${D_array[$y]} Beta ${Beta_array[$x]}
# echo "File List:"
# for i in /scratch/mtanveer/allSkeletons/*.csv; do echo $i; done
python sisc_wrapper.py -o /scratch/mtanveer/sisc_results -iter_thresh 1500 -M ${M_array[$z]} -D ${D_array[$y]} -Beta ${Beta_array[$x]} -i /scratch/mtanveer/allSkeletons/*.csv

