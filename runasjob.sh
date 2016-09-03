#!/bin/bash
#SBATCH -p standard
#SBATCH -J tanveer_sisc
#SBATCH -o /scratch/mtanveer/out/tanveer_sisc_%j_%a
#SBATCH --mem-per-cpu=4gb
#SBATCH -t 4-23:00:00
#SBATCH -c 4
#SBATCH -a 0-719

M_array=(8 16 32 64 80 128 150 256)
D_array=(5 10 20 30 40 50 70 100 150 200)
Beta_array=(25 50 100 150 200 250 300 500 1000)

let "z = $SLURM_ARRAY_TASK_ID % 8"; 
let "y = $SLURM_ARRAY_TASK_ID / 8 % 10"; 
let "x = $SLURM_ARRAY_TASK_ID / 80 % 9"; 

module load anaconda
echo M ${M_array[$z]} D ${D_array[$y]} Beta ${Beta_array[$x]}
python sisc_wrapper.py -o /scratch/mtanveer/Results/result -iter_thresh 3000 -M ${M_array[$z]} -D ${D_array[$y]} -Beta ${Beta_array[$x]} -i ../allSkeletons/*.csv

