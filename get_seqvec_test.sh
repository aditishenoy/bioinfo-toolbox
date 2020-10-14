#!/bin/bash -l
#SBATCH -A SNIC2020-5-300
#SBATCH -c 8
#SBATCH -t 25:00:00
#SBATCH --error=/home/a/aditi/pfs/current_projects/SubPred/data/EXP-5.1-20-05-06-embeddings/seqvec_test0724.error
#SBATCH --output=/home/a/aditi/pfs/current_projects/SubPred/data/EXP-5.1-20-05-06-embeddings/seqvec_test0724.out

for filename in *.fasta;
    do
	if [ -f "${filename%.*}".npz ]; then
		echo "$filename"
	else
		echo "${filename%.*}"
		seqvec -i $filename -o "${filename%.*}".npz
	fi
    done
