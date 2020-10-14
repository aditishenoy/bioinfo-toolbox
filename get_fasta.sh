#!/bin/bash -l
#SBATCH -A SNIC2020-5-300
#SBATCH -t 02:00:00
#SBATCH -c 28

while read id; do wget -nc  http://www.uniprot.org/uniprot/$id.fasta; done < /home/a/aditi/pfs/current_projects/NuDeep/data/EXP-2-SEQVEC/training_fastafiles/train_names

~
