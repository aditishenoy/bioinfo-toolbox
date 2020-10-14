#!/bin/bash -l
#SBATCH -A SNIC2020-5-300
#SBATCH -p largemem
#SBATCH -t 40:00:00
#SBATCH --error=/home/a/aditi/pfs/phd_BIT/generate_embeddings/bert_test.error
#SBATCH --output=/home/a/aditi/pfs/phd_BIT/generate_embeddings/bert_test.out

python generate_ProtBert.py --fastafilepath ~/pfs/current_projects/SubPred/data/EXP-5-20-05-03/FastaFiles25404/ --mappings /home/a/aditi/pfs/current_projects/SubPred/data/EXP-5.1-20-05-06-embeddings/testing_mappings_noCAN --output ~/pfs/phd_BIT/generate_embeddings/BERT_testing.h5

