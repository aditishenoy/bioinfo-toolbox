#!/bin/bash -l
#SBATCH -A SNIC2019-3-319
#SBATCH -c 1
#SBATCH -t 20:00:00

wget ftp://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz
#gunzip -c /home/a/aditi/pfs/databases/uniref90.fasta.gz > /home/a/aditi/pfs/databases/uniref90.fasta
