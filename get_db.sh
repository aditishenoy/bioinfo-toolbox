#!/bin/bash -l
#SBATCH -A SNIC2020-5-300
#SBATCH -t 20:00:00

#wget ftp://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz
wget http://wwwuser.gwdg.de/~compbiol/uniclust/2020_03/UniRef30_2020_03_hhsuite.tar.gz
#gunzip -c /home/a/aditi/pfs/databases/uniref90.fasta.gz > /home/a/aditi/pfs/databases/uniref90.fasta
