from allennlp.commands.elmo import ElmoEmbedder
from pathlib import Path
import torch
import argparse
import os
import sys
import numpy as np
import pandas as pd

def read_fasta_sequence(afile, query_id=''):
    seq_dict = {}
    header = ''
    seq = ''

    for aline in afile:
        aline = aline.strip()

        if aline.startswith('>'):
            if header != '' and seq != '':
                if header in seq_dict:
                    seq_dict[header].append(seq)
                else:
                    seq_dict[header] = [seq]
            seq = ''
            if aline.startswith('>%s' % query_id) and query_id !='':
                header = query_id
            else:
                header = aline[1:]
        else:
            #aline_seq = aline.translate(None, '.-').upper()
            seq += aline
    return seq

def main():

    parser = argparse.ArgumentParser(description="Script to generate per-protein and per-residue seqvec embeddings")
    parser.add_argument("--fastafilepath", type=str, help="training/testing")
    parser.add_argument("--output", type=str, help="training/testing")

    args = parser.parse_args()

    model_dir = Path('/home/ashenoy/workspace/packages/language_models/uniref50_v2')
    weights = model_dir / 'weights.hdf5'
    options = model_dir / 'options.json'
    embedder = ElmoEmbedder(options,weights, cuda_device=-1)
    fasta_dirr = args.fastafilepath
    
    sequence_list = []

    '''
    for filename in os.listdir(fasta_dirr):
        file_path = fasta_dirr + filename
        afile = open(file_path, 'r')
        sequence = read_fasta_sequence(afile)
        seqs = list(sequence)
        sequence_list.append(sequence)

    sequence_list.sort(key=len) # sorting is crucial for speed
    embedding = embedder.embed_sentences(sequence_list)'''

    afile = open(args.fastafilepath, 'r')
    print (afile)
    sequence = read_fasta_sequence(afile)
    print (sequence)
    embedding = embedder.embed_sentence(list(sequence))
    print (embedding)

    protein_embd = torch.tensor(embedding).sum(dim=0).mean(dim=0)
    with open('Q8WZ42.npy', 'wb') as f:
            np.save(f, protein_embd.numpy())

    '''for k, val in enumerate(embedding):
        residue_embd = torch.tensor(val).sum(dim=0) # Tensor with shape [L,1024]
        protein_embd = torch.tensor(val).sum(dim=0).mean(dim=0) # Vector with shape [1024]
        print (residue_embd.shape)
        print (protein_embd.shape)
        with open(args.output+'{}.npy'.format(k), 'wb') as f:
            np.save(f, protein_embd.numpy())'''

if __name__ == '__main__':
    main()

