#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
import numpy as np
import pandas as pd
import h5py

import torch
from transformers import AlbertModel, AlbertTokenizer, pipeline
import re
import numpy as np
import os
import requests
from tqdm.auto import tqdm

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

""" Reference : https://github.com/agemagician/ProtTrans """

modelUrl = 'https://www.dropbox.com/s/gtajtmege43ec7k/pytorch_model.bin?dl=1'
configUrl = 'https://www.dropbox.com/s/me7zsqrnpiz043v/config.json?dl=1'
tokenizerUrl = 'https://www.dropbox.com/s/60mg00r361vth4t/albert_vocab_model.model?dl=1'

modelFolderPath = '/home/ashenoy/workspace/packages/language_models/ProtAlbert/'

modelFilePath = os.path.join(modelFolderPath, 'pytorch_model.bin')
configFilePath = os.path.join(modelFolderPath, 'config.json')
tokenizerFilePath = os.path.join(modelFolderPath, 'spm_model.model')

parser = argparse.ArgumentParser(description="Main script to run models")
parser.add_argument("--type", type=str, help="Per-residue/Per-protein")
parser.add_argument("--fastafilepath", type=str, help="training/testing")
parser.add_argument("--mappings", type=str, help="training/testing")
parser.add_argument("--output", type=str, help="training/testing")

args = parser.parse_args()

if not os.path.exists(modelFolderPath):
    os.makedirs(modelFolderPath)

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

def download_file(url, filename):
  response = requests.get(url, stream=True)
  with tqdm.wrapattr(open(filename, "wb"), "write", miniters=1,
                    total=int(response.headers.get('content-length', 0)),
                    desc=filename) as fout:
      for chunk in response.iter_content(chunk_size=4096):
          fout.write(chunk)

if not os.path.exists(modelFilePath):
    download_file(modelUrl, modelFilePath)

if not os.path.exists(configFilePath):
    download_file(configUrl, configFilePath)

if not os.path.exists(tokenizerFilePath):
    download_file(tokenizerUrl, tokenizerFilePath)

tokenizer = AlbertTokenizer(tokenizerFilePath, do_lower_case=False )

model = AlbertModel.from_pretrained(modelFolderPath)

device = torch.device('cpu')
model = model.eval()

fasta_dirr = args.fastafilepath
mappings = args.mappings 
outputfile = args.output

df = pd.read_csv(mappings, sep = ",", names=["Uniprot_ID", "Localization"])
print (df)

sequence_list = []
target_list = []
for filename in os.listdir(fasta_dirr):
    #print (filename[:-6])
    file_path = fasta_dirr + filename

    afile = open(file_path, 'r')
    sequence = read_fasta_sequence(afile)

    if (filename[:-6]) in df['Uniprot_ID'].values:
        tar = (df.loc[df['Uniprot_ID']==(filename[:-6])]['Localization'].values)
        sequence_list.append(sequence)
        target_list.append(str(tar))

print (len(sequence_list))
print (len(target_list))
npuniq = np.unique(target_list)
print (npuniq)

sequence_list = [re.sub(r"[UZOB]", "X", str(sequence)) for sequence in sequence_list]

ids = tokenizer.batch_encode_plus(sequence_list, add_special_tokens=True, pad_to_max_length=True)
input_ids = torch.tensor(ids['input_ids']).to(device)
attention_mask = torch.tensor(ids['attention_mask']).to(device)

# Per-protein embeddings (embedding  has shape (N, 3, 4096) where N = number of proteins)
with torch.no_grad():
    embedding = model(input_ids=input_ids,attention_mask=attention_mask)[0]
    embedding = embedding.cpu().numpy()
    print (type(embedding))
    print (embedding.shape)

target_list = np.char.encode(np.array(target_list), encoding='utf8')
print (type(target_list))
print (target_list.shape)


features = [] 
for seq_num in range(len(embedding)):
    seq_len = (attention_mask[seq_num] == 1).sum()
    seq_emd = embedding[ seq_num][1:seq_len-1]
    features.append(seq_emd)

X = np.stack(features, axis = 0)
print (X.shape)

with h5py.File(outputfile, "w") as embeddings_file:
    embeddings_file.create_dataset("labels", data=target_list)
    embeddings_file.create_dataset('features', data=X)

"""
# Per-protein embedding (N, 4096)
if args.type == 'PPro':
    protein_embed = torch.tensor(embedding).sum(dim=1)
    protein_embed = protein_embed.cpu().numpy()
    print (protein_embed.shape)

    with h5py.File(outputfile, "w") as embeddings_file:
        embeddings_file.create_dataset("labels", data=target_list)
        embeddings_file.create_dataset('features', data=protein_embed)

"""

