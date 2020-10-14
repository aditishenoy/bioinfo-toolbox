import tensorflow as tf
from transformers import BertModel, BertTokenizer
import re
import os
import requests
from tqdm.auto import tqdm
import numpy as np

import argparse
import sys
import pandas as pd
import h5py

import torch

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

""" Reference : https://github.com/agemagician/ProtTrans """

modelUrl = 'https://www.dropbox.com/s/luv2r115bumo90z/pytorch_model.bin?dl=1'
configUrl = 'https://www.dropbox.com/s/33en5mbl4wf27om/bert_config.json?dl=1'
vocabUrl = 'https://www.dropbox.com/s/tffddoqfubkfcsw/vocab.txt?dl=1'

modelFolderPath = '/home/ashenoy/workspace/packages/language_models/ProtBert-BFD/'

modelFilePath = os.path.join(modelFolderPath, 'pytorch_model.bin')
configFilePath = os.path.join(modelFolderPath, 'config.json')
vocabFilePath = os.path.join(modelFolderPath, 'vocab.txt')


parser = argparse.ArgumentParser(description="Main script to run models")
parser.add_argument("--type", type=str, default = None, help="PP/PR")
parser.add_argument("--fastafilepath", type=str, help="training/testing")
parser.add_argument("--mappings", type=str, help="training/testing")
parser.add_argument("--output", type=str, help="training/testing")

args = parser.parse_args()


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

if not os.path.exists(vocabFilePath):
    download_file(vocabUrl, vocabFilePath)

<<<<<<< HEAD
tokenizer = BertTokenizer(vocabFilePath, do_lower_case=False, gradient_checkpointing=True)
self.encoder_features = 1024
=======
tokenizer = BertTokenizer(vocabFilePath, do_lower_case=False )
>>>>>>> 46fca9b4cd460b946d19f7725d3a85da84a2653a

model = BertModel.from_pretrained(modelFolderPath)
device = torch.device('cpu')
model = model.eval()

fasta_dirr = args.fastafilepath + '/'
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

#print (len(sequence_list))
#print (len(target_list))
npuniq = np.unique(target_list)
#print (npuniq)

ids = tokenizer.batch_encode_plus(sequence_list, add_special_tokens=True, pad_to_max_length=True)
input_ids = torch.tensor(ids['input_ids']).to(device)
attention_mask = torch.tensor(ids['attention_mask']).to(device)

# Embedding has shape (N, 3, 1024) where N = number of proteins
with torch.no_grad():
    embedding = model(input_ids=input_ids,attention_mask=attention_mask)[0]
    embedding = embedding.cpu().numpy()
    print (embedding.shape)

attention_mask = np.asarray(attention_mask)

target_list = np.char.encode(np.array(target_list), encoding='utf8')
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
# Per-protein embedding (N, 1024) 
if args.type == 'PPro':
    protein_embed = torch.tensor(embedding).sum(dim=1).mean(dim=1)
    protein_embed = protein_embed.cpu().numpy()
    print (protein_embed.shape)

    with h5py.File(outputfile, "w") as embeddings_file:
        embeddings_file.create_dataset("labels", data=target_list)
        embeddings_file.create_dataset('features', data=protein_embed)

"""