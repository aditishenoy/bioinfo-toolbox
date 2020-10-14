#import tensorflow as tf
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
#from torchnlp.encoders import LabelEncoder
from torch.utils.data import Dataset, DataLoader
#from torchnlp.utils import collate_tensors

import torch

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

#import seaborn as sns
#sns.set(style='whitegrid', palette='muted', font_scale=1.2)
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

""" Reference : https://github.com/agemagician/ProtTrans """

# https://github.com/UKPLab/sentence-transformers/blob/eb39d0199508149b9d32c1677ee9953a84757ae4/sentence_transformers/models/Pooling.py
def pool_strategy(features, pool_cls=True, pool_max=True, pool_mean=True,
                      pool_mean_sqrt=True):
        token_embeddings = features['token_embeddings']
        cls_token = features['cls_token_embeddings']
        attention_mask = features['attention_mask']

        ## Pooling strategy
        output_vectors = []
        if pool_cls:
            output_vectors.append(cls_token)
        if pool_max:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            max_over_time = torch.max(token_embeddings, 1)[0]
            output_vectors.append(max_over_time)
        if pool_mean or pool_mean_sqrt:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            #If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            if 'token_weights_sum' in features:
                sum_mask = features['token_weights_sum'].unsqueeze(-1).expand(sum_embeddings.size())
            else:
                sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)

            if pool_mean:
                output_vectors.append(sum_embeddings / sum_mask)
            if pool_mean_sqrt:
                output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))

        output_vector = torch.cat(output_vectors, 1)
        return output_vector

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
    spaced_seq = sep_seq(seq)
    return spaced_seq

def sep_seq(seq):
    new_seq = ""
    i = 1
    #print (seq)
    for s in seq:
        new_seq = new_seq + s 
        if i < len(seq):
            new_seq = new_seq + ' '
            i = i + 1
    #print (new_seq)
    return new_seq   

def download_file(url, filename):
  response = requests.get(url, stream=True)
  with tqdm.wrapattr(open(filename, "wb"), "write", miniters=1,
                    total=int(response.headers.get('content-length', 0)),
                    desc=filename) as fout:
      for chunk in response.iter_content(chunk_size=4096):
          fout.write(chunk)

def prepare_sample(sample: list, tokenizer, prepare_target: bool = True) -> (dict, dict):
    """
    Function that prepares a sample to input the model.        
    :param sample: list of dictionaries.
        
    Returns:
            - dictionary with the expected model inputs.
            - dictionary with the expected target labels.
    """
    sample = collate_tensors(sample)

    slist = []
    for seq in sample["seq"]:
        seqstr = list(seq)
        slist.append(seqstr)

    token_lens = []
    for s in slist:
        tokens = tokenizer.encode(s)
        print (tokens)
        token_lens.append(len(tokens))

    print (token_lens)
    sns.distplot(token_lens)
    plt.xlabel('Token count')
    plt.savefig('{}.png'.format('token_count'), bbox_inches='tight')

    ids = tokenizer.batch_encode_plus(slist, add_special_tokens=False,
                                    padding=True, truncation=True, max_length=2000)
    return ids

def main():

    modelUrl = 'https://www.dropbox.com/s/dm3m1o0tsv9terq/pytorch_model.bin?dl=1'
    configUrl = 'https://www.dropbox.com/s/d3yw7v4tvi5f4sk/bert_config.json?dl=1'
    vocabUrl = 'https://www.dropbox.com/s/jvrleji50ql5m5i/vocab.txt?dl=1'
    
    modelFolderPath = '/home/a/aditi/pfs/packages/language_models/ProtBert/'
    
    modelFilePath = os.path.join(modelFolderPath, 'pytorch_model.bin')
    configFilePath = os.path.join(modelFolderPath, 'config.json')
    vocabFilePath = os.path.join(modelFolderPath, 'vocab.txt')
    

    parser = argparse.ArgumentParser(description="Main script to run models")
    parser.add_argument("--fastafilepath", type=str, help="training/testing")
    parser.add_argument("--mappings", type=str, help="training/testing")
    parser.add_argument("--output", type=str, help="training/testing")
    
    args = parser.parse_args()
    
    if not os.path.exists(modelFilePath):
        download_file(modelUrl, modelFilePath)
        
    if not os.path.exists(configFilePath):
        download_file(configUrl, configFilePath)
        
    if not os.path.exists(vocabFilePath):
        download_file(vocabUrl, vocabFilePath)
    
    fasta_dirr = args.fastafilepath + '/'
    mappings = args.mappings 
    outputfile = args.output
    
    df = pd.read_csv(mappings, sep = ",", names=["Uniprot_ID", "Localization"])
    print (df)
    
    #sequence_list = []
    #target_list = []
    sample = []
    for filename in os.listdir(fasta_dirr):

        b_dict = {}
        file_path = fasta_dirr + filename

        afile = open(file_path, 'r')
        sequence = read_fasta_sequence(afile)
    
        if (filename[:-6]) in df['Uniprot_ID'].values:
            tar = (df.loc[df['Uniprot_ID']==(filename[:-6])]['Localization'].values)[0]
            b_dict['seq'] = sequence
            b_dict['label'] = tar
            blist = b_dict.copy()
            sample.append(blist)
            #sequence_list.append(sequence)
            #target_list.append(str(tar))

    df1 = pd.DataFrame(sample)
    print (df1)
    seqlist= df1['seq'].tolist()
    seqlist = [re.sub(r"[UZOB]", "X", sequence) for sequence in seqlist]
    #print (seqlist)
    tarlist=df1['label'].tolist()
    tarlist = np.char.encode(tarlist)

    npuniq = np.unique(tarlist)
    print (npuniq)
    tokenizer = BertTokenizer(vocabFilePath, do_lower_case=False)
    ids = tokenizer.batch_encode_plus(seqlist, add_special_tokens=False, padding=True, truncation=True, max_length=2000)
    print (type(ids))
    device = torch.device('cpu')
    print (device)
    model = BertModel.from_pretrained(modelFolderPath)
    model = model.to(device)
    model = model.eval()

    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    with torch.no_grad():
        embedding = model(input_ids=input_ids,attention_mask=attention_mask)[0]
        pooling = pool_strategy({"token_embeddings": embedding,
                                      "cls_token_embeddings": embedding[:, 0],
                                      "attention_mask": attention_mask,
                                      })

    pooling = pooling.cpu().numpy()
    print (pooling.shape)

    #embedding = embedding.cpu().numpy()
    #print (embedding.shape)


    with h5py.File(outputfile, "w") as embeddings_file:
        embeddings_file.create_dataset("labels", data=tarlist)
        embeddings_file.create_dataset('features', data=pooling)
    
    """
    features = [] 
    for seq_num in range(len(embedding)):
        seq_len = (attention_mask[seq_num] == 1).sum()
        seq_emd = embedding[seq_num][1:seq_len-1]
        features.append(seq_emd)
        print (seq_emd.shape)
    print (len(features))
    """

    """
    encoder_features = 1024
    model = BertModel.from_pretrained(modelFolderPath)
    
    label_set = "CYT,ERE,EXC,GLG,LYS,MEM,MIT,NUC,PEX,PLS"
    # Label Encoder
    label_encoder = LabelEncoder(label_set.split(","), reserved_labels=[])
    label_encoder.unknown_index = None
    print (label_encoder)
    
    device = torch.device('cpu')
    model = model.eval()

    ids = tokenizer.batch_encode_plus(sequence_list, add_special_tokens=True, pad_to_max_length=True)
   

    input_ids = torch.tensor(ds['input_ids']).to(device)
    print (input_ids)
    attention_mask = torch.tensor(ds['attention_mask']).to(device)



    # Embedding has shape (N, 3, 1024) where N = number of proteins
    with torch.no_grad():
        embedding = model(input_ids=input_ids,attention_mask=attention_mask)[0]
        pooling = pool_strategy({"token_embeddings": embedding,
                                      "cls_token_embeddings": embedding[:, 0],
                                      "attention_mask": attention_mask,
                                      })
        print (pooling.shape)
        pooling = pooling.cpu().numpy()
        embedding = embedding.cpu().numpy()
        #print (embedding.shape)

    attention_mask = np.asarray(attention_mask)

    target_list = np.char.encode(np.array(target_list), encoding='utf8')
    print (target_list.shape)
    """

    '''
    features = [] 
    for seq_num in range(len(embedding)):
        seq_len = (attention_mask[seq_num] == 1).sum()
        seq_emd = embedding[ seq_num][1:seq_len-1]
        features.append(seq_emd)

    X = np.stack(features, axis = 0)
    #print (X.shape)
    

    with h5py.File(outputfile, "w") as embeddings_file:
        embeddings_file.create_dataset("labels", data=target_list)
        embeddings_file.create_dataset('features', data=pooling)

    '''

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

if __name__ == '__main__':
    main()
