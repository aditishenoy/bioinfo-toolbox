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
from torchnlp.encoders import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from torchnlp.utils import collate_tensors

import torch

from itertools import zip_longest

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord


MAX_LEN = 1500
BATCH_SIZE = 64

""" Reference : https://github.com/agemagician/ProtTrans """

def pool_strategy(features, pool_cls=True, pool_max=True, pool_mean=True,
                      pool_mean_sqrt=True):
        token_embeddings = features['token_embeddings']
        cls_token = features['cls_token_embeddings']
        attention_mask = features['attention_mask']
        
        # https://github.com/UKPLab/sentence-transformers/blob/eb39d0199508149b9d32c1677ee9953a84757ae4/sentence_transformers/models/Pooling.py
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

def get_pddf(f):
    
    df = pd.read_csv(f, sep = ",", names=["Localization", "Seq"])
    print (df)
    sample = []
    for i in df['Seq']:
        sample.append(sep_seq(i))

    df['sep_seq'] = sample

    seqlist= df['sep_seq'].tolist()
    seqlist = [re.sub(r"[UZOB]", "X", sequence) for sequence in seqlist]

    tarlist=df['Localization'].tolist()
    #tarlist = np.char.encode(tarlist)

    df2=pd.DataFrame()
    df2['seq'] = seqlist
    df2['label'] = tarlist

    return df2
        

def embed_sentence(modelFolderPath, vocabFilePath, seq, MAX_LEN):

    device = 'cpu'

    model = BertModel.from_pretrained(modelFolderPath)
    model = model.to(device)
    model = model.eval()

    tokenizer = BertTokenizer(vocabFilePath, do_lower_case=False)
    ids = tokenizer.batch_encode_plus(seq, add_special_tokens=True, padding=True, truncation=True, max_length=MAX_LEN)

    tokenized_sequences = torch.tensor(ids["input_ids"]).to(model.device)
    attention_mask = torch.tensor(ids["attention_mask"]).to(model.device)

    with torch.no_grad():
        embeddings = model(input_ids=tokenized_sequences, attention_mask=attention_mask)[0]


    print (embeddings.shape)
    embeddings = embeddings.clone().detach()
    protein_embd = torch.tensor(embeddings).sum(dim=0).mean(dim=0)
    print (protein_embd.shape)
    '''
    pooling = pool_strategy({"token_embeddings": embeddings,
                                      "cls_token_embeddings": embeddings[:, 0],
                                      "attention_mask": attention_mask,
                                      })
    
    pooling = pooling.cpu().numpy()
    print (pooling.shape)

    embeddings = embeddings.cpu().numpy()
    print (embeddings.shape)

    features = [] 
    for seq_num in range(len(embeddings)):
        seq_len = (attention_mask[seq_num] == 1).sum()
        seq_emd = embeddings[seq_num][1:seq_len-1]
        features.append(seq_emd)
    #print (len(features))'''

    return protein_embd

def main():

    modelUrl = 'https://www.dropbox.com/s/dm3m1o0tsv9terq/pytorch_model.bin?dl=1'
    configUrl = 'https://www.dropbox.com/s/d3yw7v4tvi5f4sk/bert_config.json?dl=1'
    vocabUrl = 'https://www.dropbox.com/s/jvrleji50ql5m5i/vocab.txt?dl=1'
    
    modelFolderPath = '/home/ashenoy/workspace/packages/language_models/ProtBert/'
    
    modelFilePath = os.path.join(modelFolderPath, 'pytorch_model.bin')
    configFilePath = os.path.join(modelFolderPath, 'config.json')
    vocabFilePath = os.path.join(modelFolderPath, 'vocab.txt')
    
    parser = argparse.ArgumentParser(description="Main script to run models")
    parser.add_argument("--input", type=str, help="csv file e.g. subpred_hr_onepercluster_per_protein_train.csv")
    parser.add_argument("--output", type=str, help="folder where embedding file should be saved")
    parser.add_argument("--comment", type=str, help="")
    args = parser.parse_args()
    
    if not os.path.exists(modelFilePath):
        download_file(modelUrl, modelFilePath)
        
    if not os.path.exists(configFilePath):
        download_file(configUrl, configFilePath)
        
    if not os.path.exists(vocabFilePath):
        download_file(vocabUrl, vocabFilePath)
    
    inputfile = args.input
    outputfile = args.output
    
    df = get_pddf(inputfile)
    print (df)

    target=[]

    with h5py.File(args.output+'BERT_XY_per-protein{}.h5'.format(args.comment), "w") as hf_out:

        grp=hf_out.create_group('subpred_X')

        for s, tar in zip(tqdm(df['seq']), df['label']):
            pooled_vector = embed_sentence(modelFolderPath, vocabFilePath, s, MAX_LEN)
            #print (pooled_vector.shape)
            #print (tar)
            target.append(tar)

        print (len(target))

        Y = np.stack(target, axis=0)
        print (Y.shape)


    hf_out.create_dataset('subpred_Y', data=np.array(Y, dtype='S'))
    hf.close()


if __name__ == '__main__':
    main()
