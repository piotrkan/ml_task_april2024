"""data related functionalities, including torch dataset, ebmeddings and one-hot"""
import pandas as pd
import numpy as np
from transformers import BertModel, BertTokenizer
import re
import torch
from torch.utils.data import Dataset


def one_hot(unique_aa:set, protein_sequences:list):
    '''function for one-hot encoding protein sequences based on the number of unique aa
    NOTE: this function will work for training & modelling the data but it wouldnt be good
    for examining explainability (as we dont have features labelled )  -will need modifying
    
    Args:
        unique_aa - list of unique amino acid symbols present across all protein sequences
        protein-sequences - list or np.array of all protein sequences to be one-hot-encoded
    Out:
        np.array with n x l X a dimensions (where n is no. samples, l is length of sequences, 
        a is no. unique aa)
    '''
    #create mapping dict
    aa_dict = {aa: idx for idx, aa in enumerate(unique_aa)}
    num_aa = len(unique_aa)

    #assert same length
    seq_lengths=[len(seq) for seq in protein_sequences]
    assert min(seq_lengths)==max(seq_lengths)

    #create zero array of n x l x a dimension
    encoded_sequences = np.zeros((len(protein_sequences),
                                  seq_lengths[0],
                                  num_aa))
    #fill up the array at the appropriate positions, based on mappin
    for seq_id, seq in enumerate(protein_sequences):
        for aa_id, aa in enumerate(seq):
            if aa in aa_dict:
                encoded_sequences[seq_id, aa_id, aa_dict[aa]] = 1

    return encoded_sequences

def embed_sequence(sequence:str, model:BertModel, tokenizer:BertTokenizer) -> np.array:
    '''function for embedding protein sequences using protBert from huggingface.
        it first tokenizes, then encodes the sequences. Code modified from huggingface
    Args:
        sequence - string of sequence to be tokenized, 
        model - BertModel to be downloaded from huggingface
        tokenizer - tokenizer to be downloaded from huggingface
    Out:
        embedded array, with t x e dimensions (where t is no. tokens used, 
        e is dimensions of created embeddings
    '''
    #get model & tokenizer
    if model is None:
        print('downloading model')
        model = BertModel.from_pretrained("Rostlab/prot_bert")
    if tokenizer is None:
        print('downloading tokenizer')
        tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert")
    # the BERT tokenizer will not understand - as a symbol, need to change it to X if present
    sequence = re.sub("-", "X", sequence)
    encoded_input = tokenizer(sequence, return_tensors='pt')
    with torch.no_grad():
        embedding= model(**encoded_input)
    return embedding.last_hidden_state.squeeze()

def embed_seq_wrapper(series:pd.Series, model:BertModel, tokenizer:BertTokenizer) -> np.array:
    embedded_sequences=[]
    print(len(series))
    for i in range(len(series)):
        embedded_sequences.append(embed_sequence(series[i], model, tokenizer))
    return np.stack(embedded_sequences, axis=0)

class SequenceDataset(Dataset):
    """dataset class for pytorch neural nets"""
    def __init__(self, x, y, sequence_symbols=None):
        self.input=x
        self.label=y
        self.symbols=sequence_symbols

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        input_encoded = torch.tensor(self.input[idx])
        label = torch.tensor(self.label[idx])
        return input_encoded.float(), label.float()