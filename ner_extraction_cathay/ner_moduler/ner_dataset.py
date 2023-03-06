import pandas as pd 
import numpy as np 
import torch 
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizerFast

class NER_Dataset(Dataset):
    def __init__(self, input_data ,bert_token_path, device):
        self.idx, self.sentence = input_data
        self.tokenizer = BertTokenizerFast.from_pretrained(bert_token_path, do_lower_case=False)
        self.device = device

        assert len(self.idx) == len(self.sentence)
        self.sentence_idx = self.convert_to_idx(self.sentence)

    def convert_to_idx(self,sentence):
        sentence_idx = []
        for s in sentence:      
            sentence_idx.append([101] + self.tokenizer.convert_tokens_to_ids(list(s.lower())) + [102])        
        assert len(sentence) == len(sentence_idx)
        return sentence_idx
    
    def __len__(self):
        return len(self.sentence)

    def __getitem__(self,index):
        return self.idx[index], self.sentence_idx[index],self.sentence[index]

    def collate_fn(self,batch):
        batch_idx = [row[0] for row in batch]
        batch_sentence_idx = [torch.tensor(row[1]) for row in batch]
        batch_sentence = [row[2] for row in batch]

        batch_sentence_idx = pad_sequence(batch_sentence_idx,batch_first = True)
        #batch_sentence_idx = batch_sentence_idx.to(self.device)

        return batch_idx,batch_sentence_idx,batch_sentence



if __name__ == "__main__":
    pass