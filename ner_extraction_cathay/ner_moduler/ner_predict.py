import re 
import pandas as pd 
from pandas.core.common import flatten
import numpy as np 
from tqdm import tqdm, trange
import torch 
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import (AutoModelForTokenClassification,BatchEncoding,BertTokenizer, BertConfig)

class NER_Predict(object):
    def __init__(self, dataloader,pred_ner_type, input_colname, output_name,ner_model_path, device):

        self.dataloader = dataloader
        self.group_name, self.text_name = input_colname
        self.pred_ner_type = pred_ner_type
        self.output_name = output_name
        
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(ner_model_path, do_lower_case=False)
        self.model =  AutoModelForTokenClassification.from_pretrained(ner_model_path).to(self.device)

        self.model.eval()

    def predict(self):

        news_idx,sentence_idx,predict_idx,sentence = self._predict_ner(self.dataloader)
        ner_entity = self._extract_ner(news_idx, sentence,sentence_idx,predict_idx)
        ner_entity.update({self.group_name:news_idx, self.text_name:sentence})
        ner_result = pd.DataFrame(ner_entity)
        ner_result = ner_result.groupby(self.group_name).sum().reset_index()
        ner_result = ner_result.rename(columns = { i: i  + '_' + self.output_name for i in self.pred_ner_type})
        return ner_result

    def _get_sent_token(self,text):
        output = []
        input_sent_token = self.tokenizer.basic_tokenizer.tokenize(text)
        for char in input_sent_token:
            if (re.compile('[A-Za-z0-9]+').match(char)) or (len(char) > 1):
                standard_char = self.tokenizer.tokenize(char)
                if standard_char[0] == '[UNK]':
                    output.append(char)
                else:
                    output.append(standard_char)
            else:
                output.append(char)
        output = list(flatten(output))
        return output

    def _predict_ner(self, dataloader):
        
        news_idx,sentence_idx,predict_idx,sentence = [],[],[],[]
        

        for idx,batch  in tqdm(enumerate(dataloader),total = len(dataloader)):
            
            batch_idx,batch_sentence_idx,batch_sentence = batch 
            batch_sentence_idx = batch_sentence_idx.to(self.device)
            
            with torch.no_grad():
                predict_result = self.model(batch_sentence_idx)
            predict_result = torch.argmax(predict_result[0], dim=2)

            news_idx.extend(batch_idx)
            predict_idx.extend([predict_result[i,:].to('cpu').tolist() for i in range(predict_result.shape[0])])
            sentence_idx.extend([batch_sentence_idx[i,:].to('cpu').tolist() for i in range(batch_sentence_idx.shape[0])])
            sentence.extend(batch_sentence)

        return news_idx,sentence_idx,predict_idx,sentence

    def _extract_ner(self, news_idx, sentence, sentence_idx, predict_idx):
        
        id2label = self.model.config.id2label

        ner_entity = {i:[] for i in self.pred_ner_type}

        ner_entity_idx = {i:[] for i in self.pred_ner_type}
        for j,(news_idx, input_sent, input_idx, output_idx) in  tqdm(enumerate(zip(news_idx, sentence,sentence_idx,predict_idx)),total = len(sentence)): 
            # input_sent_token = ['[CLS]'] + self._get_sent_token(input_sent)
            input_sent_token = [101] + list(input_sent.lower()) + [102]
            entity_word_list= {i:[] for i in self.pred_ner_type}
            entity_idx_list = {i:[] for i in self.pred_ner_type}
            entity_word = {i:'' for i in self.pred_ner_type}
            entity_idx = {i:[] for i in self.pred_ner_type}
            char = None
            # print(list(zip(output_idx,input_sent_token)))
            try:
                for i, char_idx in enumerate(output_idx):
                    
                    if char_idx == 0:
                        bioes = "O"
                    else :
                        label = id2label[char_idx]
                        bioes, ner = label.split("-")
                    
                    if bioes == 'O':
                        entity_word = {i:'' for i in self.pred_ner_type}
                        entity_idx = {i:[] for i in self.pred_ner_type}
                    elif ner in self.pred_ner_type:
    
                        if  bioes == "B":
                            char = self.tokenizer.convert_ids_to_tokens(int(input_idx[i]))
                            if char == '[UNK]':
                                char = input_sent_token[i]
                            entity_word[ner]= char    
                            entity_idx[ner].append(i)
                            
                        elif bioes == "I":
                            if entity_word[ner]:
                                char = self.tokenizer.convert_ids_to_tokens(int(input_idx[i]))
                                if char == '[UNK]':
                                    char = input_sent_token[i]
                                entity_word[ner] = entity_word[ner] + char
                                entity_idx[ner].append(i)
                    
                        elif  bioes == "E":                        
                            if entity_word[ner]:
                                char = self.tokenizer.convert_ids_to_tokens(int(input_idx[i]))
                                if char == '[UNK]':
                                    char = input_sent_token[i]
                                
                                entity_word[ner] = (entity_word[ner] + char).replace('#','').strip()
                                entity_idx[ner].append(i)
                                
                                if len(entity_word[ner]) > 1:
                                    entity_idx_list[ner].append(entity_idx[ner])
                                    entity_word_list[ner].append(entity_word[ner])
                                    entity_word[ner] = ''
                                    entity_idx[ner] = []
                                else:
                                    entity_word[ner] = ''
                                    entity_idx[ner] = []   
                    else:
                        entity_word[ner] = ''
                        entity_idx[ner] = []
   
                for ner in self.pred_ner_type:      
                    ner_entity[ner].append(entity_word_list[ner])
                    ner_entity_idx[ner].append(entity_idx_list[ner])
            
            except Exception as e:
                print(e)

        return ner_entity

if __name__ == "__main__":
    pass