import logging
import warnings
import torch 
import unicodedata
from torch.utils.data import DataLoader

from ner_moduler import Sentence_Segmentation, NER_Dataset, NER_Predict

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

def extracting_company_ner(input_data,pred_ner_type, input_colname, output_name, ner_model_path, batch_size, device):
    group_name,text_name = input_colname
    assert group_name in input_data.columns
    assert text_name in input_data.columns

    TS_func = Sentence_Segmentation(maxlen=500,split_char = ['-*-','。','，'])
    sentence_table = TS_func.transform(input_data,group_name = group_name, text_name = text_name)
    sentence_table[text_name + '_sentences'] = sentence_table[text_name + '_sentences'].str.strip()
    news_id, news_texts = sentence_table[group_name].tolist(), sentence_table[text_name + '_sentences'].tolist()
    input_dataset = NER_Dataset((news_id,news_texts), ner_model_path, device)

    TestDataLoader = DataLoader(input_dataset,
                                batch_size = batch_size,
                                shuffle = False,
                                num_workers = 0,
                                pin_memory = True,
                                collate_fn = input_dataset.collate_fn)
    
    ner_result = NER_Predict(TestDataLoader,pred_ner_type, input_colname, output_name, ner_model_path ,device).predict()

    return ner_result

def full2half(c: str) -> str:
        return unicodedata.normalize("NFKC", c)

import pandas as pd 
news = pd.read_excel('./input_data.xlsx')

# data prerprocessing for news 
assert ('id' in news.columns) and ('title' in news.columns) and ('content' in news.columns)
input_data = news[['id','title','content']].copy()
input_data['text'] = (input_data['title'] + '-*-' + input_data['content']).str.lower()
input_data = input_data[['id','text']]

device = 'cuda'
if device != 'cpu':
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

input_data['text'] = input_data['text'].apply(lambda x : full2half(x).lower()) 
input_data['text'] = input_data['text'].str.split('-\*-')
input_data = input_data.explode('text')

del_chars =  ['《','》','⭐','▲','&nbsp','●','/ ','”']
for del_char in del_chars:
    input_data['text'] = input_data['text'].str.replace(del_char,'')
texts,groups = input_data['text'].tolist(), input_data['id'].tolist()

batch_size = 1 
albert_path = "C:\\Users\\cloudy822\\Desktop\\ner_extraction\\model_filtes\\albert-tiny-chinese-ner" 
bert_path = "C:\\Users\\cloudy822\\Desktop\\ner_extraction\\model_filtes\\bert-base-chinese-ner"
# ner_model_path = "C:\\Users\\cloudy822\\Desktop\\ner_extraction\\model_filtes\\bert-base-chinese-ner"
# GPE ORG
pred_ner_type = ['ORG','GPE']
albert_ner_result = extracting_company_ner(input_data, 
                                           input_colname = ('id', 'text'),
                                           pred_ner_type = pred_ner_type,
                                           output_name = 'albert',
                                           ner_model_path = albert_path,
                                           batch_size = batch_size,
                                           device = device)
bert_ner_result = extracting_company_ner(input_data, 
                                         input_colname = ('id', 'text'),
                                         pred_ner_type = pred_ner_type,
                                         output_name = 'bert',
                                         ner_model_path = bert_path,
                                         batch_size = batch_size,
                                         device = device)

ner_fuzzyname_table = albert_ner_result.merge(bert_ner_result.drop(columns = 'text'), on = 'id', how = 'left')
for ner in pred_ner_type:
    columns = [c for c in ner_fuzzyname_table.columns if c.startswith(ner)]
    ner_fuzzyname_table[ner] = ner_fuzzyname_table[columns].apply(lambda x: list(set(x.sum())), axis = 1)
ner_fuzzyname_table = ner_fuzzyname_table[['id','text'] + pred_ner_type]




