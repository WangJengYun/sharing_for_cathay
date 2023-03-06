import logging
import warnings
import torch 
import unicodedata
from torch.utils.data import DataLoader

from ner_moduler import Sentence_Segmentation, NER_Dataset, NER_Predict

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

def extracting_company_ner(input_data, input_colname, output_name, ner_model_path, batch_size, device):
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
                                num_workers = 3,
                                pin_memory = True,
                                collate_fn = input_dataset.collate_fn)

    ner_result = NER_Predict(TestDataLoader, input_colname, output_name, ner_model_path ,device).predict()

    return ner_result


def ner_fuzzyname_prediction(news, fuzzyname_table, device, batch_size, albert_path, bert_path):
    
    def full2half(c: str) -> str:
        return unicodedata.normalize("NFKC", c)
    
    # data prerprocessing for fuzzyanme data 
    fuzzyname_table['fuzzyname'] = fuzzyname_table['fuzzyname'].apply(lambda x : full2half(x).lower()) 
    fuzzyname_table['fuzzyname'] = fuzzyname_table['fuzzyname'].str.split('@')
    fuzzyname_table_explode = fuzzyname_table.explode('fuzzyname')
    fuzzyname_table_explode['fuzzyname'] = fuzzyname_table_explode['fuzzyname'].str.lower()
    fuzzyname_table_explode['fuzzyname'] = fuzzyname_table_explode['fuzzyname'].str.strip()
    fuzzyname_table_explode = fuzzyname_table_explode[fuzzyname_table_explode['fuzzyname'] != '']
    fuzzyname_table_explode['id_name'] = fuzzyname_table_explode['customer_id'] + '_' + fuzzyname_table_explode['companyname']
    
    # data prerprocessing for news 
    assert ('id' in news.columns) and ('title' in news.columns) and ('content' in news.columns)
    input_data = news[['id','title','content']].copy()
    input_data['text'] = (input_data['title'] + '-*-' + input_data['content']).str.lower()
    input_data = input_data[['id','text']]
    
    if device != 'cpu':
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    input_data['text'] = input_data['text'].apply(lambda x : full2half(x).lower()) 
    input_data['text'] = input_data['text'].str.split('-\*-')
    input_data = input_data.explode('text')
    
    del_chars =  ['《','》','⭐','▲','&nbsp','●','/ ','”']
    for del_char in del_chars:
        input_data['text'] = input_data['text'].str.replace(del_char,'')
    texts,groups = input_data['text'].tolist(), input_data['id'].tolist()

    #albert_path = "/workpool/ded/ba/00585602cub/company_fuzzy_matrix/NER_MODEL_優化/ckip-transformers-master/albert-base-chinese-ner"
    albert_ner_result = extracting_company_ner(input_data, 
                                               input_colname = ('id', 'text'),
                                               output_name = 'albert_ner',
                                               ner_model_path = albert_path,
                                               batch_size = batch_size,
                                               device = device)

    #bert_path = "/workpool/ded/ba/00585602cub/company_fuzzy_matrix/NER_MODEL_優化/ckip-transformers-master/bert_base_chinese_ner"
    bert_ner_result = extracting_company_ner(input_data, 
                                             input_colname = ('id', 'text'),
                                             output_name = 'bert_ner',
                                             ner_model_path = bert_path,
                                             batch_size = batch_size,
                                             device = device)

    ner_fuzzyname_table = albert_ner_result.merge(bert_ner_result[['id', 'bert_ner']], on = 'id', how = 'left')
    ner_fuzzyname_table['ner'] = (ner_fuzzyname_table['albert_ner'] + ner_fuzzyname_table['bert_ner']).apply(lambda x: list(set(x)))
    ner_fuzzyname_table = ner_fuzzyname_table[['id','ner']]
    
    fuzzyname_result = ner_fuzzyname_table.explode('ner').rename(columns = {'ner':'fuzzyname'})
    fuzzyname_result = fuzzyname_result.merge(fuzzyname_table_explode[['fuzzyname','id_name','customer_id','companyname']], how = 'inner', on = 'fuzzyname')    
    
    fuzzyname_result_for_fuzzyname = fuzzyname_result.groupby('id',as_index = False).agg({'fuzzyname':list, 'id_name':list})
    fuzzyname_result_for_fuzzyname['fuzzyname'] = fuzzyname_result_for_fuzzyname['fuzzyname'].apply(lambda x : list(set(x)))
    fuzzyname_result_for_fuzzyname['id_name'] = fuzzyname_result_for_fuzzyname['id_name'].apply(lambda x : list(set(x)))
    
    fuzzyname_result_for_company = fuzzyname_result.drop_duplicates(subset = ['id','customer_id']).copy()
    fuzzyname_result_for_company = fuzzyname_result_for_company.groupby('id',as_index = False).agg({'customer_id':list, 'companyname':list})

    fuzzyname_info = fuzzyname_result_for_fuzzyname.merge(fuzzyname_result_for_company, how = 'left', on = 'id')
    
    company_info = ner_fuzzyname_table.merge(fuzzyname_info, how = 'left', on = 'id')
    news = news.merge(company_info, how = 'left', on = 'id')

    return news

if __name__ == "__main__":
    pass