import pandas as pd 
import numpy as np 

class Sentence_Segmentation:
    def __init__(self,maxlen,split_char):
        self.maxlen = maxlen
        self.split_char = split_char

    def split_into_sentences(self,article,maxlen,split_char):    
        if split_char and len(article)>maxlen:
            sentences = article.split(split_char[0])
            sentences = [s for s in sentences if s]
            text,texts = '',[]
            for idx, s in enumerate(sentences):
                if len(text) + len(s) > maxlen-1:                 
                    if text:
                        texts.append(text)                
                        text = ''
                    if len(s) > maxlen-1:
                        sub_sentences = self.split_into_sentences(s,maxlen,split_char[1:])
                        texts.extend(sub_sentences)
                        s = ''
                if s :
                    if idx + 1 == len(sentences):
                        text = text + s
                        texts.append(text)
                    else:
                        text = text + s + split_char[0]
            return texts
        elif len(article)>maxlen:
            stop_idx = len(article)
            bin_list = list(range(0,stop_idx,maxlen - 2))+[stop_idx]

            texts = []
            for i in range(len(bin_list)-1):
                texts.append(article[bin_list[i]:bin_list[i+1]])

            return texts
        else:
            return [article]
    
    def transform(self,input_data,group_name = None,text_name = None,is_explode = True):
        if isinstance(input_data,pd.DataFrame):
            output = input_data
            output[text_name + '_sentences'] = output[text_name].apply(lambda x: self.split_into_sentences(x,self.maxlen,self.split_char))
            if is_explode:
                output = output[[group_name, text_name + '_sentences']].explode(text_name + '_sentences')
            return output
        
        elif isinstance(input_data,tuple) and (len(input_data) == 2):
            id_idx,articles =  input_data
            text_sentences = []
            
            for idx,article in enumerate(articles):
                text_sentences.append(self.split_into_sentences(article,self.maxlen,self.split_char))

            if is_explode:
                output = pd.DataFrame({group_name:id_idx, text_name:text_sentences})
                output = output.explode(text_name)
                id_idx,text_sentences = output[group_name].tolist(), output[text_name].tolist()

            return (id_idx,text_sentences)
        else:
            raise ValueError('Please the format of input_data')

if __name__ == "__main__":
    pass