# -*- coding: <encoding name>-*-
'''
Created on Feb 26, 2014

@author: c3h3
'''

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import jieba, nltk



articles_df = None
one_board = None
one_uid = None

# TODO: segmentalizer
tokenize_gen = lambda token: lambda text: list(token(text)) if isinstance(text,(str,unicode)) else [] 

def tokenize(text):
    if isinstance(text,(str,unicode)):
        return list(jieba.cut(nltk.clean_html(text)))
    else:
        return []




class SparseDocumentTermMatrix(dict):
    def __init__(self, word_summery, sdtm):
        self["word_summery"] = word_summery
        self["sdtm"] = sdtm
        
    



class Text2Vec(dict):
    def __init__(self, vec):
        self["vec"] = vec
        
    def fit(self, df_col):
        self["sdtm"] = self["vec"].fit_transform(df_col)
        
    def sdtm_sum(self, axis=0):
        assert len(self["sdtm"].shape) >= axis
        return self["sdtm"].sum(axis=axis)
    
    def sdtm_mean(self, axis=0):
        assert len(self["sdtm"].shape) >= axis
        return self["sdtm"].sum(axis=axis)/float(self["sdtm"].shape[axis])
    
    def summerize_words(self):
        all_words = np.array(self["vec"].get_feature_names())
            
        dt = np.dtype([('score_sum', int),('score_sum_L0_norm', int),('word_len', int),('word_ix', int), ('word', "<U%s" % max(map(len,all_words))) ])
        
        self["word_summery"] = np.array(zip(self["sdtm"].sum(axis=0).tolist()[0],
                                            self["sdtm"].sign().sum(axis=0).tolist()[0],
                                            map(len,all_words),
                                            np.arange(len(all_words)),
                                            all_words),dtype=dt)
        
    
    
    
    
    
    def _get_top_k_words(self, top_k=None):
        assert isinstance(top_k, int)
        if top_k == 0:
            return []
            
        else: 
                
            #all_words = np.array(self["vec"].get_feature_names())
            #
            #dt = np.dtype([('count', int), ('kws', "<U%s" % max(map(len,all_words))) ])
            #
            #word_counts = np.array(zip(self["sdtm"].sum(axis=0).tolist()[0],all_words),dtype=dt)
            
            if not("word_count" in self):
                self.summerize_word_count()
                    
            if top_k > 0:
                return np.sort(self["word_summery"],order="score")[-top_k:]["word"]
                
            else:
                return np.sort(self["word_summery"],order="score")["word"]
    
    def _get_top_k_words_index(self, top_k=None):
        assert isinstance(top_k, int)
        if top_k == 0:
            return []
            
        else: 
            if not("word_count" in self):
                self.summerize_words()
                    
            if top_k > 0:
                return np.argsort(self["word_summery"],order="score")[-top_k:]
                
            else:
                return np.argsort(self["word_summery"],order="score")
                
    
    
    def get_top_words(self, top_k=None, top_p=None):
        assert isinstance(top_k, int) | isinstance(top_p, (int,float))
        
        if isinstance(top_p, (int,float)):
            
            _top_k = int(float(top_p)*self["sdtm"].shape[0])
            
            print 'self["sdtm"].shape[0] = ',self["sdtm"].shape[0]
            print "top_p = ",top_p
            print "_top_k = ",_top_k
            
            return self._get_top_k_words(_top_k)
            
        
        elif isinstance(top_k, int):
            return self._get_top_k_words(top_k)
        
    
    def get_top_words_index(self, top_k=None, top_p=None):
        assert isinstance(top_k, int) | isinstance(top_p, (int,float))
        
        if isinstance(top_p, (int,float)):
            
            _top_k = int(float(top_p)*self["sdtm"].shape[0])
        
            return self._get_top_k_words_index(_top_k)
            
        
        elif isinstance(top_k, int):
            return self._get_top_k_words_index(top_k)
            
            
            
                      
    
        
        



# def vectorize_text(df=articles_df, colname="title", query={"Board":one_board, "user_id":one_uid}, 
#                    vect_gen=CountVectorizer, vect_gen_init_kwargs = {"tokenizer":tokenize,"lowercase":False}):    
    
def vectorize_text(df, colname, query={},
                   vect_gen=CountVectorizer, 
                   vect_gen_init_kwargs = {"tokenizer":tokenize,"lowercase":False}):    
    
    assert colname in df.columns
    
    for c in query:
        assert c in df.columns
    
    
    text2vec = Text2Vec(vect_gen(**vect_gen_init_kwargs))
    
    if len(query.keys()):
        
        query_conds = lambda :(df[i] == query[i] for i in query)
    
        qcs = query_conds()
    
        q_final = qcs.next()
    
        for q in qcs:
            q_final = q_final & q
    
        q_df = df[q_final]
        #print q_df.head()
        
    else:
        q_df = df
    
    text2vec.fit(q_df[colname])
    
    return text2vec


if __name__ == '__main__':
    pass

#    jieba.initialize()
#    print map(repr,tokenize(u"柯文哲")
#    jieba.add_word(u"柯文哲",3.0)
#    print map(repr,tokenize(u"柯文哲")