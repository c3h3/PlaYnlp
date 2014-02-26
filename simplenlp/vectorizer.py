'''
Created on Feb 26, 2014

@author: c3h3
'''

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

articles_df = None
one_board = None
one_uid = None


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
    
    def summerize_word_count(self):
        all_words = np.array(self["vec"].get_feature_names())
            
        dt = np.dtype([('count', int), ('kws', "<U%s" % max(map(len,all_words))) ])
        
        self["word_count"] = np.array(zip(self["sdtm"].sum(axis=0).tolist()[0],all_words),dtype=dt)
        
    
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
                return np.sort(self["word_count"],order="count")[-top_k:]["kws"]
                
            else:
                return np.sort(self["word_count"],order="count")["kws"]
                
        
    
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
        
            
            
            
                      
        
        
        



def vectorize_text(df=articles_df, colname="title", query={},#{"Board":one_board, "user_id":one_uid}, 
                   vect_gen=CountVectorizer, vect_gen_init_kwargs = {} #{"tokenizer":tokenize,"lowercase":False}
                   ):    
    
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