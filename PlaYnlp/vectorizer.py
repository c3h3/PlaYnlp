# -*- coding: utf-8 -*-


import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from .sparse import SparseDataFrame

class SparseDocumentTermMatrixSummary(dict):
    
    def __init__(self, summary_data, terms_idx):
        self["summary_data"] = summary_data
        self["terms_idx"] = terms_idx
        
    def __lt__(self, upper_bound):
        return type(self)(summary_data = self["summary_data"] < upper_bound,
                          terms_idx = self["terms_idx"])
    
    def __le__(self, upper_bound):
        return type(self)(summary_data = self["summary_data"] <= upper_bound,
                          terms_idx = self["terms_idx"])
    
    def __gt__(self, lower_bound):
        return type(self)(summary_data = self["summary_data"] > lower_bound,
                          terms_idx = self["terms_idx"])
        
    def __ge__(self, lower_bound):
        return type(self)(summary_data = self["summary_data"] > lower_bound,
                          terms_idx = self["terms_idx"])
    
    
    def __and__(self, other_summary):
        assert isinstance(other_summary, type(self))
        assert np.array_equal(self["terms_idx"],other_summary["terms_idx"])
        assert self['summary_data'].dtype == np.bool
        assert other_summary['summary_data'].dtype == np.bool
        
        return type(self)(summary_data = self["summary_data"] &  other_summary['summary_data'],
                          terms_idx = self["terms_idx"])
        
    @property
    def _is_bool(self):
        return self['summary_data'].dtype == np.bool
    
    @property
    def _filtered_terms(self):
        assert self._is_bool
        
        return self["terms_idx"][self['summary_data']]
        
        


class SparseDocumentTermMatrix(SparseDataFrame):
    def __init__(self, sdtm, term_idx=None, doc_idx=None):
        
        super(SparseDocumentTermMatrix,self).__init__(sdtm, term_idx, doc_idx)
    
    
    @property
    def T(self):
        tr_sdtm = type(self)(sdtm = self["sdtm"].T,
                            term_idx = self["row_idx"],
                            doc_idx = self["col_idx"])
        return tr_sdtm
    
    @property
    def _term_idx(self):
        return self["col_idx"]
    
    @property
    def _doc_idx(self):
        return self["row_idx"]

    
    def summarize_sdtm(self, summarizer=lambda xx:xx.sum(axis=0)):
    
        summary_data = summarizer(self["sdtm"])
        
        if len(summary_data.shape) == 1:
            _summary_data = summary_data
        else:
            assert len(summary_data.shape) == 2
            assert summary_data.shape[0] == 1 or summary_data.shape[1] == 1
            
            if summary_data.shape[0] == 1:
                _summary_data = np.array(summary_data)[0,:]
            else:
                _summary_data = np.array(summary_data)[:,0]
            
        
        if _summary_data.shape[0] == self["sdtm"].shape[0]:
            return SparseDocumentTermMatrixSummary(summary_data = _summary_data,
                                                   terms_idx = self["row_idx"])
            
        if _summary_data.shape[0] == self["sdtm"].shape[1]:
            return SparseDocumentTermMatrixSummary(summary_data = _summary_data,
                                                   terms_idx = self["col_idx"])
            

            

            
# def vectorize_text(df, colname, query={},
#                    vect_gen=CountVectorizer, 
#                    vect_gen_init_kwargs = {"tokenizer":tokenize,"lowercase":False}):    



def vectorize_text(df, text_col=None, idx_col=None, 
                   cond_query={},
                   idx_query= [],
                   vect_gen=CountVectorizer, 
                   vect_gen_init_kwargs = {}):    
    
    """ 
    demo vect_gen_init_kwargs:
    vect_gen_init_kwargs = {"tokenizer":tokenize,"lowercase":False} 
    """
    
    assert text_col in df.columns
    
    if len(cond_query.keys()):
        
        for c in cond_query:
            assert c in df.columns
    
        query_conds = lambda :(df[i] == cond_query[i] for i in cond_query)
    
        qcs = query_conds()
    
        q_final = qcs.next()
    
        for q in qcs:
            q_final = q_final & q
    
        q_df = df[q_final]
        #print q_df.head()
        
    else:
        q_df = df

    
    if len(idx_query) > 0:
        q_df = q_df.ix[idx_query]

    
    vectorizer = vect_gen(**vect_gen_init_kwargs)
    
    vectorized_sdtm = vectorizer.fit_transform(q_df[text_col])
    
    if idx_col != None:
        assert idx_col in df.columns
        
        return_sdtm = SparseDocumentTermMatrix(sdtm = vectorized_sdtm, 
                                               term_idx=vectorizer.get_feature_names(), 
                                               doc_idx=q_df[idx_col].values)
    else:
        return_sdtm = SparseDocumentTermMatrix(sdtm = vectorized_sdtm, 
                                               term_idx=vectorizer.get_feature_names())
    
    return return_sdtm

                


if __name__ == '__main__':
    pass

