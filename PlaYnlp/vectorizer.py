# -*- coding: utf-8 -*-

from .sparse import SparseDataFrame
from sklearn.feature_extraction.text import CountVectorizer

class SparseDocumentTermMatrix(SparseDataFrame):
    _key_mapper = {"sdtm":"smatrix",
                   "term_idx":"col_idx",
                   "doc_idx":"row_idx",
                   "vec":"vectorizer"}
    _dump_file_prefix = "sdtm"
    
    @property
    def T(self):
        tr_sdf = SparseTermDocumentMatrix(smatrix = self._sdtm.T,
                                          row_idx = self._term_idx,
                                          col_idx = self._doc_idx,
                                          summarizer = self["summarizer"] if self._has_default_summarizer else None,
                                          vectorizer = self["summarizer"] if "vectorizer" in self.keys() else None)
        return tr_sdf
    
    
class SparseTermDocumentMatrix(SparseDataFrame):
    _key_mapper = {"stdm":"smatrix",
                   "term_idx":"row_idx",
                   "doc_idx":"col_idx",
                   "vec":"vectorizer"}
    
    _dump_file_prefix = "stdm"
    
    @property
    def T(self):
        tr_sdf = SparseDocumentTermMatrix(smatrix = self._stdm.T,
                                          col_idx = self._term_idx,
                                          row_idx = self._doc_idx,
                                          summarizer = self["summarizer"] if self._has_default_summarizer else None,
                                          vectorizer = self["summarizer"] if "vectorizer" in self.keys() else None)
        return tr_sdf
    



    
def vectorize_text(df, text_col=None, idx_col=None, 
                   cond_query={},
                   idx_query=[],
                   vect_gen=CountVectorizer, 
                   vect_gen_init_kwargs={},
                   summarizer=None,
                   dump_out_pickle=None):    
    
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
        
        return_sdtm = SparseDocumentTermMatrix(smatrix = vectorized_sdtm, 
                                               col_idx=vectorizer.get_feature_names(), 
                                               row_idx=q_df[idx_col].values,
                                               summarizer=summarizer)#,vectorizer=vectorizer)
    else:
        return_sdtm = SparseDocumentTermMatrix(smatrix = vectorized_sdtm, 
                                               col_idx=vectorizer.get_feature_names(),
                                               summarizer=summarizer)#,vectorizer=vectorizer)
    
    if isinstance(dump_out_pickle, (file, str, unicode)):
        return_sdtm.to_pickle_file(output_file=dump_out_pickle)
    
    
    return return_sdtm
                


if __name__ == '__main__':
    pass

