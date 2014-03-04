# -*- coding: utf-8 -*-
'''
Created on Feb 26, 2014

@author: c3h3
'''

from .sparse import SparseDataFrame
import numpy as np


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
    def _filtered_terms(self):
        assert self['summary_data'].dtype == np.bool
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
            
    
    


if __name__ == '__main__':
    pass

