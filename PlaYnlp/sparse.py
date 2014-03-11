# -*- coding: utf-8 -*-

import numpy as np
import pickle

#L1_norm_col_summarizer = lambda xx:np.abs(xx).sum(axis=0)
#L0_norm_col_summarizer = lambda xx:xx.sign().sum(axis=0)

def L1_norm_col_summarizer(xx):
    return np.abs(xx).sum(axis=0)

def L0_norm_col_summarizer(xx):
    return xx.sign().sum(axis=0)
    


class SparseDataFrameSummary(dict):
    _key_mapper = {"data":"summary_data"}
    
    def __init__(self, summary_data, summary_idx, sdf=None):
        self["summary_data"] = summary_data
        self["summary_idx"] = summary_idx
        
        if sdf != None:
            self["sdf"] = sdf
            if self["sdf"].is_matched_col_shape(self['summary_data']):        
                self["summary_type"] = "col"
        
            if self["sdf"].is_matched_row_shape(self['summary_data']):     
                self["summary_type"] = "row"
    
        
    def __getstate__(self):
        pass
    
     
    def __setstate__(self, state):
        pass
    
             
    def __getattr__(self, key):
        
        if key.startswith("_") and key[1:] in self.keys():
            return self[key[1:]]
        else:
            
            if key.startswith("_") and key[1:] in self._key_mapper.keys() and self._key_mapper[key[1:]] in self.keys():
                return self[self._key_mapper[key[1:]]]
            else:
                return None
            
        
    def __lt__(self, upper_bound):
        return type(self)(summary_data = self["summary_data"] < upper_bound,
                          summary_idx = self["summary_idx"],
                          sdf = self["sdf"])
    
    
    def __le__(self, upper_bound):
        return type(self)(summary_data = self["summary_data"] <= upper_bound,
                          summary_idx = self["summary_idx"],
                          sdf = self["sdf"])
    
    
    def __gt__(self, lower_bound):
        return type(self)(summary_data = self["summary_data"] > lower_bound,
                          summary_idx = self["summary_idx"],
                          sdf = self["sdf"])
        
        
    def __ge__(self, lower_bound):
        return type(self)(summary_data = self["summary_data"] > lower_bound,
                              summary_idx = self["summary_idx"],
                              sdf = self["sdf"])
    
    
    def __and__(self, other_summary):
        assert isinstance(other_summary, type(self))
        assert np.array_equal(self["summary_idx"],other_summary["summary_idx"])
        assert self['summary_data'].dtype == np.bool
        assert other_summary['summary_data'].dtype == np.bool
        
        return type(self)(summary_data = self["summary_data"] &  other_summary['summary_data'],
                          summary_idx = self["summary_idx"],
                          sdf = self["sdf"])
        
    @property
    def _is_bool(self):
        return self['summary_data'].dtype == np.bool
    
    @property
    def _has_sdf(self):
        return "sdf" in self.keys()
    
    @property
    def _filtered_idx(self):
        assert self._is_bool
        
        return self["summary_idx"][self['summary_data']]
    
    @property
    def _summary_type(self):
        assert self._has_sdf
        return self["summary_type"]
    
    @property
    def _sub_sdf(self):
        assert self._is_bool and self._has_sdf
        
        if self["sdf"].is_matched_col_shape(self['summary_data']):        
            return self["sdf"].select_columns(select_col = self['summary_data'])
        
        if self["sdf"].is_matched_row_shape(self['summary_data']):        
            return self["sdf"].select_rows(select_row = self['summary_data'])
        


class SparseDataFrame(dict):
    _key_mapper = {}
    _summerizer_class = SparseDataFrameSummary
    _dump_file_prefix = "sdf"
    
    def __init__(self, smatrix, col_idx=None, row_idx=None, summarizer=None):
        self["smatrix"] = smatrix
        
        if col_idx != None:
            assert isinstance(col_idx, (list, np.ndarray))
            
            if isinstance(col_idx, list):
                assert self["smatrix"].shape[1] == len(col_idx)
                self["col_idx"] = np.array(col_idx)
                
            else:
                assert self["smatrix"].shape[1] == col_idx.shape[0]
                self["col_idx"] = np.array(col_idx)
        else:
            self["col_idx"] = np.arange(self["smatrix"].shape[1])
            
                        
        if row_idx != None:
            assert isinstance(row_idx, (list, np.ndarray))
            
            if isinstance(row_idx, list):
                assert self["smatrix"].shape[0] == len(row_idx)
                self["row_idx"] = np.array(row_idx)
                
            else:
                assert self["smatrix"].shape[0] == row_idx.shape[0]
                self["row_idx"] = np.array(row_idx)
                
        else:
            self["row_idx"] = np.arange(self["smatrix"].shape[0])
            
        
        if summarizer != None and callable(summarizer):
            self["summarizer"] = summarizer
            
    
    def __getstate__(self):
        pass
    
     
    def __setstate__(self, state):
        pass
    
    
    def __getattr__(self, key):
        
        if key.startswith("_") and key[1:] in self.keys():
            return self[key[1:]]
        else:
            
            if key.startswith("_") and key[1:] in self._key_mapper.keys() and self._key_mapper[key[1:]] in self.keys():
                return self[self._key_mapper[key[1:]]]
            else:
                return None

        
#     @property
#     def _smatrix(self):
#         return self["smatrix"]
    
#     @property
#     def _col_idx(self):
#         return self["col_idx"]
    
#     @property
#     def _row_idx(self):
#         return self["row_idx"]


    @property
    def T(self):
        tr_sdf = type(self)(smatrix = self["smatrix"].T,
                            col_idx = self["row_idx"],
                            row_idx = self["col_idx"],
                            summarizer = self["summarizer"] if self._has_default_summarizer else None)
        return tr_sdf
    
    
    @property
    def _has_default_summarizer(self):
        return "summarizer" in self.keys()
    
    
    @property
    def summary(self):
        if self._has_default_summarizer:
            return self.summarize_sdf(summarizer = self["summarizer"]) 
            
    
    def change_default_summerizer(self, summarizer=None):
        if summarizer != None and callable(summarizer):
            self["summarizer"] = summarizer
            return True
        else:
            return False
    
    
    def summarize_sdf(self, summarizer=L1_norm_col_summarizer):
    
        summary_data = summarizer(self["smatrix"])
        
        if len(summary_data.shape) == 1:
            _summary_data = summary_data
        else:
            assert len(summary_data.shape) == 2
            assert summary_data.shape[0] == 1 or summary_data.shape[1] == 1
            
            if summary_data.shape[0] == 1:
                _summary_data = np.array(summary_data)[0,:]
            else:
                _summary_data = np.array(summary_data)[:,0]
            
        
        if _summary_data.shape[0] == self["smatrix"].shape[0]:
            return self._summerizer_class(summary_data = _summary_data,
                                          summary_idx = self["row_idx"],
                                          sdf = self)
            
        if _summary_data.shape[0] == self["smatrix"].shape[1]:
            return self._summerizer_class(summary_data = _summary_data,
                                          summary_idx = self["col_idx"],
                                          sdf = self)

    
    def select_columns(self, select_col = None):
        
        if select_col != None:
            if isinstance(select_col, self._summerizer_class) and select_col._is_bool:
                _select_col_idx = select_col._data
            else:
                _select_col_idx = select_col
        else:
            _select_col_idx = np.arange(len(self["col_idx"]))
        
        new_col_idx = self["col_idx"][_select_col_idx]
        
        new_smatrix = self["smatrix"][:,_select_col_idx]
        
        return type(self)(smatrix = new_smatrix,
                          col_idx = new_col_idx,
                          row_idx = self["row_idx"],
                          summarizer = self["summarizer"] if self._has_default_summarizer else None)
    
    
    def select_rows(self, select_row = None):
        if select_row != None:
            if isinstance(select_row, self._summerizer_class) and select_row._is_bool:
                _select_row_idx = select_row._data
            else:
                _select_row_idx = select_row
        else:
            _select_row_idx = np.arange(len(self["row_idx"]))

        new_row_idx = self["row_idx"][_select_row_idx]
        
        new_smatrix = self["smatrix"][_select_row_idx,:]
        
        return type(self)(smatrix = new_smatrix,
                          col_idx = self["col_idx"],
                          row_idx = new_row_idx,
                          summarizer = self["summarizer"] if self._has_default_summarizer else None)
    
    
    def sub_sdf(self, select_col = None, select_row = None):
        return self.select_columns(select_col).select_rows(select_row)
        
    
    def is_matched_col_shape(self, vec):
        if isinstance(vec, list):
            return len(vec) == self["smatrix"].shape[1]
        
        if isinstance(vec, np.ndarray):
            assert len(vec.shape) == 1
            return vec.shape[0] == self["smatrix"].shape[1]
        
    def is_matched_row_shape(self,vec):
        if isinstance(vec, list):
            return len(vec) == self["smatrix"].shape[0]
        
        if isinstance(vec, np.ndarray):
            assert len(vec.shape) == 1
            return vec.shape[0] == self["smatrix"].shape[0]
            
        
    def is_col_vec(self,vec):
        return self.is_matched_row_shape(vec)
        
        
    def is_row_vec(self,vec):
        return self.is_matched_col_shape(vec)
        
        
    def to_pickle_file(self, output_file, with_prefix=True, close_after_dump=True):
        
        if isinstance(output_file, file):
            assert not output_file.closed
            pickle.dump(self, output_file)
            
            if close_after_dump:
                output_file.close()
                
        
        elif isinstance(output_file, (str,unicode)):
            #TODO: output_file includes filename and path 
            with open(output_file, "wb") as wfile:
                pickle.dump(self, wfile)
        
        
                
            
            
            
            
            
            
            
            
        
        

if __name__ == '__main__':
    pass