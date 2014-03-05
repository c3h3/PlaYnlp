# -*- coding: utf-8 -*-

import numpy as np

class SparseDataFrame(dict):
    def __init__(self, sdtm, col_idx=None, row_idx=None):
        self["sdtm"] = sdtm
        
        if col_idx != None:
            assert isinstance(col_idx, (list, np.ndarray))
            
            if isinstance(col_idx, list):
                assert self["sdtm"].shape[1] == len(col_idx)
                self["col_idx"] = np.array(col_idx)
                
            else:
                assert self["sdtm"].shape[1] == col_idx.shape[0]
                self["col_idx"] = np.array(col_idx)
        else:
            self["col_idx"] = np.arange(self["sdtm"].shape[1])
            
            
                
        if row_idx != None:
            assert isinstance(row_idx, (list, np.ndarray))
            
            if isinstance(row_idx, list):
                assert self["sdtm"].shape[0] == len(row_idx)
                self["row_idx"] = np.array(row_idx)
                
            else:
                assert self["sdtm"].shape[0] == row_idx.shape[0]
                self["row_idx"] = np.array(row_idx)
                
        else:
            self["row_idx"] = np.arange(self["sdtm"].shape[0])
    
    @property
    def _sdtm(self):
        return self["sdtm"]
    
    @property
    def T(self):
        tr_sdf = type(self)(sdtm = self["sdtm"].T,
                            col_idx = self["row_idx"],
                            row_idx = self["col_idx"])
        return tr_sdf
    
    @property
    def _col_idx(self):
        return self["col_idx"]
    
    @property
    def _row_idx(self):
        return self["row_idx"]

    

if __name__ == '__main__':
    pass