'''
Created on Mar 30, 2014

@author: c3h3
'''

import numpy as np
import inspect
import types
import PlaYnlp.tokenizer as tkr 

def clean_no_data_tensors(np_array):
    return np_array[map(lambda xx:0 if xx==1 else slice(None,None,None),np_array.shape)]

class ValuesArray(np.ndarray):
    def __new__(cls, data, dtype=None):
        if isinstance(data, cls):
            return data
        else:
            values_array = np.array(data, dtype=dtype).view(cls)
            return values_array
        
    def decompose_into_states_ptrs(self):
        u,i = np.unique(self, return_inverse=True)
        states_array = StatesArray(data=u)
        ptr_array = PtrArray(data=i, eval_cls=type(self))
        return states_array, ptr_array

    
class StatesArray(np.ndarray):
    def __new__(cls, data, dtype=None):
        if isinstance(data, cls):
            return data
        else:
            states_array = np.unique(np.array(data, dtype=dtype)).view(cls)
            return states_array
    
    def _eval(self, ptr_array):
        assert isinstance(ptr_array, PtrArray)
        return self[ptr_array].view(ptr_array._eval_cls).copy()
        
    

class PtrArray(np.ndarray):
    def __new__(cls, data, dtype=None, eval_cls=np.ndarray):
        if isinstance(data, cls):
            return data
        else:
            data_array = np.array(data, dtype).view(cls)
            data_array._eval_cls = eval_cls
            return data_array
    
    def trans(self, ptrs_transform):
        return type(self)(data = ptrs_transform[self],eval_cls=self._eval_cls)

    


class StatesDictionary(object):
    def __init__(self, states_array):
        assert isinstance(states_array, StatesArray)
        self._states_array = states_array
        self._referred_by = []
    
    def _eval(self, ptr_array):
        assert isinstance(ptr_array, PtrArray)
        return self._states_array[ptr_array].view(ptr_array._eval_cls).copy()
    
    def update(self, new_states_array, ptrs_transform):
        self._states_array = new_states_array
        for one_states_data_array in self._referred_by:
            one_states_data_array._ptr_array = one_states_data_array._ptr_array.trans(ptrs_transform)
    
    
        
    
class StatesDataArray(object):
    def __init__(self, states_array, ptr_array):
        
        self._ptr_array = ptr_array
        self._states_dict = StatesDictionary(states_array=states_array)
        self._states_dict._referred_by.append(self)
    
    @property
    def _eval(self):
        return self._states_dict._eval(self._ptr_array)
    
    def __repr__(self):
        return self._states_dict._eval(self._ptr_array).__repr__()

    
        
def check_args(checker, *args):
    assert callable(checker)
    args_len = len(args)
    return np.ones(args_len,dtype=np.int)[np.array(map(checker,args))].sum() == args_len
    
def is_class_checker(xx):
    return inspect.isclass(xx)
    


def check_type_of_args(valid_type, *args):
    if isinstance(valid_type, (tuple,list)):
        assert check_args(is_class_checker,*valid_type)
        valid_types = tuple(valid_type)
    
    elif inspect.isclass(valid_type):
        valid_types = valid_type

    return check_args(lambda xx:isinstance(xx,valid_types),*args)
    

    
class StatesDictionaryMerger(list):
    def __init__(self, *states_dicts):
        assert check_type_of_args(StatesDictionary, *states_dicts)
        list.__init__(self,states_dicts)
        
    @property
    def merge(self):
        states_array_lens = map(lambda xx:len(xx._states_array),self)
        sector_position = map(lambda xx:slice(*xx),list(tkr.ngram(np.cumsum([0] + states_array_lens),2)))
        join_all_states_arrays = np.concatenate(tuple(map(lambda xx:xx._states_array,self)),axis=0)
        u,i = np.unique(join_all_states_arrays,return_inverse=True)        
        ptrs_transforms = map(lambda xx:i[xx],sector_position)
        
        self._new_states_array = StatesArray(data=u)
        self._ptrs_transforms = ptrs_transforms
        
        return self
    
    def update(self):
        for ptrs_trans, states_dict in zip(self._ptrs_transforms,self):
            states_dict.update(new_states_array = self._new_states_array, 
                               ptrs_transform = ptrs_trans)
        



if __name__ == '__main__':
    pass