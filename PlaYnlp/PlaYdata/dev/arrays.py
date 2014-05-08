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
            data._eval_cls = eval_cls
            return data
        else:
            data_array = np.array(data, dtype).view(cls)
            data_array._eval_cls = eval_cls
            return data_array
    
    def trans(self, ptrs_transform):
        return type(self)(data = ptrs_transform[self],eval_cls=self._eval_cls)

    def ngram(self, n):
        ngram_results = list(tkr.ngram(self,n))
        return type(self)(data=ngram_results, eval_cls=self._eval_cls)



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
    
    def __getitem__(self, *args):
        print "args = ",args
        if isinstance(args[0], PtrArray):
            return self._eval(args[0])
        else:
            return self._states_array[args].view(np.ndarray).copy()
            
        
    
    @property
    def _dtype(self):
        return self._states_array.dtype
        
    
class StatesDataArray(object):
    
    def __init__(self, states_array, ptr_array):
        
        assert isinstance(states_array,(StatesDictionary,StatesArray))
        assert isinstance(ptr_array, PtrArray)
        
        self._ptr_array = ptr_array    
        
        if isinstance(states_array,StatesDictionary):
            self._states_dict = states_array
        else:
            self._states_dict = StatesDictionary(states_array=states_array)
        
        self._states_dict._referred_by.append(self)
    
    @property
    def _eval(self):
        return self._states_dict._eval(self._ptr_array)
    
    
    def __repr__(self):
        return self._states_dict._eval(self._ptr_array).__repr__()
    
    
    @property
    def _dtype(self):
        return self._states_dict._dtype
    
    
    @property
    def _shape(self):
        return self._ptr_array.shape
    
    @property
    def _ncol(self):
        return self._ptr_array.shape[1] if len(self._ptr_array.shape) > 1 else 1
    
    
    @property
    def _nrow(self):
        return self._ptr_array.shape[0]
    
    
    def _nrow_ones(self, dtype=np.int64, reshape=True):
        if reshape:
            return np.ones(self._nrow, dtype=dtype).reshape((self._nrow,1))
        else:
            return np.ones(self._nrow, dtype=dtype)
    
    
    def _ncol_ones(self, dtype=np.int64, reshape=True):
        if reshape:
            return np.ones(self._ncol, dtype=dtype).reshape((1,self._ncol))
        else:
            return np.ones(self._ncol, dtype=dtype)
        
    
    
    def ngram(self, n, update=True):
        if update:
            self._ptr_array = self._ptr_array.ngram(n)
            return self
        else:
            return type(self)(states_array=self._states_dict,
                              ptr_array=self._ptr_array.ngram(n))
            
        
    def extend_rows(self, states_data_array, do_clean_no_data_tensor=True, update=True, _nrow_ones_kwargs={}):
        assert isinstance(states_data_array, type(self))
        
        new_ptr_array = np.kron(self._ptr_array,states_data_array._nrow_ones(**_nrow_ones_kwargs))
                                            
        print "new_ptr_array = ",new_ptr_array  
        print "type(new_ptr_array) = ",type(new_ptr_array)
        
        if do_clean_no_data_tensor:
            new_ptr_array = clean_no_data_tensors(new_ptr_array)
        
        new_ptr_array = PtrArray(data=new_ptr_array,eval_cls=ValuesArray)
        print "new_ptr_array = ",new_ptr_array                          
        print "type(new_ptr_array) = ",type(new_ptr_array)
        print "new_ptr_array._eval_cls = ",new_ptr_array._eval_cls
        
        if update:
            self._ptr_array = new_ptr_array
            return self
        else:
            return type(self)(states_array=self._states_dict,
                              ptr_array=new_ptr_array)
            
        
        

    
        
def check_args(checker, *args):
    assert callable(checker)
    args_len = len(args)
    return np.ones(args_len,dtype=np.int)[np.array(map(checker,args))].sum() == args_len
    
def class_checker(xx):
    return inspect.isclass(xx)
    

def check_type_of_args(valid_type, *args):
    if isinstance(valid_type, (tuple,list)):
        assert check_args(class_checker,*valid_type)
        valid_types = tuple(valid_type)
    
    elif inspect.isclass(valid_type):
        valid_types = valid_type

    return check_args(lambda xx:isinstance(xx,valid_types),*args)
    

    
class StatesDictionaryMerger(list):
    def __init__(self, *states_dicts):
        
        # checking states_dicts are all StatesDictionary's instance
        assert check_type_of_args(StatesDictionary, *states_dicts)
        
        # checking states_dicts have the same dtype
        assert len(np.unique(np.array(map(lambda xx:xx._dtype.type,states_dicts)))) == 1
        
        list.__init__(self,states_dicts)
        
    @property
    def _unique_states_array_ids(self):
        return np.unique(np.array(map(lambda xx:id(xx._states_array),self)))
        
    
    @property
    def merge(self):
        
        if len(self._unique_states_array_ids) > 1:
            states_array_lens = map(lambda xx:len(xx._states_array),self)
            sector_position = map(lambda xx:slice(*xx),list(tkr.ngram(np.cumsum([0] + states_array_lens),2)))
            join_all_states_arrays = np.concatenate(tuple(map(lambda xx:xx._states_array,self)),axis=0)
            u,i = np.unique(join_all_states_arrays,return_inverse=True)        
            ptrs_transforms = map(lambda xx:i[xx],sector_position)
        
            self._new_states_array = StatesArray(data=u)
            self._ptrs_transforms = ptrs_transforms
        
        else:
            self._new_states_array = self[0]._states_array
        
        return self
    
    def update(self):
        if len(self._unique_states_array_ids) > 1:
            for ptrs_trans, states_dict in zip(self._ptrs_transforms,self):
                states_dict.update(new_states_array = self._new_states_array, 
                                   ptrs_transform = ptrs_trans)
    
    


class StatesDataArrayMerger(list):
    _ptr_array_cls = PtrArray
    _ptr_array_eval_cls = ValuesArray
    
    def __init__(self, *states_data_arrays):
        
        # checking states_data_arrays are all StatesDataArray's instance
        assert check_type_of_args(StatesDataArray, *states_data_arrays)
        
        list.__init__(self,states_data_arrays)
        
    def merge(self,axis=0):
        assert axis in (0,1)
        
        # [axis=0] Appned more Rows
        if axis==0:
            
            # checking states_data_arrays has the same _ncol
            assert len(np.unique(map(lambda xx:xx._ncol,self))) == 1
            
            states_dict_merger = StatesDictionaryMerger(*map(lambda xx:xx._states_dict,self))
            states_dict_merger.merge.update()
            #print "states_dict_merger.merge._new_states_array = ",states_dict_merger.merge._new_states_array
            
            new_ptrs_array = self._ptr_array_cls(np.concatenate(tuple(map(lambda xx:xx._ptr_array,self)),axis=0),
                                                 eval_cls=self._ptr_array_eval_cls)
            new_states_array = states_dict_merger.merge._new_states_array
            
            return StatesDataArray(states_array=new_states_array, ptr_array=new_ptrs_array)
            
            
        # [axis=1] Appned more Columns
        else:
            
            # checking states_data_arrays has the same _nrow
            assert len(np.unique(map(lambda xx:xx._ncol,self))) == 1
            
        
    

    
    
class StatesDataFrame(object):
    pass

class StatesModelFrame(object):
    pass

        



if __name__ == '__main__':
    pass