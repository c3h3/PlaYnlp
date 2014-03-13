
import numpy as np


from PlaYnlp.sparse import L0_norm_col_summarizer as L0_col_sum
from PlaYnlp.sparse import L1_norm_col_summarizer as L1_col_sum 


class WieghtedFeaturesNeighborhood(dict):
    _key_mapper = {"proj_sdtm":"projected_sdtm",
                   "inv_summarizer":"inversed_summarizer",}
    
    def __init__(self, sdtm, init_ptrs, inversed_summarizer=L1_col_sum):
        self["sdtm"] = sdtm
        self["init_ptrs"] = init_ptrs
        self["projected_sdtm"] = self["sdtm"].select_columns(self["sdtm"].select_rows(self["init_ptrs"]).summary > 0)
        self["inversed_summarizer"] = inversed_summarizer
        
        
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
    
    
    @property
    def _within_wieghts(self):
        return self["projected_sdtm"].select_rows(self["init_ptrs"]).summary._data
    
    
    @property
    def _all_inversed_wieghts(self):
        return 1.0 / self["projected_sdtm"].summarize_sdf(self["inversed_summarizer"])._data
    
    
    @property
    def _words_weights(self):
        temp_words_weights = self._all_inversed_wieghts*self._within_wieghts 
        return temp_words_weights / temp_words_weights.sum()

    
    @property
    def _weighted_summary(self):
        return self["projected_sdtm"].summarize_sdf(lambda xx:self._words_weights*xx.T)
    
    
    def get_topk_neighbors_ptrs(self, k=20,  reverse=False):
        return self._weighted_summary.top_k_ptrs(k, reverse)

    
    def get_topk_neighbors(self, k=20,  reverse=False):
        return type(self)(sdtm = self._sdtm,
                          init_ptrs = self.get_topk_neighbors_ptrs(k=k,reverse=reverse),
                          inversed_summarizer = self._inversed_summarizer)


    def find_stable_topk_neighborhood_ptrs(self, k=20, max_iters=50, min_eps=0.1, reverse=False, return_only_ptrs=True):
        n_iteration = 0
    
        old_neighborhood = self
        old_neighborhood_ptrs_set = set(old_neighborhood._init_ptrs)
        
        
        new_neighborhood = old_neighborhood.get_topk_neighbors(k=k, reverse=reverse)
        new_neighborhood_ptrs_set = set(new_neighborhood._init_ptrs)
        
        sym_diff = new_neighborhood_ptrs_set.symmetric_difference(old_neighborhood_ptrs_set)
        
        #TODO: with min_eps constraint
        do_iteration = (len(sym_diff) > 0) and (n_iteration <= max_iters) 
        
        while do_iteration:
            
            n_iteration = n_iteration + 1
            #print "n_iteration = ",n_iteration
            
            old_neighborhood = new_neighborhood
            old_neighborhood_ptrs_set = set(old_neighborhood._init_ptrs)
        
        
            new_neighborhood = old_neighborhood.get_topk_neighbors(k=k, reverse=reverse)
            new_neighborhood_ptrs_set = set(new_neighborhood._init_ptrs)
        
            sym_diff = new_neighborhood_ptrs_set.symmetric_difference(old_neighborhood_ptrs_set)
            
            #TODO: with min_eps constraint
            do_iteration = (len(sym_diff) > 0) and (n_iteration <= max_iters)
        
        
        return_neighborhood = new_neighborhood 
        
        if return_only_ptrs:
            return return_neighborhood._init_ptrs
        else:
            return return_neighborhood

        
    def get_mins_neighbors_ptrs(self, mins=0.1):
        return (self._weighted_summary >= mins)._filtered_ptrs
        
        
    def get_mins_neighbors(self, mins=0.1):
        return type(self)(sdtm = self._sdtm,
                          init_ptrs = self.get_mins_neighbors_ptrs(mins=mins),
                          inversed_summarizer = self._inversed_summarizer)
    

    def find_stable_mins_neighborhood(self, mins=0.1, max_iters=50, max_group_size=50, return_only_ptrs=True):
        n_iteration = 0
    
        old_neighborhood = self
        old_neighborhood_ptrs_set = set(old_neighborhood._init_ptrs)
        
        
        new_neighborhood = old_neighborhood.get_mins_neighbors(mins=mins)
        new_neighborhood_ptrs_set = set(new_neighborhood._init_ptrs)
        
        sym_diff = new_neighborhood_ptrs_set.symmetric_difference(old_neighborhood_ptrs_set)
        
        do_iteration = (len(sym_diff) > 0) and (n_iteration <= max_iters) and (len(new_neighborhood_ptrs_set) <= max_group_size) 
        
        while do_iteration:
            
            n_iteration = n_iteration + 1
            #print "n_iteration = ",n_iteration
            
            old_neighborhood = new_neighborhood
            old_neighborhood_ptrs_set = set(old_neighborhood._init_ptrs)
        
        
            new_neighborhood = old_neighborhood.get_mins_neighbors(mins=mins)
            new_neighborhood_ptrs_set = set(new_neighborhood._init_ptrs)
        
            sym_diff = new_neighborhood_ptrs_set.symmetric_difference(old_neighborhood_ptrs_set)
        
            do_iteration = (len(sym_diff) > 0) and (n_iteration <= max_iters) and (len(new_neighborhood_ptrs_set) <= max_group_size)
        
        
        return_neighborhood = new_neighborhood if  len(new_neighborhood_ptrs_set) > max_group_size else old_neighborhood
        
        if return_only_ptrs:
            return return_neighborhood._init_ptrs
        else:
            return return_neighborhood

        
        
    
    


def weighted_features_summarizer(sdtm, init_group_ptr):
    sdtm_proj_on_group_sdtm = sdtm.select_rows(init_group_ptr)
    filtered_words_sdtm = sdtm.select_columns(sdtm_proj_on_group_sdtm.summary > 0)
    within_wieghts = filtered_words_sdtm.select_rows(init_group_ptr).summary._data

    all_weights = 1.0 / filtered_words_sdtm.summarize_sdf(L1_col_sum)._data
    words_weights = all_weights*within_wieghts
    words_weights = words_weights / words_weights.sum()

    sdtm_weighted_summary = filtered_words_sdtm.summarize_sdf(lambda xx:words_weights*xx.T)
    
    return sdtm_weighted_summary


def get_eps_neighborhood_ptrs(sdtm, init_group_ptr=[], eps=0.1):
    return (weighted_features_summarizer(sdtm=sdtm, init_group_ptr=init_group_ptr) >= eps)._filtered_ptrs

    
def get_topk_neighborhood_ptrs(sdtm, init_group_ptr=[], k=20, reverse=False):
    return weighted_features_summarizer(sdtm=sdtm, init_group_ptr=init_group_ptr).top_k_ptrs(k, reverse)


def find_stable_eps_neighborhood(sdtm, init_group_ptr=[], eps=0.1, max_iters=50, max_group_size=50):

    #TODO: if not isinstance(init_group_ptr, (np.int,np.bool)):
    
    n_iteration = 0
    
    old_group_ptr = init_group_ptr
    old_step_n_text = len(old_group_ptr)
    
    
    new_group_ptr = get_eps_neighborhood_ptrs(sdtm = sdtm,
                                               init_group_ptr = old_group_ptr,
                                               eps = eps)
    
    new_step_n_text = len(new_group_ptr)
    
    do_iteration = (old_step_n_text != new_step_n_text) and (n_iteration <= max_iters) and (len(new_group_ptr) <= max_group_size) 
    
    while do_iteration:
        
        n_iteration = n_iteration + 1
        #print "n_iteration = ",n_iteration
        
        old_group_ptr = new_group_ptr
        old_step_n_text = len(old_group_ptr)
    
    
        new_group_ptr = get_eps_neighborhood_ptrs(sdtm = sdtm,
                                                   init_group_ptr = old_group_ptr,
                                                   eps = eps)
    
        new_step_n_text = len(new_group_ptr)
    
        do_iteration = (old_step_n_text != new_step_n_text) and (n_iteration <= max_iters) and (len(new_group_ptr) <= max_group_size)
        
    return new_group_ptr if len(new_group_ptr) > max_group_size else old_group_ptr


def find_stable_topk_neighborhood(sdtm, init_group_ptr=[], k=20, max_iters=50, min_eps=0.05, reverse=False):

    #TODO: if not isinstance(init_group_ptr, (np.int,np.bool)):
    n_iteration = 0
    #print "n_iteration = ",n_iteration
    
    old_group_ptr = init_group_ptr
    old_group_ptr_set = set(old_group_ptr)
    #print "old_group_ptr = ",old_group_ptr
    
    new_group_ptr = get_topk_neighborhood_ptrs(sdtm=sdtm, 
                                               init_group_ptr=old_group_ptr, 
                                               k=k, 
                                               reverse=reverse)
    
    #print "new_group_ptr = ",new_group_ptr
    
    new_group_ptr_set = set(new_group_ptr)
    sym_diff = new_group_ptr_set.symmetric_difference(old_group_ptr_set)
    #print len(sym_diff)
    
        
    #TODO: with min_eps constraint
    do_iteration = (len(sym_diff) > 0) and (n_iteration <= max_iters)
    #print "do_iteration = ",do_iteration
    
    while do_iteration:
            
        n_iteration = n_iteration + 1
        #print "n_iteration = ",n_iteration
            
        old_group_ptr = new_group_ptr
        old_group_ptr_set = set(old_group_ptr)
        #print "old_group_ptr = ",old_group_ptr
        
        
        new_group_ptr = get_topk_neighborhood_ptrs(sdtm=sdtm, 
                                                   init_group_ptr=old_group_ptr, 
                                                   k=k, 
                                                   reverse=reverse)
        
        new_group_ptr_set = set(new_group_ptr)
        sym_diff = new_group_ptr_set.symmetric_difference(old_group_ptr_set)
        #print len(sym_diff)
    
        do_iteration = (len(sym_diff) > 0) and (n_iteration <= max_iters)
        #print "do_iteration = ",do_iteration
    
    return new_group_ptr 



if __name__ == '__main__':
    pass