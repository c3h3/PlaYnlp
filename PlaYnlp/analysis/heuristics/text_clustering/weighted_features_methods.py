
import numpy as np


from PlaYnlp.sparse import L0_norm_col_summarizer as L0_col_sum
from PlaYnlp.sparse import L1_norm_col_summarizer as L1_col_sum 



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
    
        do_iteration = (old_step_n_text != new_step_n_text) and (n_iteration <= max_iters) and (len(new_group_ptr) <= max_gropu_size)
        
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