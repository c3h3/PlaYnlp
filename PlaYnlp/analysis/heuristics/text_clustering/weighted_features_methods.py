
import numpy as np


from PlaYnlp.sparse import L0_norm_col_summarizer as L0_col_sum
from PlaYnlp.sparse import L1_norm_col_summarizer as L1_col_sum 


def find_similar_text_ptrs(sdtm, init_group_ptr=[], eps=0.1):
    #    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    #    print "init_group_ptr = ",init_group_ptr
    
    #TODO: if not isinstance(init_group_ptr, (np.int,np.bool)):

    sdtm_proj_on_group_sdtm = sdtm.select_rows(init_group_ptr)
    filtered_words_sdtm = sdtm.select_columns(sdtm_proj_on_group_sdtm.summary > 0)
    within_wieghts = filtered_words_sdtm.select_rows(init_group_ptr).summary._data
    # within_wieghts = within_wieghts / float(within_wieghts.sum())
    #     print "within_wieghts = ",within_wieghts
    
    all_weights = 1.0 / filtered_words_sdtm.summarize_sdf(L1_col_sum)._data
    #     print "all_weights = ",all_weights
    words_weights = all_weights*within_wieghts
    words_weights = words_weights / words_weights.sum()
    
    #    print "words_weights = ",words_weights
    #    print "// ".join(filtered_words_sdtm._col_idx)
    
    sdtm_weighted_summary = filtered_words_sdtm.summarize_sdf(lambda xx:words_weights*xx.T)
    #    selected_post_idx = (sdtm_weighted_summary > eps)._filtered_idx
    selected_post_ptr = (sdtm_weighted_summary > eps)._filtered_ptrs
    
    return selected_post_ptr
    

def find_text_eps_neighborhood(sdtm, init_group_ptr=[], eps=0.1, max_iters=50, max_gropu_size=50):

    #TODO: if not isinstance(init_group_ptr, (np.int,np.bool)):
    
    n_iteration = 0
    
    old_group_ptr = init_group_ptr
    old_step_n_text = len(old_group_ptr)
    
    
    new_group_ptr = find_similar_text_ptrs(sdtm = sdtm,
                                           init_group_ptr = old_group_ptr,
                                           eps = eps)
    
    new_step_n_text = len(new_group_ptr)
    
    do_iteration = (old_step_n_text != new_step_n_text) and (n_iteration <= max_iters) and (len(new_group_ptr) <= max_gropu_size) 
    
    while do_iteration:
        
        n_iteration = n_iteration + 1
        #print "n_iteration = ",n_iteration
        
        old_group_ptr = new_group_ptr
        old_step_n_text = len(old_group_ptr)
    
    
        new_group_ptr = find_similar_text_ptrs(sdtm = sdtm,
                                               init_group_ptr = old_group_ptr,
                                               eps = eps)
    
        new_step_n_text = len(new_group_ptr)
    
        do_iteration = (old_step_n_text != new_step_n_text) and (n_iteration <= max_iters) and (len(new_group_ptr) <= max_gropu_size)
        
    return new_group_ptr if len(new_group_ptr) > max_gropu_size else old_group_ptr


if __name__ == '__main__':
    pass