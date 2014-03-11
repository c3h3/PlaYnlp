

from PlaYnlp.sparse import L0_norm_col_summarizer as L0_col_sum
from PlaYnlp.sparse import L1_norm_col_summarizer as L1_col_sum 



def find_similar_texts(sdtm, init_group_idx=[], eps=0.1):
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    print "init_group_idx = ",init_group_idx
    sdtm_proj_on_group_sdtm = sdtm.select_rows(init_group_idx)
    filtered_words_sdtm = sdtm.select_columns(sdtm_proj_on_group_sdtm.summary > 0)
    within_wieghts = filtered_words_sdtm.select_rows(init_group_idx).summary._data
    # within_wieghts = within_wieghts / float(within_wieghts.sum())
#     print "within_wieghts = ",within_wieghts

    all_weights = 1.0 / filtered_words_sdtm.summarize_sdf(L1_col_sum)._data
#     print "all_weights = ",all_weights
    words_weights = all_weights*within_wieghts
    words_weights = words_weights / words_weights.sum()
    
    print "words_weights = ",words_weights
    print "// ".join(filtered_words_sdtm._col_idx)

    sdtm_weighted_summary = filtered_words_sdtm.summarize_sdf(lambda xx:words_weights*xx.T)
    selected_post_idx = (sdtm_weighted_summary > eps)._filtered_idx
    
    print "selected_post_idx = ",selected_post_idx
    print "len(selected_post_idx) = ",len(selected_post_idx)
    
    return selected_post_idx


def find_text_eps_neighborhood(sdtm, init_group_idx=[], eps=0.1, max_iters=50, max_gropu_size=50):
    
    n_iteration = 0
    
    old_group_idx = init_group_idx
    old_step_n_text = len(old_group_idx)
    
    
    new_group_idx = find_similar_texts(sdtm = sdtm,
                                       init_group_idx = old_group_idx,
                                       eps = eps)
    
    new_step_n_text = len(new_group_idx)
    
    do_iteration = (old_step_n_text != new_step_n_text) and (n_iteration <= max_iters) and (len(new_group_idx) <= max_gropu_size) 
    
    while do_iteration:
        
        n_iteration = n_iteration + 1
        #print "n_iteration = ",n_iteration
        
        old_group_idx = new_group_idx
        old_step_n_text = len(old_group_idx)
    
    
        new_group_idx = find_similar_texts(sdtm = sdtm,
                                           init_group_idx = old_group_idx,
                                           eps = eps)
    
        new_step_n_text = len(new_group_idx)
    
        do_iteration = (old_step_n_text != new_step_n_text) and (n_iteration <= max_iters) and (len(new_group_idx) <= max_gropu_size)
        
    return new_group_idx if len(new_group_idx) > max_gropu_size else old_group_idx


if __name__ == '__main__':
    pass