'''
Created on Mar 12, 2014

@author: c3h3
'''

import numpy as np
import scipy as sp
from PlaYnlp.sparse import L0_norm_col_summarizer as L0_col_sum
from PlaYnlp.sparse import L1_norm_col_summarizer as L1_col_sum


def top_m_weighted_features_sdtm(sdtm, init_group_ptr, top_m_features=50):
    sdtm_proj_on_group_sdtm = sdtm.select_rows(init_group_ptr)
    filtered_cols_sdtm = sdtm.select_columns(sdtm_proj_on_group_sdtm.summary > 0)
    within_wieghts = filtered_cols_sdtm.select_rows(init_group_ptr).summary._data
    all_weights = 1.0 / filtered_cols_sdtm.summarize_sdf(L1_col_sum)._data
    words_weights = all_weights * within_wieghts
    words_weights = words_weights / words_weights.sum()

    if len(words_weights) > top_m_features:
        filtered_features_ptrs = np.argsort(words_weights)[-top_m_features:]
        filtered_words_weights = words_weights[filtered_features_ptrs]
    else:
        filtered_features_ptrs = np.argsort(words_weights)
        filtered_words_weights = words_weights

    results_dict = {}
    results_dict["sdtm"] = filtered_cols_sdtm.select_columns(filtered_features_ptrs)
    results_dict["ws"] = filtered_words_weights
    results_dict["init_group_ptr"] = init_group_ptr

    return results_dict


def weighted_features_knn(sdtm, init_group_ptr, ws, top_k=20):
    diff = sdtm._smatrix - sp.sparse.kron(np.ones(len(sdtm._row_idx)), sdtm.select_rows(init_group_ptr)._smatrix.T).T
    ws_norm = np.apply_along_axis(lambda xx: (np.multiply(ws, np.abs(xx)).sum()), 1, diff.todense())
    ws_norm = ws_norm / np.max(ws_norm)
    results_dict = {}
    results_dict["top_k_idx"] = sdtm._row_idx[ws_norm.argsort()[:20]]
    results_dict["top_k_dist"] = ws_norm[ws_norm.argsort()[:20]]
    return results_dict


if __name__ == '__main__':
    pass
