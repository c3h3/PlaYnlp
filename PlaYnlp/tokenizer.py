'''
Created on Feb 27, 2014

@author: c3h3
'''

ngram_no_filter = lambda text, n: (text[k:k+n] for k in range(len(text)-n+1))

ngram = lambda text, n, filter_list=[]: (text[k:k+n] for k in range(len(text)-n+1) if not(text[k:k+n] in filter_list))




if __name__ == '__main__':
    pass