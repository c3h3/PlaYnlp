# -*- coding: utf-8 -*-


import jieba, nltk

#ngram_no_filter = lambda text, n: (text[k:k+n] for k in range(len(text)-n+1))
#
#ngram = lambda text, n, filter_list=[]: (text[k:k+n] for k in range(len(text)-n+1) if not(text[k:k+n] in filter_list))

tokenize_gen = lambda token_fn: lambda text: list(token_fn(text)) if isinstance(text,(str,unicode)) else [] 


def ngram(text, n ,filter_list=[]):
    if isinstance(n, int):
        for k in range(len(text)-n+1):
            if not(text[k:k+n] in filter_list):
                yield text[k:k+n]
            
    if isinstance(n, list):
        for n_i in n:
            for xx in ngram(text=text, n=n_i, filter_list=filter_list):
                yield xx
                
                
def ngram_no_filter(text, n):
    if isinstance(n, int):
        for k in range(len(text)-n+1):
            yield text[k:k+n]
            
    if isinstance(n, list):
        for n_i in n:
            for xx in ngram(text=text, n=n_i):
                yield xx




if __name__ == '__main__':
    pass