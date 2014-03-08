# -*- coding: utf-8 -*-


import jieba, nltk

ngram_no_filter = lambda text, n: (text[k:k+n] for k in range(len(text)-n+1))

ngram = lambda text, n, filter_list=[]: (text[k:k+n] for k in range(len(text)-n+1) if not(text[k:k+n] in filter_list))

tokenize_gen = lambda token_fn: lambda text: list(token_fn(text)) if isinstance(text,(str,unicode)) else [] 



if __name__ == '__main__':
    pass