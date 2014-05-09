# -*- coding: utf-8 -*-


import jieba, nltk
import re

#ngram_no_filter = lambda text, n: (text[k:k+n] for k in range(len(text)-n+1))
#
#ngram = lambda text, n, filter_list=[]: (text[k:k+n] for k in range(len(text)-n+1) if not(text[k:k+n] in filter_list))

tokenize_gen = lambda token_fn: lambda text: list(token_fn(text)) if isinstance(text, (str, unicode)) else []


def ngram(text, n , filter_list=[]):
    if isinstance(n, int):
        for k in range(len(text) - n + 1):
            if not(text[k:k + n] in filter_list):
                yield text[k:k + n]

    if isinstance(n, list):
        for n_i in n:
            for xx in ngram(text=text, n=n_i, filter_list=filter_list):
                yield xx


def ngram_no_filter(text, n):
    if isinstance(n, int):
        for k in range(len(text) - n + 1):
            yield text[k:k + n]

    if isinstance(n, list):
        for n_i in n:
            for xx in ngram(text=text, n=n_i):
                yield xx


def skipped_ngram(text, n, sep=" ", skip_pattern=r"[A-Za-z0-9]+", filter_list=[]):
    sep_text = text.split(sep)
    for one_text in sep_text:
        if len(one_text) > 0:
            check_skip_pattern = re.match(skip_pattern, one_text)
            if check_skip_pattern == None:
                for yy in ngram(one_text, n, filter_list=filter_list):
                    yield yy
            else:
                yield one_text



if __name__ == '__main__':
    pass
