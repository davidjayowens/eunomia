"""


"""

from __future__ import annotations


from typing import Literal
import re

import numpy as np
import pandas as pd

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.util import ngrams


def make_bow(text: str, 
             filter: bool = True, 
             stopwords: list[str] | None = None,
             stem: bool = False) -> list[list[str]] | None:
    """
    Turn a text into a Bag of Words representation.

    Parameters
    ----------
    filter : bool, default True
        If True, applies a series of filters to the text during processing.

    stopwords : list of str, optional
        Can provide a custom list of stopwords to be removed from document texts.
        If no list is provided, uses default stopwords list.

    stem : bool
        If True, stems (truncates) word tokens. This increases matchability 
        across the bag (document) and the larger collection (corpus).

    """
    # FIXME: Refine the filters; should be possible to filter numbers but not stopwords, eg
    # Stopwords to be filtered out (list based on: http://www.leginfo.ca.gov/help/stopwords.html)
    stopwords = ['about','above','according','across','actually','adj','after',
                'afterwards','again','all','almost','alone','along','already',
                'also','although','always','among','amongst','and','another',
                'any','anyhow','anyone','anything','anywhere','are','aren','around',
                'because','become','becomes','becoming',
                'been','before','beforehand','begin','beginning','behind','below',
                'beside','besides','between','beyond','billion','both','but',
                'can','cannot','caption','could','couldn','did','didn',
                'does','doesn','don','down','during','each','eight','eighty',
                'either','else','elsewhere','end','ending','enough','etc','even',
                'ever','every','everyone','everything','everywhere','except','few',
                'fifty','first','five','for','former','formerly','forty','found',
                'four','form','from','further','had','has','hasn','have','haven',
                'hence','her','here','hereafter','hereby','herein','hereupon',
                'hers','herself','him','himself','his','how','however','hundred',
                'inc','indeed','instead','into','isn',
                'its','itself','last','later','latter','latterly','least','less',
                'let','like','likely','ltd','made','make','makes','many',
                'maybe','meantime','meanwhile','might','million','miss','more',
                'moreover','most','mostly','mrs','much','must','myself',
                'namely','neither','never','nevertheless','next','nine','ninety',
                'nobody','none','nonetheless','noon','nor','not','nothing',
                'now','nowhere','off','often','once','one','only','onto',
                'other','others','otherwise','our','ours','ourselves','out',
                'over','overall','own','per','perhaps','rather','recent',
                'recently','same','seem','seemed','seeming','seems','seven',
                'seventy','several','she','should','shouldn','since','six','sixty',
                'some','somehow','someone','sometime','sometimes','somewhere',
                'still','stop','such','talking','ten','than','that','the','their',
                'them','themselves','thence','there','thereafter','thereby',
                'therefore','therein','thereupon','these','they','thirty',
                'this','those','though','thousand','three','through','throughout',
                'thru','thus','together','too','toward','towards','trillion',
                'twenty','two','under','unless','unlike','unlikely','until',
                'upon','used','using','via','very','was','wasn',
                'well','were','weren','what','whatever','when','whence',
                'whenever','where','whereafter','whereas','whereby','wherein',
                'whereupon','wherever','whether','which',
                'while','whither','who','whoever','whole','whom','whomever','whose',
                'why','will','with','within','without','won','would','wouldn','xxxx','yes',
                'yet','you','your','yours','yourself','yourselves',
                'endstatute','pdf','wpd']
    
    # First filter pass:
    def filter_nums(text):
        # Filter numbers and non-terminal punctuation
        pattern1 = re.compile(r'(\d+)') # numbers
        pattern2 = re.compile(r'["#$%&\'()*+,\-/\\:;<=>@[\]^_`{|}~]') # all punctuation except for: . ! ?
        pattern3 = re.compile(r'\b(?=[mdclxvi])m*(c[md]|d?c{0,3})(x[cl]|l?x{0,3})(i[xv]|v?i{0,3})\b') # roman numerals

        text_filtered = text
        for pattern in [pattern1, pattern2, pattern3]:
            text_filtered = re.sub(pattern, ' ', text_filtered)
        
        return(text_filtered)

    # Second filter pass:
    def filter_small(sent: str):
        # Remove remaining punctuation, words with fewer than 3 chars,
        pattern4 = re.compile(r'([.!?]+)') # terminal punctuation that was skipped in first filter: . ! ?
        pattern5 = re.compile(r'\b\w{1,2}\b')  
        
        sent_filtered = sent
        for pattern in [pattern4, pattern5]:
            sent_filtered = re.sub(pattern, '', sent_filtered)
        
        return(sent_filtered)
    
    # Create sentence-level strings (tokens)
    if filter:
        sent_tokens = sent_tokenize(filter_nums(text))
    else:
        sent_tokens = sent_tokenize(text)

    # Create word-level strings (tokens)
    if filter:
        word_tokens = [word_tokenize(filter_small(sent)) for sent in sent_tokens]
    else:
        word_tokens = [word_tokenize(sent) for sent in sent_tokens]

    # Filter out stopwords & empty strings
    if filter:
        final_words = [[word for word in sent if (word not in stopwords) and (len(word) > 0)] for sent in word_tokens]
    else:
        final_words = [[word for word in sent if (len(word) > 0)] for sent in word_tokens]
    # Filter empty sentences
    final_words = [sent for sent in final_words if len(sent)>0]

    # Stem
    if stem:
        stemmer = SnowballStemmer('english')
        bag_of_words = [[stemmer.stem(word) for word in sent] for sent in final_words]
    else:
        bag_of_words = final_words

    return(bag_of_words)

    
def make_gram_tf(bow: list[list[str]],
                 gram_n: int = 3) -> pd.Series:
    """
    Make stem-/n-gram term frequency (tf) dictionary from a bag of words.

    Term frequencies are calculated as the term occurence (number of 
    times a term appears in a document) out of the total count of
    all (1-gram) terms in the document (bag size).

    Terms with only a single occurrence are not retained in the final
    vector vocabulary.

    Parameters
    ----------
    bow : list[][] of type str
        Contains lists of lists of strings. Each inner list in the outer list
        is a 'sentence' of word tokens.

    gram_n : int, default 3
        Makes n-grams between n=1 and gram_n. N-grams are concatenations 
        (hyphen-delimited) of sequences of adjacent terms in a sentence.
        If gram_n=1, only individual terms are used in the production of 
        the term frequencies (equivalent to using make_tf).

    TODO: Need to think about how to handle n-gram frequency vectors
        > For 'tf' scores should 3-grams be count / sum of all 3-gram counts?
        > Use bag size (count of all 1-grams) as denominator of all n-grams? (default)
        > Return Series instead of dict - labeled indices; math will be faster
    """
    bag_size = len([word for sent in bow for word in sent])

    def sent_grams(sent, gram_n):
        results = []
        for i in range(1,gram_n+1):
            # Add each n-gram to the list
            results.extend(['-'.join(grams) for grams in ngrams(sent, i)])

        return(results)
    
    token_grams = [sent_grams(sent, gram_n) for sent in bow]

    return(make_tf(token_grams, bag_size))


def make_tf(bow: list[list[str]],
            bag_size: int | None = None) -> pd.Series:
    """
    Uses the Bag of Words from a single document's text to create 
    a dictionary of term frequencies.

    Term frequencies are calculated as the term occurence (number of 
    times a term appears in a document) out of the total count of
    all terms in the document (bag size).

    Terms with only a single occurrence are not retained in the final
    vector vocabulary.

    Parameters
    ----------
    bow : list[list[str]]
        A bag of words is a 2-D list of strings, which are the tokenized,
        filtered words for each sentence in the document.

    bag_size : int, optional
        If provided, bag_size overrides the number of terms in bow.
    """
    # term occurrence = number of times a term appears within a document ("bag")
    term_occ = pd.Series([word for sent in bow for word in sent]).value_counts()
    
    # Find total number of words in the bag
    if bag_size is None:
        bag_size = term_occ.sum()

    # Remove terms with only 1 occurrence
    top_terms = term_occ.loc[term_occ > 1]

    # term frequency = occurrences over length of bag
    term_freq = top_terms / bag_size

    return(term_freq)


def make_df(tf_vectors: pd.Series[pd.Series]) -> pd.Series:
    """
    Uses all term frequency dictionaries from a collection of documents 
    to create a dictionary of document frequencies.

    Document frequency (DF) is calculated as the document occurence (number 
    of documents a term appears in) out of the total number of documents/bags.
    
    Note that make_tf() and make_gram_tf() only retain terms that appear 
    at least twice in a document, and those are the terms used in 
    calculating 

    Parameters
    ----------
    tf_dicts : list or Series of dicts or Series
        List or column of term frequency dicts, like those produced by 
        make_tf() or make_gram_tf().
    
    """
    # Convert each dictionary into a list of its keys
    bag_vocabs = pd.Series([vect.index.tolist() for vect in tf_vectors])

    # document occurrence = number of documents containing this word 
    doc_occ = pd.Series([word for bag in bag_vocabs for word in bag]).value_counts()

    # Find total number of bags in the collection
    corpus_size = len(tf_vectors)

    # document frequency = number of docs in which a term occurs
    #                      / total num of docs in collection
    doc_freq = doc_occ / corpus_size

    return(doc_freq)


def make_vocab(df: pd.Series, 
               min_df: float = 0.0, 
               max_df: float = 1.0) -> list:
    """
    Take document frequencies dictionary and filter by min/max_df.

    TF-IDF vectors will be calculated using the vocabulary that
    passes the filter.

    Parameters
    ----------
    df : Series
        Document frequencies dictionary, as produced by make_df().

    min_df : float, between 0.0 and 1.0
        Minimum allowed document frequency for inclusion in the vector vocab.

    max_df : float, between 0.0 and 1.0
        Maximum allowed document frequency for inclusion in the vector vocab.
    """
    vocab = df.loc[(min_df <= df) & (df <= max_df)].index.tolist()
    
    return(vocab)


def unit_vector(vector: pd.Series[float | int]) -> list:
    """ 
    Returns the normalized version of the vector, using 
    Euclidean normalization.
    """
    euclidean_norm = np.sqrt((vector**2).sum())

    normalized_vector = vector / euclidean_norm
    
    return(normalized_vector)


def make_tfidf(tf: pd.Series, 
               df: pd.Series, 
               vocab: list,
               norm: bool | Literal['max'] = False) -> dict:
    """
    Create document TF-IDF vector.

    Parameters
    ----------
    tf : Series
        Term frequencies dictionary, as produced by make_tf().
        This will be unique for each document.

    df : Series
        Document frequencies dictionary, as produced by make_df().
        All documents in a sample must use the same value for this parameter.

    vocab : list
        Vector vocabulary list, as produced by make_vocab().
        All documents in a sample must use the same value for this parameter.

    norm : bool or 'max', default False
        If True, normalizes all values in the vector;
        if False, the true TF-IDF scores are returned;
        if 'max', all values greater than 0 will be set to 1,
        which allows for clustering on the presence/absence of vocab terms.
    """
    # {term: tf-idf} for term in document frequencies if term is in vector vocab
    if norm == 'max':
        tfidf = (tf * df).fillna(0).astype(bool).astype(int)
    else:
        tfidf = (tf * np.log(1 + (1/df))).fillna(0)
    tfidf = tfidf.loc[tfidf.index.isin(vocab)]

    if norm is True:
        return(unit_vector(tfidf))
    else:
        return(tfidf)
