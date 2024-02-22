import os
import math
import pickle
from collections import defaultdict
from string_processing import (
    process_tokens,
    tokenize_text,
)


def get_query_tokens(query_string):
    """Turns a query text string into a sequence of tokens.
    Applies the same set of linguistic modules as during
    index construction.

    Args:
        query_string (str): the input query string

    Returns:
        list(str): a list of processed tokens
    """
    toks = tokenize_text(query_string)
    return process_tokens(toks)


def count_query_tokens(query_tokens):
    """Given a list of query tokens, count them and return a list
    containing (unique token, term frequency)

    Args:
        query_tokens (list(string)): a list of processed tokens

    Returns:
        list(tuple(str, int)): a list of tokens and their term frequency counts
    """
    counts = defaultdict(int)
    for tok in query_tokens:
        counts[tok] += 1
    return list(counts.items())


def get_doc_to_norm(index, doc_freq, num_docs):
    """Pre-compute the norms for each document vector in the corpus using term frequency

    Args:
        index (dict(str : list(tuple(int, int)))): The index aka dictionary of posting lists
        doc_freq (dict(str : int)): document frequency for each term
        num_docs (int): number of documents in the corpus

    Returns:
        dict(int: float): a dictionary mapping doc_ids to document norms
    """
    doc_norm = defaultdict(float)

    # calculate square of norm for all docs
    for term in index.keys():
        for (docid, tf) in index[term]:
            doc_norm[docid] += tf**2

    # take square root
    for docid in doc_norm.keys():
        doc_norm[docid] = math.sqrt(doc_norm[docid])

    return doc_norm


def run_query(query_string, index, doc_freq, doc_norm, num_docs):
    """Run a query on the index and return a sorted list of documents. 
    Sorted by most similar to least similar.
    Documents not returned in the sorted list are assumed to have 0 similarity.

    Args:
        query_string (str): the query string
        index (dict(str : list(tuple(int, int)))): the index aka dictionary of posting lists
        doc_freq (dict(str : int)): document frequency for each term
        doc_norm (dict(int : float)): a map from docid to pre-computed document norms
        num_docs (int): number of documents in the corpus

    Returns:
        list(tuple(int, float)): a list of document ids and the similarity scores with the query
            sorted so that the most similar documents to the query are at the top
    """
    # pre-process the query string
    qt = get_query_tokens(query_string)
    query_token_counts = count_query_tokens(qt)

    # calculate the norm of the query vector
    query_norm = 0
    for (term, tf) in query_token_counts:
        # ignore term if not in index (to be comparable to doc_norm)
        # note that skipping this will not change the rank of retrieved docs
        if term not in index:
            continue
        query_norm += tf**2
    query_norm = math.sqrt(query_norm)

    # calculate cosine similarity for all relevant documents
    doc_to_score = defaultdict(float)
    for (term, tf_query) in query_token_counts:
        # ignore query terms not in the index
        if term not in index:
            continue
        # add to similarity for documents that contain current query word
        for (docid, tf_doc) in index[term]:
            doc_to_score[docid] += tf_query * tf_doc / (doc_norm[docid] * query_norm)

    sorted_docs = sorted(doc_to_score.items(), key=lambda x:-x[1])
    return sorted_docs


def query_main(queries=None, query_func=None, doc_norm_func=None):
    """Run all the queries in the evaluation dataset (and the specific queries if given)
    and store the result for evaluation.

    Args:
        queries (list(str)): a list of query strings (optional)
        query_func (callable): a function to run the query, e.g., the run_query function.
        doc_norm_func (callable): a function to compute the norms for document vectors,
            e.g., the get_doc_to_norm function.
    """
    assert query_func is not None
    assert doc_norm_func is not None

    # load the index from disk
    (index, doc_freq, doc_ids, num_docs) = pickle.load(open("stored_index.pkl", "rb"))

    # compute doc norms (in practice we would want to store this on disk, for
    # simplicity in this assignment it is computed here)
    doc_norms = doc_norm_func(index, doc_freq, num_docs) 

    # get a reverse mapping from doc_ids to document paths
    ids_to_doc = {docid: path for (path, docid) in doc_ids.items()}

    # if a list of query strings are specified, run the query and output the top ranked documents
    if queries is not None and len(queries) > 0:
        for query_string in queries:
            print(f'Query: {query_string}')
            res = query_func(query_string, index, doc_freq, doc_norms, num_docs)
            print('Top-5 documents (similarity scores):')
            for (docid, sim) in res[:5]:
                print(f'{ids_to_doc[docid]} {sim:.4f}')

    # run all the queries in the evaluation dataset and store the result for evaluation
    result_strs = []
    with open(os.path.join('gov', 'topics', 'gov.topics'), 'r') as f:
        for line in f:
            # read the evaluation query
            terms = line.split()
            qid = terms[0]
            query_string = " ".join(terms[1:])

            # run the query
            res = query_func(query_string, index, doc_freq, doc_norms, num_docs)

            # write the results in the correct trec_eval format
            # see https://trec.nist.gov/data/terabyte/04/04.guidelines.html
            for rank, (docid, sim) in enumerate(res):
                result_strs.append(f"{qid} Q0 {os.path.split(ids_to_doc[docid])[-1]} {rank+1} {sim} MY_IR_SYSTEM\n")

    with open('retrieved.txt', 'w') as fout:
        for line in result_strs:
            fout.write(line)


if __name__ == '__main__':
    query_main(query_func=run_query, doc_norm_func=get_doc_to_norm)

