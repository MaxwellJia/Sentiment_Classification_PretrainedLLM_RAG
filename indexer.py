import os
import pickle
from itertools import groupby
from string_processing import (
    process_tokens,
    tokenize_text,
)


def read_doc(file_path):
    """Read a document from a path, tokenize, process it and return
    the list of tokens.

    Args:
        file_path (str): path to document file

    Returns:
        list(str): list of processed tokens
    """
    with open(file_path, "r", encoding='utf-8') as f:
        data = f.read()
    toks = tokenize_text(data)
    return process_tokens(toks)


def gov_list_docs(docs_path):
    """List the documents in the gov directory.

    Args:
        docs_path (str): path to the gov directory

    Returns:
        list(str): list of paths to the document 
    """
    return [os.path.join(dpath, f) for (dpath, dnames, fnames) in os.walk(docs_path) for f in fnames]


def make_doc_ids(path_list):
    """Assign unique doc_ids to documents.

    Args:
        path_list (list(str)): list of document paths 

    Returns:
        dict(str : int): dictionary of document paths to document ids
    """
    return {path: docid for (docid, path) in enumerate(path_list)}    


def get_token_list(path_list, doc_ids):
    """Read all the documents and get a list of all the tokens

    Args:
        path_list (list(str)): list of paths
        doc_ids (dict(str : int)): dictionary mapping a path to a doc_id

    Returns:
        list(tuple(str, int)): an asc sorted list of (token, doc_id) tuples
    """
    all_toks = []
    for path in path_list:
        docid = doc_ids[path]
        toks = read_doc(path)
        all_toks.extend([(tok, docid) for tok in toks])
    return sorted(all_toks)


def index_from_tokens(all_toks):
    """Construct an index from the sorted list of (token, doc_id) tuples.

    Args:
        all_toks (list(tuple(str, int))): an asc sorted list of (token, doc_id) tuples
            this is sorted first by token, then by doc_id

    Returns:
        dict(str : list(tuple(int, int))): a dictionary that maps tokens to
            list of (doc_id, term_frequency) tuples.
        dict(str : int): a dictionary that maps tokens to document frequency.
    """

    # First grouping the list of (tok, docid) by tok (i.e., one group for each unique token)
    # then grouping each group (for a particular tok) by docid (i.e., one subgroup for each unique docid)
    index = {
        tok: [(docid, len(list(sg))) for (docid, sg) in groupby(g, key=lambda y: y[1])] \
             for (tok, g) in groupby(all_toks, key=lambda x: x[0])
    }

    doc_freq = {tok: len(index[tok]) for tok in index}

    return index, doc_freq


if __name__ == '__main__':
    # get a list of documents 
    doc_list = gov_list_docs("./gov/documents")
    num_docs = len(doc_list)
    print(f"Found {num_docs} documents.")

    # assign unique doc_ids to each of the documents
    doc_ids = make_doc_ids(doc_list)

    # get the list of tokens in all the documents
    tok_list = get_token_list(doc_list, doc_ids)

    # build the index from the list of tokens
    index, doc_freq = index_from_tokens(tok_list)
    del tok_list # free some memory

    # store the index to disk
    pickle.dump((index, doc_freq, doc_ids, num_docs), open("stored_index.pkl", "wb"))

