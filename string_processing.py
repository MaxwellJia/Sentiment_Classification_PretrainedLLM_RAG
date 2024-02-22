import nltk


# get the nltk stopwords list
stopwords = set(nltk.corpus.stopwords.words("english"))

def process_tokens_original(toks):
    """ Perform processing on tokens. This is the Linguistics Modules
    phase of index construction

    Args:
        toks (list(str)): all the tokens in a single document

    Returns:
        list(str): tokens after processing
    """
    new_toks = []
    for t in toks:
        # ignore stopwords
        if t in stopwords or t.lower() in stopwords:
            continue
        # lowercase token
        t = t.lower()
        new_toks.append(t)
    return new_toks


def process_tokens(toks):
    return process_tokens_original(toks)


def tokenize_text(data):
    """Convert a document as a string into a document as a list of
    tokens. The tokens are strings.

    Args:
        data (str): The input document

    Returns:
        list(str): The list of tokens.
    """
    # split text on spaces
    tokens = data.split()
    return tokens

