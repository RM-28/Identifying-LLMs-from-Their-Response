'''
helper file for tokenizing text for CNN model
from Chris Tran's code
'''

from nltk.tokenize import word_tokenize
from collections import defaultdict
import numpy as np

def tokenize(texts):
    """Tokenize texts, build vocabulary and find maximum sentence length.
        SPLITS THE SENTENCE INTO TOKENS, NOT INTO NUMBERS
    Args:
        texts (List[str]): List of text data
    
    Returns:
        tokenized_texts (List[List[str]]): List of list of tokens
        word2idx (Dict): Vocabulary built from the corpus
        max_len (int): Maximum sentence length
    """

    max_len = 0
    tokenized_texts = []
    word2idx = {}

    # Add <pad> and <unk> tokens to the vocabulary
    word2idx['<pad>'] = 0
    word2idx['<unk>'] = 1

    # Building our vocab from the corpus starting from index 2
    idx = 2
    for sentence in texts:
        tokenized_sent = word_tokenize(sentence) # wonder how this handles special chars
        # list looks like ["Do", "n't", "be", "bad"]

        # Add `tokenized_sent` to `tokenized_texts`
        tokenized_texts.append(tokenized_sent)

        # Add new token to `word2idx`, updating vocabulary 
        for token in tokenized_sent:
            if token not in word2idx:
                word2idx[token] = idx
                idx += 1

        # Update `max_len`
        max_len = max(max_len, len(tokenized_sent))

    return tokenized_texts, word2idx, max_len

def encode(tokenized_texts, word2idx, max_len):
    """Pad each sentence to the maximum sentence length (longest prompt) and encode tokens to
    their index in the vocabulary.

    Returns:
        input_ids (np.array): Array of token indexes in the vocabulary with
            shape (N, max_len). It will the input of our CNN model.
    """

    input_ids = []
    for tokenized_sent in tokenized_texts:
        # Pad sentences to max_len
        tokenized_sent += ['<pad>'] * (max_len - len(tokenized_sent))

        # Encode tokens to input_ids
        input_id = [word2idx.get(token) for token in tokenized_sent]
        # so we have the token-id equivalent, [4, 7, 2, 9, 0, 0, 0, etc]

        input_ids.append(input_id)
    
    return np.array(input_ids)