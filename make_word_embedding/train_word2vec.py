#!/bin/env python3
# coding: utf-8

import gensim
import logging
import multiprocessing
import argparse
from os import path

# This script trains a word2vec word embedding model using Gensim
# Example corpus: /fp/projects01/ec30/corpora/enwiki
# To run this on Fox, load this module:
# nlpl-gensim/4.2.0-foss-2021a-Python-3.9.5

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--corpus", help="Path to a training corpus (can be compressed)", required=True)
    arg("--cores", default=False, help="Limit on the number of cores to use")
    arg("--sg", default=0, type=int, help="Use Skipgram (1) or CBOW (0)")
    arg("--window", default=5, type=int, help="Size of context window")
    arg("--vocab", default=100000, type=int, help="Max vocabulary size")
    args = parser.parse_args()

    # Setting up logging:
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    # This will be our training corpus to infer word embeddings from.
    # Most probably, a gzipped text file, one doc/sentence per line:
    corpus = args.corpus

    # Iterator over lines of the corpus
    data = gensim.models.word2vec.LineSentence(corpus)

    # How many workers (CPU cores) to use during the training?
    if args.cores:
        # Use the number of cores we are told to use (in a SLURM file, for example):
        cores = int(args.cores)
    else:
        # Use all cores we have access to except one
        cores = (
            multiprocessing.cpu_count() - 1
        )
    logger.info(f"Number of cores to use: {cores}")

    # Setting up training hyperparameters:
    # Use Skipgram (1) or CBOW (0) algorithm?
    skipgram = args.sg
    # Context window size (e.g., 2 words to the right and to the left)
    window = args.window
    # How many words types we want to be considered (sorted by frequency)?
    vocabsize = args.vocab

    vectorsize = 300  # Dimensionality of the resulting word embeddings.

    # For how many epochs to train a model (how many passes over corpus)?
    iterations = 100


    # We have not removed stopwords from our corpora since these might contain 
    # imoortant information for our task
    model = gensim.models.Word2Vec(
        data,
        vector_size=vectorsize,
        window=window,
        workers=cores,
        sg=skipgram,
        max_final_vocab=vocabsize,
        epochs=iterations,
        sample=0.001,
    )

    # Saving the resulting model to a file
    filename = path.basename(corpus).replace(".txt", ".model")
    logger.info(filename)

    # Save the model without the output vectors (what you most probably want):
    model.wv.save(filename)

    # model.save(filename)  # If you intend to train the model further
