#!/bin/sh

# FastText
wget -P ./data/embeddings/ "https://s3-us-west-1.amazonaws.com/fasttext-vectors/crawl-300d-2M.vec.zip"
unzip ./data/embeddings/crawl-300d-2M.vec.zip ./data/embeddings/

# GloVe
wget -P ./data/embeddings/ "http://nlp.stanford.edu/data/glove.840B.300d.zip"
unzip ./data/embeddings/glove.840B.300d.zip ./data/embeddings/
