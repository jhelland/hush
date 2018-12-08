"""
Script to download trained Keras models from Dropbox.
"""


import urllib.request
import os

if not os.path.exists("./data/models/"):
    os.makedirs("./data/models")

urls = ["https://www.dropbox.com/s/k4i5xjoxc20suo4/bi_gru_raw_binary_fasttext.h5?dl=1", # fasttext
        "https://www.dropbox.com/s/it9bkjt4nbcktxr/bi_gru_raw_binary_glove.h5?dl=1", # glove
        "https://www.dropbox.com/s/6le95jv70j30qwd/tokenizer_raw_binary.pkl?dl=1"] # tokenizer
files = ["./data/models/bi_gru_raw_binary_fasttext.h5",
         "./data/models/bi_gru_raw_binary_glove.h5",
         "./data/models/tokenizer_raw_binary.pkl"]


print("...Downloading models")
for url, fname in zip(urls, files):
    u = urllib.request.urlopen(url)
    data = u.read()
    u.close()

    with open(fname, "wb") as f:
        f.write(data)
    print("\twrote {:s}".format(fname))
print("...Models downloaded to \"./data/models/\"")
print("DONE\n")

