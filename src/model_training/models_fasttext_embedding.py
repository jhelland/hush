
# coding: utf-8

# ## Train various models using FastTest embeddings
# 
# **reference:** https://github.com/sermakarevich/jigsaw-toxic-comment-classification-challenge/tree/master/bin
# 
# **NOTES:**
# - SpatialDropout1D: same as Dropout but drops entire vector features instead of individual entries
# - keras **embedding layer** Embedding($n,m,k$): trainable *(automatic differentiation is magic)*, $n$ distinct words in dictionary, embedding $m$-vectors, max phrase length $k$
#     - Works by assigning distinct integer indices to unique words appearing in phrases. Then those integers are then referenced in the lookup table of $m$-vectors to construct a corresponding numerical sequence
#     - We can use pretrained weights (e.g. word2vec, fasttext, glove) for the lookup table and then freeze the layer via the $\verb|trainable=False|$ flag to prevent training

# In[1]:


import sys
sys.path.append("../utils/")

from imports import *

from utilities import LossHistory, RocAucEvaluation
from models import get_model_biGRU


# In[2]:


# load data
#   n_keep_train : number of training samples to keep. Set to -1 to keep all

EMBEDDING_FILE = "../data/toxic_comments/features/crawl-300d-2M.vec"
n_keep_train = -1

categories = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
train = pd.read_csv("../data/toxic_comments/train.csv")
X_train = train["comment_text"].fillna("fillna").values[:n_keep_train]

# collapse all categories into toxic vs. non-toxic
y_train = train[categories].iloc[:,:].sum(axis=1).astype(int)
y_train[y_train > 0] = 1
y_train = y_train.values[:n_keep_train]
y_train = np.vstack([y_train, 1-y_train]).T


# In[3]:


get_ipython().run_cell_magic('time', '', '## tokenize\n\nmax_features = 30000\nmaxlen = 100\nembed_size = 300\n\ntokenizer_path = "../data/models/tokenizer_raw_binary.pkl"\nif pathlib.Path(tokenizer_path).is_file():\n    print("...loading tokenizer from {:s}".format(tokenizer_path))\n    with open(tokenizer_path, "rb") as handle:\n        tokenizer = pickle.load(handle)\nelse:\n    print("...training tokenizer")\n    tokenizer = text.Tokenizer(num_words=max_features)\n    tokenizer.fit_on_texts(list(X_train))\n    with open(tokenizer_path, "wb") as handle:\n        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)\n    print("...tokenizer saved to {:s}".format(tokenizer_path))\n    \nX_train = tokenizer.texts_to_sequences(X_train)\nx_train = sequence.pad_sequences(X_train, maxlen=maxlen)')


# In[4]:


get_ipython().run_cell_magic('time', '', '## Create embedding matrix using fasttext embedding.\n#       Uses max_features most common words in dataset.\n#       Use precomputed matrix when possible b/c it takes a WHILE otherwise.\nembedding_matrix_dir = "../data/models/features/fasttext_embedding_matrix_" + str(max_features) + ".npy"\ncreate_embed_flag = True\nif pathlib.Path(embedding_matrix_dir).is_file():\n    embedding_matrix = np.load(embedding_matrix_dir)\n    print("embedding matrix loaded from " + embedding_matrix_dir)\n    if embedding_matrix.shape == (max_features, embed_size):\n        create_embed_flag = False\n    else:\n        print("\\tERROR: embedding_matrix.shape != {}\\n\\tcreating new embedding_matrix...".format((max_features, embed_size)))\n\nif create_embed_flag:\n    def get_coefs(word, *arr):\n        return word, np.asarray(arr, dtype="float32")\n\n    embeddings_index = dict(get_coefs(*o.rstrip().rsplit(\' \')) for o in open(EMBEDDING_FILE))\n    word_index = tokenizer.word_index\n    nb_words = min(max_features, len(word_index))\n    embedding_matrix = np.zeros((nb_words, embed_size))\n    for word, i in word_index.items():\n        if i >= max_features:\n            continue\n        embedding_vector = embeddings_index.get(word)\n        if embedding_vector is not None:\n            embedding_matrix[i] = embedding_vector\n    np.save(embedding_matrix_dir, embedding_matrix)\n    print("embeddings matrix saved to " + embedding_matrix_dir)\n    ')


# In[12]:


model = get_model_biGRU(maxlen,
                        max_features,
                        embed_size,
                        embedding_matrix,
                        outp_dim=2,
                        activation="softmax")
checkpointer = ModelCheckpoint(filepath="../data/models/checkpoints/gru_raw_binary_{epoch:02d}-{val_loss:.2f}.h5", verbose=1, save_best_only=True)

batch_size = 32
epochs = 6

X_trn, X_val, y_trn, y_val = train_test_split(x_train, y_train, train_size=0.90)
RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)
loss_history = LossHistory()

init_scores = model.evaluate(X_val, y_val)  # untrained model validation [loss, accuracy]

hist = model.fit(X_trn, y_trn, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), callbacks=[loss_history, RocAuc, checkpointer], verbose=2)


# In[11]:


# load weights from best checkpoint
model.load_weights("../data/toxic_comments/models/checkpoints/gru_raw_binary_03-0.12.h5")
# move weights from best to official model file
# os.rename(checkpointer.filepath, "../data/toxic_comments/models/bi_gru_raw_binary.h5")
print("Weights loaded from best epoch with loss {}\n".format(checkpointer.best))


# In[12]:


# model scoring
preds = model.predict(X_val, batch_size=batch_size)
print(classification_report(y_val, preds.round()))


# In[13]:


plt.figure(figsize=(10,6))

# plot dividing lines between each epoch
epoch_idx = X_trn.shape[0] // batch_size - 1
for i in range(1,epochs+1,1):
    if i == 1:
        plt.plot([epoch_idx*i, epoch_idx*i], [0., max(loss_history.losses)], "k--", linewidth=1.5, label="epoch")
    else: 
        plt.plot([epoch_idx*i, epoch_idx*i], [0., max(loss_history.losses)], "k--", linewidth=1.5)

# plot losses
plt.plot(loss_history.losses, linewidth=1.5, label="training loss")
plt.plot([i*epoch_idx for i in range(epochs+1)], [init_scores[1]] + hist.history["val_loss"], "-o", linewidth=2, label="validation loss")
plt.legend(loc="upper right")
plt.title("bi-GRU training")
plt.xlabel("batch number")
plt.ylabel("loss")
plt.grid()


# In[14]:


# save bi-GRU model
model.save("../data/toxic_comments/models/bi_gru_raw_binary.h5")

