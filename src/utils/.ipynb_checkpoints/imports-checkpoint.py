# misc python stuff
import os
import re
import string
from string import digits
import configparser
from slackclient import SlackClient  # slack messaging when model is finished training
import pathlib                       # detecting if file exists in a path
import pickle                        # saving and loading keras tokenizer

# graphics libraries
import matplotlib.pyplot as plt

# numerical libraries
import numpy as np
import pandas as pd
from scipy import sparse

# keras model building
import keras
from keras.models import Model, Sequential
from keras.layers import (Input, Dense, Embedding, SpatialDropout1D, concatenate,  RepeatVector, Flatten, Conv1D,
        GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, CuDNNGRU, CuDNNLSTM, MaxPooling1D, Layer,
        Dropout, K, Activation, BatchNormalization, PReLU, add, Reshape)
from keras.preprocessing import text, sequence
from keras import optimizers
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

# sklearn feature/model building
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

# embedding similar to t-SNE but with custom metrics e.g. cosine distance
import umap

# NLP library
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords

# parallel processing stuff
from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import Pool
