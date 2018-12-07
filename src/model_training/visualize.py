
# coding: utf-8

# In[84]:


# -*- coding: utf-8 -*-

import sys
sys.path.append("../utils/")

from imports import *

import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from nltk.tokenize import TweetTokenizer
import scipy.stats as ss
from itertools import product
import matplotlib
from sortedcontainers import SortedList
from PIL import Image
from collections import OrderedDict
import gensim
import matplotlib_venn as venn


# In[85]:


# initialize stuff and load data
font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : 12}
matplotlib.rc('font', **font)

color = sns.color_palette()
sns.set_style("white")
eng_stopwords = set(stopwords.words("english"))

lem = WordNetLemmatizer()
tokenizer = TweetTokenizer()

train = pd.read_csv("../data/toxic_comments/train.csv")
test = pd.read_csv("../data/toxic_comments/test.csv")

train["comment_text"].fillna("fillna", inplace=True)
test["comment_text"].fillna("fillna", inplace=True)

label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
train["clean"] = 1 - train[label_cols].max(axis=1)


# In[4]:


train.head()


# In[3]:


print("number of total comments: train {} | test {}".format(len(train), len(test)))


# In[87]:


# histogram of comment categories
kwargs = {"weight": "bold", "fontsize": 15}

x = train.iloc[:,2:].sum()
percentages = x / x.sum()
plt.figure(figsize=(20,7))
plt.subplot(1,2,1)
ax = sns.barplot(x.index, x.values, alpha=0.8)
plt.title("# per class", **kwargs)
plt.ylabel("# of occurences", **kwargs)
plt.xlabel("comment category", **kwargs)

rects = ax.patches
labels = x.values
for rect, perc in zip(rects, percentages):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height+5, "{:.2f}%".format(100*perc), ha="center", va="bottom")


# histogram of number of labels excluding clean category
plt.subplot(1,2,2)
rowsums = train.iloc[:,2:].sum(axis=1)
rowsums = rowsums[rowsums > 0]
x = rowsums.value_counts()
percentages = x / rowsums.sum()
ax = sns.barplot(x.index, x.values, alpha=0.8, color=color[2])
plt.title("multiple categories per comment", **kwargs)
plt.ylabel("# of occurences", **kwargs)
plt.xlabel("# of categories", **kwargs)

rects = ax.patches
for rect, perc in zip(rects, percentages):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height+5, "{:.2f}%".format(100*perc), ha="center", va="bottom")
plt.show()


# It is important to note that almost all of the comments are not classified in any of the unsavory categories. This is something to keep in mind while training as it could lead to stupid behavior such as a model classifying everything as clean.
# 
# We can also see that the largest number of unsavory comments only have one label.

# In[172]:


# multi-label comments
rowsums = train.iloc[:,2:].sum(axis=1)
x = train[rowsums > 1].iloc[:,2:].sum()
percentages = 100 * x / train.iloc[:,2:].sum()
plt.figure(figsize=(20,15))
plt.subplot(2,2,1)
ax = sns.barplot(percentages.index, percentages.values, alpha=0.8)
plt.title("% per class with multiple labels", **kwargs)
plt.ylabel("% within category", **kwargs)
plt.xlabel("comment type", **kwargs)

rects = ax.patches
labels = x.values
for rect, perc in zip(rects, percentages):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height, "{:.2f}%".format(perc), ha="center", va="bottom")
    

# # single label comments
# x = train[rowsums == 1].iloc[:,2:-1].sum()
# percentages = 100 * x / train.iloc[:,2:].sum()
# plt.subplot(2,2,2)
# ax = sns.barplot(percentages.index, percentages.values, alpha=0.8)
# plt.title("% per class with one label", **kwargs)
# plt.ylabel("% of occurences", **kwargs)
# plt.xlabel("comment type", **kwargs)

# rects = ax.patches
# labels = x.values
# for rect, perc in zip(rects, percentages):
#     height = rect.get_height()
#     ax.text(rect.get_x() + rect.get_width()/2, height+5, "{:.2f}%".format(perc), ha="center", va="bottom")
    
    
# labeled comments that include "toxic"
x = train[(rowsums >= 1) & (train.iloc[:,2] == 1)].iloc[:,2:].sum()
percentages = 100 * x / train.iloc[:,2:].sum()
plt.subplot(2,2,2)
ax = sns.barplot(percentages.index, percentages.values, alpha=0.8)
plt.title("% labeled comments that are also \"toxic\"", **kwargs)
plt.ylabel("% within category", **kwargs)
plt.xlabel("comment type", **kwargs)

rects = ax.patches
labels = x.values
for rect, perc in zip(rects, percentages):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height, "{:.2f}%".format(perc), ha="center", va="bottom")


# Most comments within each category are multi-label. Almost all unsavory comments are labeled "toxic" (every "severe_toxic" comment is also "toxic").

# In[10]:


# confusion matrices for toxic
temp_df = train.iloc[:,2:]
main_col = "toxic"
corr_mats = []
for other_col in temp_df.columns[1:]:
    confusion_matrix = pd.crosstab(temp_df[main_col], temp_df[other_col])
    corr_mats.append(confusion_matrix)
pd.concat(corr_mats, axis=1, keys=temp_df.columns[1:])


# Some observations:
# - Every "severe_toxic" comment is also labeled as "toxic".
# - In fact, most unsavory comments belong to the toxic category.

# In[11]:


# cramer's corrected correlations between categories
def cramers_corrected_stat(confusion_matrix):
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

corrs = {}
max_pair = ''
max_corr = 0
for col1, col2 in product(label_cols, repeat=2):
    if col1 == col2:
        continue
    confusion_matrix = pd.crosstab(train[col1], train[col2])
    corr = cramers_corrected_stat(confusion_matrix)
    pair = repr(sorted([col1, col2]))
    corrs[pair] = corr
    if corr > max_corr:
        max_corr = corr
        max_pair = pair

# display
for key in sorted(corrs, key=(lambda key: corrs[key]), reverse=True):
    print("{:<35s} {:^5s} {:>5.4f}".format(key, '|', corrs[key]))

# compute average correlation for each category i.e. measure each category's significance in the dataset
category_corrs_avg = {category: 0 for category in label_cols}
for category in label_cols:
    for key in corrs:
        if category in eval(key):
            category_corrs_avg[category] += corrs[key] / len(label_cols)

# display
print()
for key in sorted(category_corrs_avg, key=(lambda key: category_corrs_avg[key]), reverse=True):
    print("{:<15s} {:^5s} {:>5.4f}".format(key, '|', category_corrs_avg[key]))


# **Observations:**
# - "threat" is the category that is the least related to the others. This could indicate that "threat" comments have relatively unique features and perhaps that a model that is able to strongly classify, say, "identity_hate" comments will generalize poorly to "threat" comments.
# - "toxic" is surprisingly not the most correlated category -- "insult" has the highest average correlation with the others.
# - An "insult" comment is likely to be "obscene" -- people on Twitter evidently are profane when insulting others.
# - It makes sense that "obscene" is a highly correlated category -- people on the internet like to say "fuck" (incidentally the most negative polarity word according to our sentiment analysis)

# In[83]:


# random clean example
idx = np.random.randint(0,train["clean"].sum()-1)
comment = train[train["clean"]==1].iloc[idx,1]
print("---------------------------------------")
print(comment)
print("\n\n")


# In[67]:


# random example comments from each category
x = train.iloc[:,2:].sum()
for category in label_cols:
    idx = np.random.randint(0,x[category].sum()-1)
    comment = train[train[category]==1].iloc[idx,1]
    lbls = train[train[category]==1].iloc[idx,2:].values
    print(category, ":\t", [label_cols[i] for i in range(len(lbls)) if lbls[i] == 1])
    print("---------------------------------------")
    print(comment)
    print("\n\n")


# Comments often contain misspellings, repeated characters e.g. "fuuuck", contractions e.g. "don't", IP addresses and usernames e.g. "72.76.10.207" "Renata3".

# In[125]:


# some comments have excessive numbers of special symbols -- this could lead to 
very_excited = test["comment_text"].str.count('!').sort_values(ascending=False)
vals = very_excited.values
print(test["comment_text"].loc[very_excited == vals[0]].iloc[0])


# Some comments also have excessive numbers of special symbols. This kind of comment could lead to overfitting in the training process.

# In[31]:


# unique symbols
corpus = ''
for comment in pd.concat([train["comment_text"], test["comment_text"]]).values:
    corpus += comment.replace(' ', '')  # strip spaces

symbols = {}
for c in corpus:
    if c in symbols: 
        symbols[c] += 1
    else:
        symbols[c] = 1
symbols_list = np.array( [(k, symbols[k]) for k in sorted(symbols, key=symbols.get, reverse=True)] )


# In[32]:


n_symbols = 40

sns.set_style("whitegrid")

# top 40
plt.figure(figsize=(15,15))
plt.barh(symbols_list[:n_symbols,0], symbols_list[:n_symbols,1].astype(int) / len(corpus), color=sns.color_palette("GnBu_d", n_symbols)[::-1])
plt.gca().invert_yaxis()
sns.despine(left=True, bottom=True)
plt.xlabel("Percentage of corpus", **kwargs)
plt.ylabel("Top {} symbols".format(n_symbols), **kwargs)
plt.yticks(fontsize=15)

print("Bottom {} symbols:\t".format(n_symbols), symbols_list[-n_symbols:,0])


# # Leaky features
# We know that there are features present in the comments that will lead to overfitting.

# In[9]:


merge = pd.concat([train.iloc[:,:2], test.iloc[:,:2]])
df = merge.reset_index(drop=True)


# In[10]:


# count number of sentences per comment using '\n' as delimiter
df["count_sent"] = df["comment_text"].apply(lambda x: len(re.findall('\n', str(x)))+1)

# word count per comment
df["count_word"] = df["comment_text"].apply(lambda x: len(str(x).split()))

# unique word count
df["count_unique_word"] = df["comment_text"].apply(lambda x: len(set(str(x).split())))

# letter count
df["count_letters"] = df["comment_text"].apply(lambda x: len(str(x)))

# punctuation count
df["count_punctuations"] = df["comment_text"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

# upper case words count
df["count_words_upper"] = df["comment_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

# title case words count
df["count_words_title"] = df["comment_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

# average length of the words
df["mean_word_len"] = df["comment_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))


# In[11]:


df["word_unique_percent"] = df["count_unique_word"]*100 / df["count_word"]

df["punct_percent"] = df["count_punctuations"]*100 / df["count_word"]


# In[12]:


# separate train/test features
train_features = df.iloc[:len(train),]
test_features = df.iloc[len(train):,]

# join the tags
train_tags = train.iloc[:,2:]
train_features = pd.concat([train_features, train_tags], axis=1)


# In[14]:


# are longer comments more toxic?
train_features["count_sent"].loc[train_features["count_sent"] > 10] = 10

plt.figure(figsize=(20,10))
# sentences
plt.subplot(1,2,1)
plt.suptitle("Are longer comments more toxic?", **kwargs)
sns.violinplot(y="count_sent", x="clean", data=train_features, split=True)
plt.xlabel("Clean?", **kwargs)
plt.ylabel("# of sentences", **kwargs)
plt.title("Number of sentences in each comment", **kwargs)

# words
train_features["count_word"].loc[train_features["count_word"] > 200] = 200
plt.subplot(1,2,2)
sns.violinplot(y="count_word", x="clean", data=train_features, split=True, inner="quart")
plt.xlabel("Clean?", **kwargs)
plt.ylabel("# of words", **kwargs)
plt.title("Number of words in each comment", **kwargs)

plt.show()


# There does not appear to be a statistically significant difference between long and short comments at both the sentence and word level.
# 
# **Note:**
# - a violin plot is an alternative to a box plot with kernel density estimates. Inner markings show percentiles and width shows volume of comments at that level
# 
# ***
# 
# Now, we can analyze uniqueness of words and identify spammers based on low uniqueness percentages.

# In[174]:


train_features["count_unique_word"].loc[train_features["count_unique_word"] > 200] = 200
temp_df = pd.melt(train_features, value_vars=["count_word", "count_unique_word"], id_vars="clean")
spammers = train_features[train_features["word_unique_percent"] < 30]


# In[178]:


from matplotlib.pyplot import GridSpec

plt.figure(figsize=(20,15))
plt.suptitle("What's so unique?", **kwargs)
GridSpec(2,2)
plt.subplot2grid((2,2), (0,0))
sns.violinplot(x="variable", y="value", hue="clean", data=temp_df, split=True, inner="quartile")
plt.title("Absolute word count and unique words count", **kwargs)
plt.xlabel("Feature", **kwargs)
plt.ylabel("Count", **kwargs)

plt.subplot2grid((2,2), (0,1))
plt.title("Percentage of unique words of total words in comment", **kwargs)
ax = sns.kdeplot(train_features[train_features["clean"] == 0]["word_unique_percent"], label="Bad", shade=True, color='r')
ax = sns.kdeplot(train_features[train_features["clean"] == 1]["word_unique_percent"], label="Clean", color='b')
plt.legend()
plt.ylabel("Number of occurences", **kwargs)
plt.xlabel("Percent unique words", **kwargs)

x = spammers.iloc[:,-7:].sum()
percentages = x / train.iloc[:,2:].sum()
plt.subplot2grid((2,2), (1,0), colspan=2)
plt.title("Count of comments with low (<30%) unique words", **kwargs)
ax = sns.barplot(x=x.index, y=x.values, color=sns.color_palette()[3], alpha=0.8)

# adding text labels
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height+5, label, ha="center", va="bottom")
    
plt.xlabel("Threat class", **kwargs)
plt.ylabel("# of comments", **kwargs)
plt.show()


# **Note:**
# - The bumps at 200 are because we set all comments of longer length to 200
# - There is a bump in unsavory comment unique words percentage below the 10% threshold -- this indicates that there are a sizable amount of toxic comments with almost no word variety.
# 
# ***
# 
# # Spammers are more toxic
# Who could have predicted this?

# In[18]:


# low word variety random comment example
print("Unsavory example:\n--------------------------------------\n")
example_unsavory = spammers[spammers["clean"] == 0]["comment_text"]  #train[(train_features["word_unique_percent"] <= 10) & (train["clean"] == 0)]["comment_text"]
print(example_unsavory.iloc[np.random.randint(0, len(example_unsavory))])
print("\n\n")
print("Clean example:\n--------------------------------------\n")
example_clean = spammers[spammers["clean"] == 1]["comment_text"] #train[(train_features["word_unique_percent"] <= 10) & (train["clean"] == 1)]["comment_text"]
print(example_clean.iloc[np.random.randint(0, len(example_clean))])


# # Does spam cause overfitting?
# Spam could lead to model overfitting. It could be pragmatic to remove excessive reptitions in the final dataset.
# 
# ***
# 
# # Leaky features
# These probably shouldn't be used in the final model(s)

# In[100]:


# IP address count
df["ip"] = df["comment_text"].apply(lambda x: re.findall("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", str(x)))
df["count_ip"] = df["ip"].apply(lambda x: len(x))

# URLs
df["link"] = df["comment_text"].apply(lambda x: re.findall("^(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?$", str(x)))
df["count_links"] = df["link"].apply(lambda x: len(x))

# article IDs
df["article_id"] = df["comment_text"].apply(lambda x: re.findall("\d:\d\d\s{0,5}$", str(x)))
df["article_id_flag"] = df["article_id"].apply(lambda x: len(x))

# username
## regex for matching anything with [[User: -------]]
df["username"] = df["comment_text"].apply(lambda x: re.findall("\[\[User(.*)\|", str(x)))
df["count_usernames"] = df["username"].apply(lambda x: len(x))

# leaky IP
cv = CountVectorizer()
count_features_ip = cv.fit_transform(df["ip"].apply(lambda x: str(x)))

# leaky usernames
cv = CountVectorizer()
count_features_ip = cv.fit_transform(df["username"].apply(lambda x: str(x)))


# In[86]:


df[df["count_usernames"] != 0]["comment_text"].iloc[0]


# In[64]:


# check a few names
cv.get_feature_names()[120:130]


# # Leaky feature stability
# Check reoccurance of leaky features for the sake of determining their utility in test set predictions 

# In[66]:


leaky_features = df[["ip", "link", "article_id", "username", "count_ip", "count_links", "count_usernames", "article_id_flag"]]
leaky_features_train = leaky_features.iloc[:len(train)]
leaky_features_test = leaky_features.iloc[len(train):]


# In[79]:


# filter out the entries without IPs
train_ips = leaky_features_train["ip"][leaky_features_train["count_ip"] > 0]
test_ips = leaky_features_test["ip"][leaky_features_test["count_ip"] > 0]

# get the unique list of IPs in test and train datasets
train_ip_list = list(set([a for b in train_ips.tolist() for a in b]))
test_ip_list = list(set([a for b in test_ips.tolist() for a in b]))

# get common elements
common_ip_list = list(set(train_ip_list).intersection(test_ip_list))

plt.figure(figsize=(7,7))
plt.title("Common IP addresses", **kwargs)
venn.venn2(subsets=(len(train_ip_list), len(test_ip_list), len(common_ip_list)), set_labels=("# of unique IP in train", "# of unique IP in test"))
plt.show()


# In[101]:


# filter out entries without links
train_links = leaky_features_train["link"][leaky_features_train["count_links"] > 0]
test_links = leaky_features_test["link"][leaky_features_test["count_links"] > 0]

# get unique list of links in test and train datasets
train_links_list = list(set([a for b in train_links.tolist() for a in b]))
test_links_list = list(set([a for b in test_links.tolist() for a in b]))

# get common links
common_links_list = list(set(train_links_list).intersection(test_links_list))

plt.figure(figsize=(7,7))
plt.title("Common links", **kwargs)
venn.venn2(subsets=(len(train_links_list), len(test_links_list), len(common_links_list)), set_labels=("# fo unique links in train", "# of unique links in test"))
plt.show()


# In[83]:


# filter out entries without usernames
# filter out entries without links
train_users = leaky_features_train["username"][leaky_features_train["count_usernames"] > 0]
test_users = leaky_features_test["username"][leaky_features_test["count_usernames"] > 0]

# get unique list of links in test and train datasets
train_users_list = list(set([a for b in train_users.tolist() for a in b]))
test_users_list = list(set([a for b in test_users.tolist() for a in b]))

# get common links
common_users_list = list(set(train_users_list).intersection(test_users_list))

plt.figure(figsize=(7,7))
plt.title("Common usernames", **kwargs)
venn.venn2(subsets=(len(train_users_list), len(test_users_list), len(common_users_list)), set_labels=("# fo unique usernames in train", "# of unique usernames in test"))
plt.show()


# 
# It doesn't make sense to use, say, IP addresses that are not common to both the training and test sets.
# 
# We could use the lists of IPs, usernames, URLs in reference a block lists. For example, wikipedia has a database of [indefinitely blocked IPs]( https://en.wikipedia.org/wiki/Wikipedia:Database_reports/Indefinitely_blocked_IPs). We could also build up a table of known toxic users that can be blocked in the future (same for URLs).

# In[103]:


#https://en.wikipedia.org/wiki/Wikipedia:Database_reports/Indefinitely_blocked_IPs)

blocked_ips=["216.102.6.176",
"216.120.176.2",
"203.25.150.5",
"203.217.8.30",
"66.90.101.58",
"125.178.86.75",
"210.15.217.194",
"69.36.166.207",
"213.25.24.253",
"24.60.181.235",
"71.204.14.32",
"216.91.92.18",
"212.219.2.4",
"194.74.190.162",
"64.15.152.246",
"59.100.76.166",
"146.145.221.129",
"146.145.221.130",
"74.52.44.34",
"68.5.96.201",
"65.184.176.45",
"209.244.43.209",
"82.46.9.168",
"209.200.236.32",
"209.200.229.181",
"202.181.99.22",
"220.233.226.170",
"212.138.64.178",
"220.233.227.249",
"72.14.194.31",
"72.249.45.0/24",
"72.249.44.0/24",
"80.175.39.213",
"81.109.164.45",
"64.157.15.0/24",
"208.101.10.54",
"216.157.200.254",
"72.14.192.14",
"204.122.16.13",
"217.156.39.245",
"210.11.188.16",
"210.11.188.17",
"210.11.188.18",
"210.11.188.19",
"210.11.188.20",
"64.34.27.153",
"209.68.139.150",
"152.163.100.0/24",
"65.175.48.2",
"131.137.245.197",
"131.137.245.199",
"131.137.245.200",
"64.233.172.37",
"66.99.182.25",
"67.43.21.12",
"66.249.85.85",
"65.175.134.11",
"201.218.3.198",
"193.213.85.12",
"131.137.245.198",
"83.138.189.74",
"72.14.193.163",
"66.249.84.69",
"209.204.71.2",
"80.217.153.189",
"83.138.136.92",
"83.138.136.91",
"83.138.189.75",
"83.138.189.76",
"212.100.250.226",
"212.100.250.225",
"212.159.98.189",
"87.242.116.201",
"74.53.243.18",
"213.219.59.96/27",
"212.219.82.37",
"203.38.149.226",
"66.90.104.22",
"125.16.137.130",
"66.98.128.0/17",
"217.33.236.2",
"24.24.200.113",
"152.22.0.254",
"59.145.89.17",
"71.127.224.0/20",
"65.31.98.71",
"67.53.130.69",
"204.130.130.0/24",
"72.14.193.164",
"65.197.143.214",
"202.60.95.235",
"69.39.89.95",
"88.80.215.14",
"216.218.214.2",
"81.105.175.201",
"203.108.239.12",
"74.220.207.168",
"206.253.55.206",
"206.253.55.207",
"206.253.55.208",
"206.253.55.209",
"206.253.55.210",
"66.64.56.194",
"70.91.90.226",
"209.60.205.96",
"202.173.191.210",
"169.241.10.83",
"91.121.195.205",
"216.70.136.88",
"72.228.151.208",
"66.197.167.120",
"212.219.232.81",
"208.86.225.40",
"63.232.20.2",
"206.219.189.8",
"212.219.14.0/24",
"165.228.71.6",
"99.230.151.129",
"72.91.11.99",
"173.162.177.53",
"60.242.166.182",
"212.219.177.34",
"12.104.27.5",
"85.17.92.13",
"91.198.174.192/27",
"155.246.98.61",
"71.244.123.63",
"81.144.152.130",
"198.135.70.1",
"71.255.126.146",
"74.180.82.59",
"206.158.2.80",
"64.251.53.34",
"24.29.92.238",
"76.254.235.105",
"68.96.242.239",
"203.202.234.226",
"173.72.89.88",
"87.82.229.195",
"68.153.245.37",
"216.240.128.0/19",
"72.46.129.44",
"66.91.35.165",
"82.71.49.124",
"69.132.171.231",
"75.145.183.129",
"194.80.20.237",
"98.207.253.170",
"76.16.222.162",
"66.30.100.130",
"96.22.29.23",
"76.168.140.158",
"202.131.166.252",
"89.207.212.99",
"81.169.155.246",
"216.56.8.66",
"206.15.235.10",
"115.113.95.20",
"204.209.59.11",
"27.33.141.67",
"41.4.65.162",
"99.6.65.6",
"60.234.239.169",
"2620:0:862:101:0:0:2:0/124",
"183.192.165.31",
"50.68.6.12",
"37.214.82.134",
"96.50.0.230",
"60.231.28.109",
"64.90.240.50",
"49.176.97.12",
"209.80.150.137",
"24.22.67.116",
"206.180.81.2",
"195.194.39.100",
"87.41.52.6",
"169.204.164.227",
"50.137.55.117",
"50.77.84.161",
"90.202.230.247",
"186.88.129.224",
"2A02:EC80:101:0:0:0:2:0/124",
"142.4.117.177",
"86.40.105.198",
"120.43.20.149",
"198.199.64.0/18",
"192.34.56.0/21",
"192.81.208.0/20",
"2604:A880:0:0:0:0:0:0/32",
"108.72.107.229",
"2602:306:CC2B:7000:41D3:B92D:731C:959D",
"185.15.59.201",
"180.149.1.229",
"207.191.188.66",
"210.22.63.92",
"117.253.196.217",
"119.160.119.172",
"90.217.133.223",
"194.83.8.3",
"194.83.164.22",
"217.23.228.149",
"65.18.58.1",
"168.11.15.2",
"65.182.127.31",
"207.106.153.252",
"64.193.88.2",
"152.26.71.2",
"199.185.67.179",
"117.90.240.73",
"108.176.58.170",
"195.54.40.28",
"185.35.164.109",
"192.185.0.0/16",
"2605:E000:1605:C0C0:3D3D:A148:3039:71F1",
"107.158.0.0/16",
"85.159.232.0/21",
"69.235.4.10",
"86.176.166.206",
"108.65.152.51",
"10.4.1.0/24",
"103.27.227.139",
"188.55.31.191",
"188.53.13.34",
"176.45.58.252",
"176.45.22.37",
"24.251.44.140",
"108.200.140.191",
"117.177.169.4",
"72.22.162.38",
"24.106.242.82",
"79.125.190.93",
"107.178.200.1",
"123.16.244.246",
"83.228.167.87",
"128.178.197.53",
"14.139.172.18",
"207.108.136.254",
"184.152.17.217",
"186.94.29.73",
"217.200.199.2",
"66.58.141.104",
"166.182.81.30",
"89.168.206.116",
"92.98.163.145",
"77.115.31.71",
"178.36.118.74",
"157.159.10.14",
"103.5.212.139",
"203.174.180.226",
"69.123.252.95",
"199.200.123.233",
"121.45.89.82",
"71.228.87.155",
"68.189.67.92",
"216.161.176.152",
"98.17.30.139",
"2600:1006:B124:84BD:0:0:0:103",
"117.161.0.0/16",
"12.166.68.34",
"96.243.149.64",
"74.143.90.218",
"76.10.176.221",
"104.250.128.0/19",
"185.22.183.128/25",
"89.105.194.64/26",
"202.45.119.0/24",
"73.9.140.64",
"164.127.71.72",
"50.160.129.2",
"49.15.213.207",
"83.7.192.0/18",
"201.174.63.79",
"2A02:C7D:4643:8F00:D09D:BE1:D2DE:BB1F",
"125.60.195.230",
"49.145.113.145",
"168.18.160.134",
"72.193.218.222",
"199.216.164.10",
"120.144.130.89",
"104.130.67.208",
"50.160.221.147",
"163.47.141.50",
"91.200.12.136",
"83.222.0.0/19",
"67.231.16.0/20",
"72.231.0.196",
"180.216.68.197",
"183.160.178.135",
"183.160.176.16",
"24.25.221.150",
"92.222.109.43",
"142.134.243.215",
"216.181.221.72",
"113.205.170.110",
"74.142.2.98",
"192.235.8.3",
"2402:4000:BBFC:36FC:E469:F2F0:9351:71A0",
"80.244.81.191",
"2607:FB90:1377:F765:D45D:46BF:81EA:9773",
"2600:1009:B012:7D88:418B:54BA:FCBC:4584",
"104.237.224.0/19",
"2600:1008:B01B:E495:C05A:7DD3:926:E83C",
"168.8.249.234",
"162.211.179.36",
"138.68.0.0/16",
"145.236.37.195",
"67.205.128.0/18",
"2A02:C7D:2832:CE00:B914:19D6:948D:B37D",
"107.77.203.212",
"2607:FB90:65C:A136:D46F:23BA:87C2:3D10",
"2A02:C7F:DE2F:7900:5D64:E991:FFF0:FA93",
"82.23.32.186",
"106.76.243.74",
"82.33.48.223",
"180.216.160.0/19",
"94.102.184.35",
"94.102.184.26",
"109.92.162.54",
"2600:8800:7180:BF00:4C27:4591:347C:736C",
"178.41.186.50",
"184.97.134.128",
"176.221.32.0/22",
"207.99.40.142",
"109.97.241.134",
"82.136.64.19",
"91.236.74.119",
"197.210.0.0/16",
"173.230.128.0/19",
"162.216.16.0/22",
"80.111.222.211",
"191.37.28.21",
"124.124.103.194",
"50.207.7.198",
"220.233.131.98",
"107.77.241.11",
"68.112.39.0/27",
"173.236.128.0/17",
"49.49.240.24",
"96.31.10.178",
"50.251.229.75"]


# In[105]:


train_ip_list = list(set([a for b in train_ips.tolist() for a in b]))
text_ip_list = list(set([a for b in test_ips.tolist() for a in b]))

# get common elements
blocked_ip_list_train = list(set(train_ip_list).intersection(blocked_ips))
blocked_ip_list_test = list(set(test_ip_list).intersection(blocked_ips))

print("There are", len(blocked_ip_list_train), "blocked IPs in train dataset")
print("There are", len(blocked_ip_list_test), "blocked IPs in test dataset")


# # Word clouds

# In[95]:


# wordclouds
categories = label_cols + ["clean"]

mask_files = ["../data/wordcloud_masks/toxic-sign.png",
             "../data/wordcloud_masks/biohazard-symbol.png",
             "../data/wordcloud_masks/megaphone.png",
             "../data/wordcloud_masks/bomb.png",
             "../data/wordcloud_masks/anger.png",
             "../data/wordcloud_masks/swords.png",
             "../data/wordcloud_masks/twitter_mask.png"]
masks = [abs(255-np.array(Image.open(file))[:,:,1]) for file in mask_files]
masks[-1] = abs(255-masks[-1])  # don't invert the twitter bird mask

stopword = set(STOPWORDS)
for i, mask in enumerate(masks):
    subset = train[train.iloc[:,i+2] == 1]
    text = subset["comment_text"].values
    wc = WordCloud(width=1800, height=1400, background_color="white", max_words=2000, stopwords=stopword, mask=mask, collocations=False)
    wc.generate(" ".join(text))
    plt.figure(figsize=(20,10))
    plt.axis("off")
    plt.title("Words frequented in {:s} comments".format(categories[i]), **kwargs)
    plt.imshow(wc.recolor(colormap="Reds" if i < 6 else "summer"), alpha=0.98)
    plt.savefig('word_cloud_' + str(i) + ".png", dpi=800)
    plt.show()


# ## UMAP embedding
# 
# The following code performs a UMAP embedding on FastText embeddings of the top 100 negative and positive words according to polarity analysis.

# In[96]:


get_ipython().run_cell_magic('time', '', '# load FastText embedding model with gensim\n# WARNING: takes a long time -- the file is ~4gb\nfasttext_model = gensim.models.KeyedVectors.load_word2vec_format("../data/toxic_comments/features/crawl-300d-2M.vec")')


# In[97]:


from contractions import negative_100, positive_100

embeddings_negative = []
embeddings_positive = []
words_n = []
words_p = []
for word_n, word_p in zip(negative_100, positive_100):
    if word_n not in fasttext_model or word_p not in fasttext_model:
        continue
    embeddings_negative.append(fasttext_model[word_n])
    embeddings_positive.append(fasttext_model[word_p])
    words_n.append(word_n)
    words_p.append(word_p)
embeddings = embeddings_negative + embeddings_positive
words = words_n + words_p

umap_model = umap.UMAP(n_components=2, metric="cosine", n_neighbors=20)
umap_embeddings = umap_model.fit_transform(embeddings)


# In[101]:


x_coords = umap_embeddings[:,0]
y_coords = umap_embeddings[:,1]

fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(1,1,1)
sc = plt.scatter(x_coords, y_coords, cmap="bwr", c=[1 for _ in words_n] + [0 for _ in words_p])
#for label, x, y in zip(words, x_coords, y_coords):
#    annot = ax.annotate(label, xy=(x,y), xytext=(0,0), textcoords="offset points")#, bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"))
plt.axis("off")
plt.savefig("./figures/umap_100posneg.png", dpi=300)
plt.show()


# **Observations:**
# - Using UMAP embeddings under cosine-distance (with only 2 components!) of the top 100 most negative and 100 most positive words, we can see a clear clustering that is linearly separable with minimal overlap. This might give some intuition for why a linear classifier can perform so well on this dataset.

# # Data cleaning

# In[137]:


corpus = merge["comment_text"]

# dictionary of common misspellings
df_tmp = pd.read_csv("../data/utils/misspellings.csv")
misspellings_ = dict(df_tmp.values)
# lowercase
misspellings = {}
for key, val in misspellings_.items():
    misspellings[key.lower()] = val.lower()

# dictionary of contractions and other substitutions
df_tmp = pd.read_csv("../data/utils/contractions.csv")
contractions_ = dict(df_tmp.values)
# lowercase
contractions = {}
for key, val in contractions_.items():
    contractions[key.lower()] = val.lower()
    
del df_tmp, misspellings_, contractions_  # free up memory


# In[133]:


# show stopwords
print(eng_stopwords)


# In[138]:


def clean(comment, dicts=[]):
    # lowercase
    comment = comment.lower()
    
    # remove \n
    comment = re.sub("\\n", "", comment)
    
    # leaky elements like IP, user, links, article IDs
    # remove IP addresses
    comment = re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", "", comment)
    # remove usernames
    comment = re.sub("\[\[.*\]", "", comment)
    # remove URLs
    comment = re.sub("^(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?$", "", comment)
    # remove article IDs
    comment = re.sub("\d:\d\d\s{0,5}$", "", comment)
    # remove over-repeated characters
    comment = re.sub(r'(.)\1+', r'\1\1', comment) 
    # substitue words BEFORE tokenizer to avoid cases like {"f*ck"} -> {'f', '*', "ck"}
    for key, val in APPO.items():
        if key in comment:
            comment = comment.replace(key, val)
#     # Isolate punctuation
#     comment = re.sub(r'([\'\"\.\(\)\!\?\-\\\/\,])', r' \1 ', comment)
    # Remove some special characters
    comment = re.sub(r'([\;\:\|•«\n])', ' ', comment)
    
    # split sentences into words
    words = tokenizer.tokenize(comment)
    
    # (') apostrophe replacment: you're -> you are
    # (basic dictionary lookup using cell above)
#     words = [APPO[word] if word in APPO else word for word in words]
    # perform substitutions based on input dictionaries e.g. common misspellings
    for d in dicts:
        words = [d[word] if word in d else word for word in words]
    # lemmatization: change words based on morphology e.g. "studying" -> "study", "studies" -> "study"
    words = [lem.lemmatize(word, "v") for word in words]
    # remove stopwords e.g. "is", "the", "i"
    words = [w for w in words if not w in eng_stopwords]
    
    clean_sent = " ".join(words)
    
    return clean_sent


# In[139]:


# get index of a particular comment with instances of "f*ck"
idx = train[train["comment_text"].str.contains("Yes I f\*cking do assert you should add ALL his grants. If you've got a f\*cking")].index  #12235

corpus.iloc[idx].values[0]


# In[140]:


clean(corpus.iloc[idx].values[0])


# - occurences of "f*ck" have been substituted
# - excessive repetitions of characters are reduced e.g. "..." -> "..", "fuuuuuuuuck" -> "fuuck"
# - lemmatization replaces e.g. "fucking" -> "fuck" based on morphology

# In[141]:


get_ipython().run_cell_magic('time', '', '\nclean_corpus = corpus.apply(clean)')


# In[142]:


# split into train and test
clean_train = clean_corpus.iloc[:len(train),]
clean_test = clean_corpus.iloc[len(train):,]

# merge with labels
clean_train = pd.concat([clean_train, train.iloc[:,2:-1]], axis=1)

# save cleaned-up data for future use
clean_train.to_csv("../data/toxic_comments/clean_train.csv", encoding="utf-8")
clean_test.to_csv("../data/toxic_comments/clean_test.csv", encoding="utf-8")


# # Direct features for clean corpus
# 
# TF-IDF unigrams to create count features

# In[153]:


get_ipython().run_cell_magic('time', '', '"""\nTF-IDF vectorizer parameters:\n    min_df=n : ignore terms that appear less than n times\n    max_features=None : create as many words as present in the text corpus\n        WARNING: memory consumptive is left as "None"\n    analyzer="word" : create features from words (can use character level instead)\n    ngram_range=(n,m) : use grams of size n to m e.g. (1,1) chooses only single words\n    strip_accents="unicode" : removes accents from characters\n    use_idf=1, smooth_idf=1 : enable IDF\n    sublinear_tf=1 : apply sublinear tf scaling i.e. replace tf with 1+log(tf)\n"""\n\ntfv = TfidfVectorizer(min_df=10, max_features=10000,\n                     strip_accents="unicode", analyzer="word", ngram_range=(1,1),\n                     use_idf=1, smooth_idf=1, sublinear_tf=1,\n                     stop_words="english")\ntfv.fit(clean_corpus)\nfeatures = np.array(tfv.get_feature_names())\n\ntrain_unigrams = tfv.transform(clean_corpus.iloc[:len(train)])\ntest_unigrams = tfv.transform(clean_corpus.iloc[len(train):])')


# In[2]:


def top_tfidf_features(row, features, top_n=25):
    """
    Get top tfidf values in row and return them with their corresponding feature names.
    """
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_features = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_features)
    df.columns = ["feature", "tfidf"]
    return df


def top_features_in_doc(Xtr, features, row_id, top_n=25):
    """
    Top tfidf features in specific document (i.e. matrix row)
    """
    row = np.squeeze(Xtr[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)


def top_mean_features(Xtr, features, grp_ids, min_tfidf=0.1, top_n=25):
    """
    Return the top n features that on average are most important amongst documents in rows
    identified by indices in grp_ids.
    """
    D = Xtr[grp_ids.astype(int)].toarray()
    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_features(tfidf_means, features, top_n)


def top_features_by_class(Xtr, features, min_tfidf=0.1, top_n=25):
    """
    Return a list of dfs, where each df holds top_n features and their mean tfidf value
    is calculated across documents with the same class label.
    """
    dfs = []
    cols = train_tags.columns
    for col in cols:
        ids = train_tags.index[train_tags[col] == 1]
        features_df = top_mean_features(Xtr, features, ids, min_tfidf=min_tfidf, top_n=top_n)
        features_df.label = label
        dfs.append(features_df)
    return dfs


# In[154]:


get_ipython().run_cell_magic('time', '', '# get top n for unigrams\ntfidf_top_n_per_class = top_features_by_class(train_unigrams, features)')


# In[155]:


plt.figure(figsize=(16,22))
plt.suptitle("TF-IDF top words per class (unigrams)", **kwargs)
GridSpec(4,2)

color = sns.color_palette()
grid_locs = [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1), (3,0)]
for i, gloc, category in zip(range(len(grid_locs)), grid_locs, label_cols + ["clean"]):
    plt.subplot2grid((4,2), gloc, colspan=2 if i == len(grid_locs)-1 else 1)
    if i == len(grid_locs)-1:
        sns.barplot(tfidf_top_n_per_class[i].feature.iloc[0:19], tfidf_top_n_per_class[i].tfidf.iloc[0:19], alpha=0.8)
    else:
        sns.barplot(tfidf_top_n_per_class[i].feature.iloc[0:9], tfidf_top_n_per_class[i].tfidf.iloc[0:9], color=color[i], alpha=0.8)
    plt.title("class : {:s}".format(category), **kwargs)
    plt.xlabel("Word", **kwargs)
    plt.ylabel("TF-IDF score", **kwargs)


# In[169]:


get_ipython().run_cell_magic('time', '', '# 2-grams\ntfv = TfidfVectorizer(min_df=100, max_features=30000,\n                     strip_accents="unicode", analyzer="word", ngram_range=(2,2),\n                     use_idf=1, smooth_idf=1, sublinear_tf=1,\n                     stop_words="english")\ntfv.fit(clean_corpus)\nfeatures = np.array(tfv.get_feature_names())\n\ntrain_bigrams = tfv.transform(clean_corpus.iloc[:len(train)])\ntest_bigrams = tfv.transform(clean_corpus.iloc[len(train):])')


# In[170]:


get_ipython().run_cell_magic('time', '', 'tfidf_top_n_per_class = top_features_by_class(train_bigrams, features)')


# In[171]:


plt.figure(figsize=(16,22))
plt.suptitle("TF-IDF top words per class (unigrams)", **kwargs)
GridSpec(4,2)

color = sns.color_palette()
grid_locs = [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1), (3,0)]
for i, gloc, category in zip(range(len(grid_locs)), grid_locs, label_cols + ["clean"]):
    plt.subplot2grid((4,2), gloc, colspan=2 if i == len(grid_locs)-1 else 1)
    if i == len(grid_locs)-1:
        sns.barplot(tfidf_top_n_per_class[i].feature.iloc[0:9], tfidf_top_n_per_class[i].tfidf.iloc[0:9], alpha=0.8)
    else:
        sns.barplot(tfidf_top_n_per_class[i].feature.iloc[0:5], tfidf_top_n_per_class[i].tfidf.iloc[0:5], color=color[i], alpha=0.8)
    plt.title("class : {:s}".format(category), **kwargs)
    plt.xlabel("Word", **kwargs)
    plt.ylabel("TF-IDF score", **kwargs)


# In[180]:


get_ipython().run_cell_magic('time', '', '# character level\ntfv = TfidfVectorizer(min_df=100, max_features=30000,\n                     strip_accents="unicode", analyzer="char", ngram_range=(1,4),\n                     use_idf=1, smooth_idf=1, sublinear_tf=1,\n                     stop_words="english")\ntfv.fit(clean_corpus)\nfeatures = np.array(tfv.get_feature_names())\n\ntrain_charlevel = tfv.transform(clean_corpus.iloc[:len(train)])\ntest_charlevel = tfv.transform(clean_corpus.iloc[len(train):])')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'tfidf_top_n_per_class = top_features_by_class(train_charlevel, features)')


# In[ ]:


plt.figure(figsize=(16,22))
plt.suptitle("TF-IDF top words per class (unigrams)", **kwargs)
GridSpec(4,2)

color = sns.color_palette()
grid_locs = [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1), (3,0)]
for i, gloc, category in zip(range(len(grid_locs)), grid_locs, label_cols + ["clean"]):
    plt.subplot2grid((4,2), gloc, colspan=2 if i == len(grid_locs)-1 else 1)
    if i == len(grid_locs)-1:
        sns.barplot(tfidf_top_n_per_class[i].feature.iloc[0:9], tfidf_top_n_per_class[i].tfidf.iloc[0:9], alpha=0.8)
    else:
        sns.barplot(tfidf_top_n_per_class[i].feature.iloc[0:5], tfidf_top_n_per_class[i].tfidf.iloc[0:5], color=color[i], alpha=0.8)
    plt.title("class : {:s}".format(category), **kwargs)
    plt.xlabel("Character", **kwargs)
    plt.ylabel("TF-IDF score", **kwargs)

