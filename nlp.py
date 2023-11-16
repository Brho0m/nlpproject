#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <p style="text-align: center;"><img src="https://docs.google.com/uc?id=1lY0Uj5R04yMY3-ZppPWxqCr5pvBLYPnV" class="img-fluid" alt="CLRSWY"></p>
# 
# ___

# # WELCOME!

# Welcome to the "***Sentiment Analysis and Classification Project***" project, the first and only project of the ***Natural Language Processing (NLP)*** course.
# 
# This analysis will focus on using Natural Language techniques to find broad trends in the written thoughts of the customers.
# The goal in this project is to predict whether customers recommend the product they purchased using the information in their review text.
# 
# One of the challenges in this project is to extract useful information from the *Review Text* variable using text mining techniques. The other challenge is that you need to convert text files into numeric feature vectors to run machine learning algorithms.
# 
# At the end of this project, you will learn how to build sentiment classification models using Machine Learning algorithms (***Logistic Regression, Naive Bayes, Support Vector Machine, Random Forest*** and ***Ada Boosting***), **Deep Learning algorithms** and **BERT algorithm**.
# 
# Before diving into the project, please take a look at the Determines and Tasks.
# 
# - ***NOTE:*** *This tutorial assumes that you already know the basics of coding in Python and are familiar with the theory behind the algorithms mentioned above as well as NLP techniques.*
# 
# 

# ---
# ---
# 

# # #Determines
# The data is a collection of 23486 Rows and 10 column variables. Each row includes a written comment as well as additional customer information.
# Also each row corresponds to a customer review, and includes the variables:
# 
# 
# **Feature Information:**
# 
# **Clothing ID:** Integer Categorical variable that refers to the specific piece being reviewed.
# 
# **Age:** Positive Integer variable of the reviewers age.
# 
# **Title:** String variable for the title of the review.
# 
# **Review Text:** String variable for the review body.
# 
# **Rating:** Positive Ordinal Integer variable for the product score granted by the customer from 1 Worst, to 5 Best.
# 
# **Recommended IND:** Binary variable stating where the customer recommends the product where 1 is recommended, 0 is not recommended.
# 
# **Positive Feedback Count:** Positive Integer documenting the number of other customers who found this review positive.
# 
# **Division Name:** Categorical name of the product high level division.
# 
# **Department Name:** Categorical name of the product department name.
# 
# **Class Name:** Categorical name of the product class name.
# 
# ---
# 
# The basic goal in this project is to predict whether customers recommend the product they purchased using the information in their *Review Text*.
# Especially, it should be noted that the expectation in this project is to use only the "Review Text" variable and neglect the other ones.
# Of course, if you want, you can work on other variables individually.
# 
# Project Structure is separated in five tasks: ***EDA, Feature Selection and Data Cleaning , Text Mining, Word Cloud*** and ***Sentiment Classification with Machine Learning, Deep Learning and BERT model.***.
# 
# Classically, you can start to know the data after doing the import and load operations.
# You need to do missing value detection for Review Text, which is the only variable you need to care about. You can drop other variables.
# 
# You will need to apply ***noise removal*** and ***lexicon normalization*** processes by using the capabilities of the ***nltk*** library to the data set that is ready for text mining.
# 
# Afterwards, you will implement ***Word Cloud*** as a visual analysis of word repetition.
# 
# Finally, You will build models with five different algorithms and compare their performance. Thus, you will determine the algorithm that makes the most accurate emotion estimation by using the information obtained from the * Review Text * variable.
# 
# 
# 
# 
# 

# ---
# ---
# 

# # #Tasks
# 
# #### 1. Exploratory Data Analysis
# 
# - Import Modules, Load Discover the Data
# 
# #### 2. Feature Selection and Data Cleaning
# 
# - Feature Selection and Rename Column Name
# - Missing Value Detection
# 
# #### 3. Text Mining
# 
# - Tokenization
# - Noise Removal
# - Lexicon Normalization
# 
# #### 4. WordCloud - Repetition of Words
# 
# - Detect Reviews
# - Collect Words
# - Create Word Cloud
# 
# 
# #### 5. Sentiment Classification with Machine Learning
# 
# - Train - Test Split
# - Vectorization
# - TF-IDF
# - Logistic Regression
# - Naive Bayes
# - Support Vector Machine
# - Random Forest
# - AdaBoost
# - Deep Leraning Model
# - BERT Model
# - Model Comparison

# ---
# ---
# 

# # Sentiment analysis of women's clothes reviews
# 
# 
# In this project we used sentiment analysis to determined whether the product is recommended or not. We used different machine learning algorithms to get more accurate predictions. The following classification algorithms have been used: Logistic Regression, Naive Bayes, Support Vector Machine (SVM), Random Forest and Ada Boosting, Deep learning algorithm and BERT algorithm. The dataset comes from Woman Clothing Review that can be find at (https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews.
# 

# ## 1. Exploratory Data Analysis

# ### Import Libraries, Load and Discover the Data

# In[1]:


from google.colab import drive
drive.mount('/content')


# In[ ]:


import tensorflow as tf
import os

# Note that the `tpu` argument is for Colab-only
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])

tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
print("All devices: ", tf.config.list_logical_devices('TPU'))


# In[ ]:


import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from warnings import filterwarnings
filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# In[ ]:


df = pd.read_csv("/content/MyDrive/Womens Clothing E-Commerce Reviews.csv")
df.head()


# In[ ]:


df.drop(columns=["Unnamed: 0"], inplace=True)
df.head()


# ### Data Wrangling
# 
# Data wrangling typically involves pre-processing steps such as data collection, cleaning, and organizing. The goal during this process is to transform the data into a usable and meaningful form for subsequent stages.

# In[ ]:


plt.figure(figsize = (7,4))
ax = sns.countplot(x="Rating",
                   data=df,
                   hue = "Recommended IND")
for p in ax.containers:
    ax.bar_label(p)


# In[ ]:


df = df[~((df["Rating"] == 1) & (df["Recommended IND"] == 1))]
df = df[~((df["Rating"] == 2) & (df["Recommended IND"] == 1))]
df = df[~((df["Rating"] == 3) & (df["Recommended IND"] == 1))]
df = df[~((df["Rating"] == 4) & (df["Recommended IND"] == 0))]
df = df[~((df["Rating"] == 5) & (df["Recommended IND"] == 0))]


# In[ ]:


plt.figure(figsize = (7,4))
ax = sns.countplot(x="Rating",
                   data=df,
                   hue = "Recommended IND")
for p in ax.containers:
  ax.bar_label(p)


# #### Check Proportion of Target Class Variable:

# The target class variable is imbalanced, where "Recommended" values are more dominating then "Not Recommendation".

# In[ ]:


plt.fig = plt.figure(figsize = (7,5))
ax = sns.countplot(x="Recommended IND",
                   data=df)
ax.bar_label(ax.containers[0]);


# ## 2. Feature Selection and Data Cleaning
# 
# From now on, the DataFrame you will work with should contain two columns: **"Review Text"** and **"Recommended IND"**. You can do the missing value detection operations from now on. You can also rename the column names if you want.
# 
# 

# ### Feature Selection and Rename Column Name

# In[ ]:


df.rename(columns={"Review Text":"text", "Recommended IND":"label"},
          inplace=True)


# In[ ]:


df=df[['text','label']]
df.head()


# ### Missing Value Detection

# In[ ]:


df.isnull().sum()


# In[ ]:


df.dropna(inplace = True)
df.reset_index(drop=True,
               inplace=True)
df


# ## 3. Text Mining
# 
# Text is the most unstructured form of all the available data, therefore various types of noise are present in it. This means that the data is not readily analyzable without any pre-processing. The entire process of cleaning and standardization of text, making it noise-free and ready for analysis is known as **text preprocessing**.
# 
# The three key steps of text preprocessing:
# 
# - **Tokenization:**
# This step is one of the top priorities when it comes to working on text mining. Tokenization is essentially splitting a phrase, sentence, paragraph, or an entire text document into smaller units, such as individual words or terms. Each of these smaller units are called tokens.
# 
# - **Noise Removal:**
# Any piece of text which is not relevant to the context of the data and the end-output can be specified as the noise.
# For example – language stopwords (commonly used words of a language – is, am, the, of, in etc), URLs or links, upper and lower case differentiation, punctuations and industry specific words. This step deals with removal of all types of noisy entities present in the text.
# 
# 
# - **Lexicon Normalization:**
# Another type of textual noise is about the multiple representations exhibited by single word.
# For example – “play”, “player”, “played”, “plays” and “playing” are the different variations of the word – “play”. Though they mean different things, contextually they all are similar. This step converts all the disparities of a word into their normalized form (also known as lemma).
# There are two methods of lexicon normalisation; **[Stemming or Lemmatization](https://www.guru99.com/stemming-lemmatization-python-nltk.html)**. Lemmatization is recommended for this case, because Lemmatization as this will return the root form of each word (rather than just stripping suffixes, which is stemming).
# 
# As the first step change text to tokens and convertion all of the words to lower case.  Next remove punctuation, bad characters, numbers and stop words. The second step is aimed to normalization them throught the Lemmatization method.
# 
# 
# ***Note:*** *Use the functions of the ***[nltk Library](https://www.guru99.com/nltk-tutorial.html)*** for all the above operations.*
# 
# 

# ### Tokenization, Noise Removal, Lexicon Normalization

# In[ ]:


stop_words = stopwords.words('english')

for i in ["not", "no"]:
        stop_words.remove(i)


# In[ ]:


def cleaning(data):

    #1. Tokenize
    text_tokens = word_tokenize(data.replace("'", "").lower())

    #2. Remove Puncs and numbers
    tokens_without_punc = [w for w in text_tokens if w.isalpha()]

    #3. Removing Stopwords
    tokens_without_sw = [t for t in tokens_without_punc if t not in stop_words]

    #4. lemma
    text_cleaned = [WordNetLemmatizer().lemmatize(t) for t in tokens_without_sw]

    #joining
    return " ".join(text_cleaned)


# ## 4. WordCloud - Repetition of Words
# 
# Now you'll create a Word Clouds for reviews, representing most common words in each target class.
# 
# Word Cloud is a data visualization technique used for representing text data in which the size of each word indicates its frequency or importance. Significant textual data points can be highlighted using a word cloud.
# 
# You are expected to create separate word clouds for positive and negative reviews. You can qualify a review as positive or negative, by looking at its recommended status. You may need to use capabilities of matplotlib for visualizations.
# 
# You can follow the steps below:
# 
# - Detect Reviews
# - Collect Words
# - Create Word Cloud
# 

# ### Detect Reviews (positive and negative separately)

# In[ ]:


positive_sentences = df[df["label"] == 1]["text"]
positive_sentences = positive_sentences.apply(cleaning)
positive_sentences


# In[ ]:


negative_sentences = df[df["label"] == 0]["text"]
negative_sentences = negative_sentences.apply(cleaning)
negative_sentences


# ### Collect Words (positive and negative separately)

# In[ ]:


positive_words = " ".join(positive_sentences)
positive_words[:1000]


# In[ ]:


negative_words = " ".join(negative_sentences)
negative_words[:1000]


# ### Create Word Cloud (for most common words in recommended not recommended reviews separately)

# In[ ]:


from wordcloud import WordCloud


# In[ ]:


wordcloud_positive = WordCloud(background_color="black",
                               max_words =250,
                               scale=3)


# In[ ]:


wordcloud_positive.generate(positive_words)


# In[ ]:


import matplotlib.pyplot as plt
plt.figure(figsize = (13,13))
plt.imshow(wordcloud_positive,
           interpolation="bilinear")
plt.axis("off")
plt.show()


# In[ ]:


wordcloud_negative = WordCloud(background_color="black",
                               max_words=250,
                               colormap='gist_heat',
                               scale=3)

wordcloud_negative.generate(negative_words)

plt.figure(figsize=(13,13))
plt.imshow(wordcloud_negative,
           interpolation="bilinear")
plt.axis("off")
plt.show()


# ## 5. Sentiment Classification with Machine Learning and Deep Learning
# 
# Before moving on to modeling, as data preprocessing steps you will need to perform **[vectorization](https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/)** and **train-test split**. You have performed many times train test split process before.
# But you will perform the vectorization for the first time.
# 
# Machine learning algorithms most often take numeric feature vectors as input. Thus, when working with text documents, you need a way to convert each document into a numeric vector. This process is known as text vectorization. Commonly used vectorization approach that you will use here is to represent each text as a vector of word counts.
# 
# At this moment, you have your review text column as a token (which has no punctuations and stopwords). You can use Scikit-learn’s CountVectorizer to convert the text collection into a matrix of token counts. You can imagine this resulting matrix as a 2-D matrix, where each row is a unique word, and each column is a review.
# 
# Train all models using TFIDF and Count vectorizer data.
# 
# **For Deep learning model, use embedding layer for all words.**
# **For BERT model, use TF tensor.**
# 
# After performing data preprocessing, build your models using following classification algorithms:
# 
# - Logistic Regression,
# - Naive Bayes,
# - Support Vector Machine,
# - Random Forest,
# - Ada Boosting,
# - Deep Learning Model,
# - BERT Model.

# ### Train - Test Split

# To run machine learning algorithms we need to convert text files into numerical feature vectors. We will use bag of words model for our analysis.
# 
# First we spliting the data into train and test sets:

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X = df["text"].values
y = df["label"].map({0:1, 1:0}).values


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.1,
                                                    stratify=y,
                                                    random_state=101)


# In the next step we create a numerical feature vector for each document:

# ### Count Vectorization

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


vectorizer = CountVectorizer(preprocessor=cleaning,
                             min_df=3,
                             max_df=0.90)
X_train_count = vectorizer.fit_transform(X_train)
X_test_count = vectorizer.transform(X_test)


# In[ ]:


X_train_count.toarray()


# In[ ]:


pd.DataFrame(X_train_count.toarray(), columns = vectorizer.get_feature_names_out())


# ### TF-IDF

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


tf_idf_vectorizer = TfidfVectorizer(preprocessor=cleaning,
                                    min_df=3,
                                    max_df=0.90)
X_train_tf_idf = tf_idf_vectorizer.fit_transform(X_train)
X_test_tf_idf = tf_idf_vectorizer.transform(X_test)


# In[ ]:


X_train_tf_idf.toarray()


# In[ ]:


pd.DataFrame(X_train_tf_idf.toarray(),
             columns = tf_idf_vectorizer.get_feature_names_out())


# ## Eval functions

# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report, f1_score, recall_score, accuracy_score, precision_score


# In[ ]:


def eval(model, X_train, X_test):
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    print(confusion_matrix(y_test, y_pred))
    print("Test_Set")
    print(classification_report(y_test,y_pred))
    print("Train_Set")
    print(classification_report(y_train,y_pred_train))


# ## Logistic Regression

# ### CountVectorizer

# In[ ]:


from sklearn.linear_model import LogisticRegression
log = LogisticRegression(C=0.01,
                         max_iter=1000,
                         class_weight= "balanced",
                         random_state=101)
log.fit(X_train_count,y_train)


# In[ ]:


print("LOG MODEL")
eval(log, X_train_count, X_test_count)


# In[ ]:


y_pred = log.predict(X_test_count)
y_pred_proba= log.predict_proba(X_test_count)[:,1]

log_AP_count = average_precision_score(y_test, y_pred_proba)
log_count_rec = recall_score(y_test, y_pred)
log_count_f1 = f1_score(y_test,y_pred)


# ## Support Vector Machine (SVM)
# 
# ### Countvectorizer

# In[ ]:


from sklearn.svm import LinearSVC
svc = LinearSVC(C=0.001,
                class_weight="balanced",
                random_state=101)
svc.fit(X_train_count,y_train)


# In[ ]:


print("SVC MODEL")
eval(svc, X_train_count, X_test_count)


# In[ ]:


y_pred = svc.predict(X_test_count)
decision_function= svc.decision_function(X_test_count)

svc_AP_count = average_precision_score(y_test, decision_function)
svc_count_rec = recall_score(y_test, y_pred)
svc_count_f1 = f1_score(y_test,y_pred)


# ### TD-IDF

# In[ ]:


svc = LinearSVC(C=0.03,
                class_weight="balanced",
                random_state=101)
svc.fit(X_train_tf_idf,y_train)


# In[ ]:


print("SVC MODEL")
eval(svc, X_train_tf_idf, X_test_tf_idf)


# In[ ]:


y_pred = svc.predict(X_test_tf_idf)
decision_function= svc.decision_function(X_test_tf_idf)

svc_AP_tfidf = average_precision_score(y_test, decision_function)
svc_tfidf_rec = recall_score(y_test, y_pred)
svc_tfidf_f1 = f1_score(y_test,y_pred)


# ## Ada Boosting
# 
# ### Countvectorizer

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(n_estimators= 100,
                         random_state = 42)
ada.fit(X_train_count, y_train)


# In[ ]:


print("Ada MODEL")
eval(ada, X_train_count, X_test_count)


# In[ ]:


y_pred = ada.predict(X_test_count)
y_pred_proba= ada.predict_proba(X_test_count)[:,1]

ada_AP_count = average_precision_score(y_test, y_pred_proba)
ada_count_rec = recall_score(y_test, y_pred)
ada_count_f1 = f1_score(y_test,y_pred)


# ### TF-IDF

# In[ ]:


ada = AdaBoostClassifier(n_estimators= 100,
                         random_state = 42,
                         learning_rate=0.7)
ada.fit(X_train_tf_idf, y_train)


# In[ ]:


print("Ada MODEL")
eval(ada, X_train_tf_idf, X_test_tf_idf)


# In[ ]:


y_pred = ada.predict(X_test_tf_idf)
y_pred_proba= ada.predict_proba(X_test_tf_idf)[:,1]

ada_AP_tfidf = average_precision_score(y_test, y_pred_proba)
ada_tfidf_rec = recall_score(y_test, y_pred)
ada_tfidf_f1 = f1_score(y_test,y_pred)


# ## DL modeling

# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Embedding, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ### Tokenization

# In[ ]:


num_words = 15001
tokenizer = Tokenizer(num_words=num_words) #filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n1234567890'


# In[ ]:


tokenizer.fit_on_texts(X)


# ### Creating word index

# In[ ]:


tokenizer.word_index


# ### Converting tokens to numeric

# In[ ]:


X_num_tokens = tokenizer.texts_to_sequences(X)


# In[ ]:


num_tokens = [len(tokens) for tokens in X_num_tokens]
num_tokens = np.array(num_tokens)


# ### Fixing token counts of all documents (pad_sequences)

# In[ ]:


X_pad = pad_sequences(X_num_tokens,
                      maxlen = max_tokens)


# ### Train Set Split

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_pad,
                                                    y,
                                                    test_size=0.1,
                                                    stratify=y,
                                                    random_state=101)


# ### Modeling

# In[ ]:


model = Sequential()


# In[ ]:


embedding_size = 50


# In[ ]:


model.add(Embedding(input_dim=num_words,
                    output_dim=embedding_size,
                    input_length=max_tokens))
model.add(Dropout(0.2))

model.add(GRU(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(GRU(units=25, return_sequences=True))
model.add(Dropout(0.2))

model.add(GRU(units=12))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))


# In[ ]:


optimizer = Adam(learning_rate=0.001)


# In[ ]:


model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=["Recall"])


# In[ ]:


model.summary()


# In[ ]:


from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor="val_recall",
                           mode="max",
                           verbose=1,
                           patience = 2,
                           restore_best_weights=True)


# In[ ]:


from sklearn.utils import class_weight
classes_weights = class_weight.compute_sample_weight(class_weight='balanced',
                                                     y=y_train)
pd.Series(classes_weights).unique()


# In[ ]:


model.fit(X_train, y_train, epochs=10, batch_size=128, sample_weight=classes_weights,
         validation_data=(X_test, y_test), callbacks=[early_stop])


# ### Model evaluation

# In[ ]:


model_loss = pd.DataFrame(model.history.history)
model_loss.head()


# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, roc_auc_score

y_pred = model.predict(X_test) >= 0.5

print(confusion_matrix(y_test, y_pred))
print("-------------------------------------------------------")
print(classification_report(y_test, y_pred))


# In[ ]:


from sklearn.metrics import average_precision_score

average_precision_score(y_test, y_pred_proba)


# In[ ]:


DL_AP = average_precision_score(y_test, y_pred_proba)
DL_f1 = f1_score(y_test, y_pred)
DL_rec = recall_score(y_test, y_pred)


# ## BERT Modeling

# ### Tokenization

# In[ ]:


from transformers import AutoTokenizer #BertTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# For every sentence...
num_of_sent_tokens = []
for sent in X:

    # Tokenize the text and add `[CLS]` and `[SEP]` tokens.

    input_ids = tokenizer.encode(sent,
                                 add_special_tokens=True)
    num_of_sent_tokens.append(len(input_ids))

print('Max sentence length: ', max(num_of_sent_tokens))


# In[ ]:


np.array(num_of_sent_tokens).mean()


# In[ ]:


sum(np.array(num_of_sent_tokens) <= 162) / len(num_of_sent_tokens)


# In[ ]:


X_train2, X_test2, y_train2, y_test2 = train_test_split(X,
                                                        y,
                                                        test_size=0.1,
                                                        stratify=y,
                                                        random_state=101)


# In[ ]:


all_sentence_tokens = tokenizer(list(X),
                                max_length=162,
                                truncation=True,
                                padding='max_length',
                                add_special_tokens=True)


# In[ ]:


np.array(all_sentence_tokens['input_ids'])


# In[ ]:


np.array(all_sentence_tokens['attention_mask'])


# In[ ]:


def transformation(X):
  # set array dimensions
  seq_len = 162

  all_sentence_tokens = tokenizer(list(X),
                                  max_length=seq_len,
                                  truncation=True,
                                  padding='max_length',
                                  add_special_tokens=True)

  return np.array(all_sentence_tokens['input_ids']), np.array(all_sentence_tokens['attention_mask'])


# In[ ]:


Xids_train, Xmask_train = transformation(X_train2)

Xids_test, Xmask_test = transformation(X_test2)


# In[ ]:


Xids_train


# In[ ]:


print("Xids_train.shape  :", Xids_train.shape)
print("Xmask_train.shape :", Xmask_train.shape)
print("Xids_test.shape   :", Xids_test.shape)
print("Xmask_test.shape  :", Xmask_test.shape)


# In[ ]:


labels_train = y_train2.reshape(-1,1)
labels_train


# In[ ]:


labels_test = y_test2.reshape(-1,1)
labels_test


# ### Transformation Matrix to Tensorflow Tensor

# In[ ]:


import tensorflow as tf

dataset_train = tf.data.Dataset.from_tensor_slices((Xids_train,
                                                    Xmask_train,
                                                    labels_train))
dataset_train


# In[ ]:


dataset_test = tf.data.Dataset.from_tensor_slices((Xids_test,
                                                   Xmask_test,
                                                   labels_test))
dataset_test


# In[ ]:


def map_func(Tensor_Xids, Tensor_Xmask, Tensor_labels):
    # we convert our three-item tuple into a two-item tuple where the input item is a dictionary
    return {'input_ids': Tensor_Xids, 'attention_mask': Tensor_Xmask}, Tensor_labels


# In[ ]:


# then we use the dataset map method to apply this transformation
dataset_train = dataset_train.map(map_func)
dataset_test = dataset_test.map(map_func)


# In[ ]:


dataset_train


# In[ ]:


dataset_test


# ## Batch Size

# In[ ]:


batch_size = 32

train_ds = dataset_train.batch(batch_size)
val_ds = dataset_test.batch(batch_size)


# In[ ]:


from official.nlp import optimization
epochs = 2
#batch_size = 32
steps_per_epoch = len(train_ds)

num_train_steps = steps_per_epoch * epochs
num_warmup_steps = int(0.1*num_train_steps)

init_lr = 2e-5  # 3e-5, 5e-5
optimizer= optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')


# In[ ]:


print(len(train_ds)*2)
print(int(0.1*len(train_ds)*2))


# ### Creating Model with TPU

# In[ ]:


def create_model():
    from transformers import TFAutoModel #TFBertModel
    from tensorflow.keras.layers import Input, Dropout, Dense, BatchNormalization
    from tensorflow.keras import Model

    model = TFAutoModel.from_pretrained("bert-base-uncased")

    input_ids = Input(shape=(162,), name='input_ids', dtype='int32')
    attention_mask = Input(shape=(162,), name='attention_mask', dtype='int32')

    embeddings = model.bert(input_ids=input_ids, attention_mask=attention_mask)["pooler_output"] #[1]

    x = Dense(80, activation='relu')(embeddings)
    x = BatchNormalization()(x)
    x = Dropout(0.1, name="dropout")(x) #0.1
    y = Dense(1, activation='sigmoid', name='outputs')(x)

    return Model(inputs=[input_ids, attention_mask], outputs=y)


# In[ ]:


with strategy.scope():

  #from tensorflow.keras.optimizers import Adam
  from tensorflow.keras.losses import BinaryCrossentropy
  from tensorflow.keras.metrics import Recall

  #optimizer = Adam(learning_rate=2e-5) #3e-5, 5e-5
  loss = BinaryCrossentropy()
  recall = Recall()
  model3 = create_model()
  model3.compile(optimizer=optimizer, loss=loss, metrics=[recall])


# In[ ]:


model3.summary()


# In[ ]:


model3.fit(train_ds, validation_data= val_ds, epochs=epochs)


# ## Model evaluation

# In[ ]:


model_loss = pd.DataFrame(model3.history.history)
model_loss.head()


# In[ ]:


model_loss.plot()


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix

y_pred = model3.predict(val_ds) >= 0.5

print(classification_report(y_test2, y_pred))


# In[ ]:


y_train_pred = model3.predict(train_ds) >= 0.5

print(classification_report(y_train2, y_train_pred))


# In[ ]:


average_precision_score(y_test2, y_pred_proba)


# ___
# 
# <p style="text-align: center;"><img src="https://docs.google.com/uc?id=1lY0Uj5R04yMY3-ZppPWxqCr5pvBLYPnV" class="img-fluid" alt="CLRSWY"></p>
# 
# ___
