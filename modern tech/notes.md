### 1. Introduction

Natural language processing (NLP) is a theory-motivated range of computational techniques for the **automatic analysis and representation of human language**.

Recent NLP research is now increasingly focusing on the use of new **deep learning methods**.

Deep learning enables multi-level **automatic feature representation** learning. In contrast, traditional machine learning based NLP systems liaise heavily on **hand-crafted features**. Such hand-crafted features are **time-consuming** and often incomplete.

现在顶会大概70%NLP论文是深度学习的

### 2. Distributed Representation

Statistical NLP has emerged as the primary option for modeling complex natural language tasks. 维度诅咒

#### 2.1 Word Embeddings

**Distributional vectors or word embeddings** essentially follow the distributional hypothesis, according to which ***words with similar meanings tend to occur in similar context***. 

Thus, these vectors try to capture the characteristics of the **neighbors of a word**. The main advantage of distributional vectors is that they capture **similarity** between words. Measuring similarity between vectors is possible, using measures such as **cosine similarity**. Word embeddings are often used as the **first data processing layer in a deep learning model**.  Word embeddings have been responsible for state-of-the-art results in a wide range of NLP tasks.



模型不识别字符，而word embeddings就是将字符转化成数字。这篇介绍的文章：https://www.analyticsvidhya.com/blog/2017/06/word-embeddings-count-word2veec/。

> 1. frequency based embedding，基于频率的
>    1. Count vector，单词在文章中出现的次数组成的向量
>    2. TF-IDF Vector， 
> 2. prediction based vector
>    1. CBOW(continuous bag of words)，给定一个Context，预测在该context下出现该词的概率
>    2. skip-gram model， 与前者相反，给定一个word时，预测context出现的概率



#### 2.2 Word2vec

**CBOW** computes the conditional probability of a target word given the context words surrounding it across a window of size *k*. On the other hand, the **skip-gram** model does the exact opposite of the CBOW model, by predicting the surrounding context words given the central target word. The context words are assumed to be located symmetrically to the target words within a distance equal to the window size in both directions.

CBOW是个nn

#### 2.3 Character Embeddings

基于字符

#### 2.4 Contextualized Word Embeddings

The quality of word representations is generally gauged by its ability to encode syntactical information and handle polysemic behavior (or word senses). These properties result in improved semantic word representations.





### 3. Convolutional Neural Networks

Following the popularization of word embeddings and its ability to represent words in a distributed space, the need arose for an effective feature function that **extracts higher-level features from constituting words or n-grams**. These abstract features would then be used for numerous NLP tasks such as sentiment analysis, summarization, machine translation, and question answering (QA). CNNs turned out to be the natural choice given their effectiveness in computer vision tasks

#### 3.1 Basic CNN

##### 3.1.1 Sentence Modeling

conditional random field (CRF)

##### 3.1.2 Applications











































reference: https://nlpoverview.com/