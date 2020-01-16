### 1. Introduction

So, if you plan to create chatbots this year, or you want to use the power of unstructured text, this guide is the right starting point.

The aim of the article is to teach the concepts of natural language processing and apply it on real data set. Moreover, we also have a video based [**course on NLP**](https://courses.analyticsvidhya.com/courses/natural-language-processing-nlp?utm_source=blog&utm_medium=ultimate-guide-to-understand-implement-natural-language-processing-codes-in-python) with 3 real life projects.

NLP is a branch of data science that consists of systematic processes for analyzing, understanding, and deriving information from the text data in a smart and efficient manner.

```python
# install nltk
sudo pip install -U nltk
# download nltk data
import nltk  
nltk.download() 
```

### 2. Text Preprocessing

Since, text is the **most unstructured** form of all the available data, various types of noise are present in it and the data is not readily analyzable without any pre-processing. The entire process of **cleaning and standardization** of text, making it noise-free and ready for analysis is known as **text preprocessing**.

It is predominantly comprised of three steps:

- Noise Removal
- Lexicon Normalization
- Object Standardization

#### 2.1 Noise Removal

**Any piece of text which is not relevant to the context of the data and the end-output can be specified as the noise.**

*For example – language stopwords (commonly used words of a language – is, am, the, of, in etc), URLs or links, social media entities (mentions, hashtags), punctuations and industry specific words. This step deals with removal of all types of noisy entities present in the text.*

A general approach for noise removal is to prepare a **dictionary of noisy entities**, and iterate the text object by tokens (or by words), eliminating those tokens which are present in the noise dictionary.

```python
# Sample code to remove noisy words from a text

noise_list = ["is", "a", "this", "..."] 
def _remove_noise(input_text):
    words = input_text.split() 
    noise_free_words = [word for word in words if word not in noise_list] 
    noise_free_text = " ".join(noise_free_words) 
    return noise_free_text

_remove_noise("this is a sample text")
>>> "sample text"
```

Another approach is to use the regular expressions while dealing with special patterns of noise.

```python
# Sample code to remove a regex pattern 
import re 

def _remove_regex(input_text, regex_pattern):
    urls = re.finditer(regex_pattern, input_text) 
    for i in urls: 
        input_text = re.sub(i.group().strip(), '', input_text)
    return input_text

regex_pattern = "#[\w]*"  

_remove_regex("remove this #hashtag from analytics vidhya", regex_pattern)
>>> "remove this  from analytics vidhya"
```

#### 2.2 Lexicon Normalization

词典标准化

Another type of textual noise is about the multiple representations exhibited by single word.

For example – “play”, “player”, “played”, “plays” and “playing” are the different variations of the word – “play”, Though they mean different but contextually all are similar. 

The step converts all the disparities of a word into their **normalized form** (also known as lemma). 

The most common lexicon normalization practices are :

- **Stemming:** Stemming is a rudimentary rule-based process of stripping the suffixes (“ing”, “ly”, “es”, “s” etc) from a word.
- **Lemmatization:** Lemmatization, on the other hand, is an organized & step by step procedure of obtaining the root form of the word, it makes use of vocabulary (dictionary importance of words) and morphological analysis (word structure and grammar relations).

```python
from nltk.stem.wordnet import WordNetLemmatizer 
lem = WordNetLemmatizer()

from nltk.stem.porter import PorterStemmer 
stem = PorterStemmer()

word = "multiplying" 
lem.lemmatize(word, "v")
>> "multiply" 
stem.stem(word)
>> "multipli"
```



#### 2.3 Object Standardization

Text data often contains words or phrases which are not present in any standard lexical dictionaries. These pieces are not recognized by search engines and models.

Some of the examples are – acronyms, hashtags with attached words, and colloquial slangs. With the help of regular expressions and manually prepared data dictionaries, this type of noise can be fixed, the code below uses a dictionary lookup method to replace social media slangs from a text.

```python
lookup_dict = {'rt':'Retweet', 'dm':'direct message', "awsm" : "awesome", "luv" :"love", "..."}
def _lookup_words(input_text):
    words = input_text.split() 
    new_words = [] 
    for word in words:
        if word.lower() in lookup_dict:
            word = lookup_dict[word.lower()]
        new_words.append(word) new_text = " ".join(new_words) 
        return new_text

_lookup_words("RT this is a retweeted tweet by Shivam Bansal")
>> "Retweet this is a retweeted tweet by Shivam Bansal"
```

but cc for credit card, while 'accept' is not

### 3. Text to Features (Feature Engineering on text data)

To analyse a preprocessed data, it needs to be converted into features. Depending upon the usage, text features can be constructed using assorted techniques – **Syntactical Parsing, Entities / N-grams / word-based features, Statistical features, and word embeddings.**

#### 3.1 Syntactic Parsing

Syntactical parsing involves the analysis of words in the sentence for grammar and their arrangement in a manner that shows the relationships among the words. Dependency Grammar and Part of Speech tags are the important attributes of text syntactics.

- **Dependency Trees**:   Dependency grammar is a class of syntactic text analysis that deals with (labeled) asymmetrical binary relations between two lexical items (words). Every relation can be represented in the form of a **triplet** (relation, governor, dependent).
  The python wrapper [StanfordCoreNLP](http://stanfordnlp.github.io/CoreNLP/) (by Stanford NLP Group, only commercial license) and NLTK dependency grammars can be used to generate dependency trees.
- **Part of speech tagging**:  Apart from the grammar relations, every word in a sentence is also associated with a part of speech (pos) tag (nouns, verbs, adjectives, adverbs etc). The pos tags defines the usage and function of a word in the sentence.

#### 3.2 Entity Extraction (Entities as features)

Entities are defined as the most important chunks of a sentence – noun phrases, verb phrases or both. Entity Detection algorithms are generally **ensemble models** of rule based parsing, dictionary lookups, pos tagging and dependency parsing. The applicability of entity detection can be seen in the automated **chat bots**, content analyzers and consumer insights.

- **Named Entity Recognition (NER)**:  The process of detecting the named entities such as person names, location names, company names etc from the text is called as NER.



















reference： https://www.analyticsvidhya.com/blog/2017/01/ultimate-guide-to-understand-implement-natural-language-processing-codes-in-python/