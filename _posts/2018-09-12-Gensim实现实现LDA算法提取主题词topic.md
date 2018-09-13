---
title:  "gensim实现LDA(Latent Dirichlet Allocation)算法提取主题词(topic)"
header:
  image: /assets/images/header-3.bmp
  teaser: /assets/images/test_teaser.jpg
search: true
toc: true
categories: 
  - python 
 tags:
  - python
  - LDA
  - Latent Dirichlet Allocation
  - gensim
  - topic
last_modified_at: 2018-09-12T08:06:00-05:00
---

主题模型是用来从文本中提取主题（topic）的算法。Latent Dirichlet Allocation（LDA) 隐含分布作为目前最受欢迎的主题模型算法被广泛使用。LDA能够将文本集合转化为不同概率的主题集合。需要注意的是LDA是利用统计手段对主题词汇进行到的处理，是一种词袋（bag-of-words）方法
如：
输入：

> 第一段："Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. It is altogether fitting and proper that we should do this."
> 第二段：'Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal.'
> 第三段："We are met on a great battle-field of that war. We have come to dedicate a portion of that field, as a final resting place for those who here gave their lives that nation might live. "

输出：

    (0, u'0.032*"conceive" + 0.032*"dedicate" + 0.032*"nation" + 0.032*"life"')
    (1, u'0.059*"conceive" + 0.059*"score" + 0.059*"seven" + 0.059*"proposition"')
    (2, u'0.103*"nation" + 0.071*"dedicate" + 0.071*"great" + 0.071*"field"')
    (3, u'0.032*"conceive" + 0.032*"nation" + 0.032*"dedicate" + 0.032*"rest"')
    (4, u'0.032*"conceive" + 0.032*"nation" + 0.032*"dedicate" + 0.032*"battle"')

本文将简单介绍如何使用Python 的nltk、spacy、gensim包，实现包括预处理流程在内的LDA算法。
## 1. 预处理：
### 1.1 分词处理
```python
#第一次使用需要首先下载en包:
#python -m spacy download en
import spacy
spacy.load('en_core_web_sm')
from spacy.lang.en import English
parser = English()
#对文章内容进行清洗并将单词统一降为小写
def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens
```
### 1.2 lemma处理
lemma与stem都是NLP中常用的对于单词的处理：
lemma 将
stem  将
```python
#引入一个同义词、近义词、反义词包
import nltk
#第一次使用需要下载这个nltk包
# nltk.download('wordnet')

from nltk.corpus import wordnet as wn
def get_lemma(word):
    #dogs->dog
    #aardwolves->aardwolf'
    #sichuan->sichuan
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma
```
### 1.3 从nltk包中引入英文停顿词停顿词处理
```python
#第一次使用需要下载停顿词
# nltk.download('stopwords')

en_stop = set(nltk.corpus.stopwords.words('english'))
```
### 1.4 预处理流程
预处理的过程包括以上所提及的分词、lemma处理及停顿词处理
```python
#定义预处理函数
def prepare_text_for_lda(text):
    #分词处理
    tokens = tokenize(text)
    #取出长度大于4的单词
    tokens = [token for token in tokens if len(token) > 4]
    #取出非停顿词
    tokens = [token for token in tokens if token not in en_stop]
    #对词语进行还原
    tokens = [get_lemma(token) for token in tokens]
    return tokens
```
## 2. LDA算法
### 2.1 预处理文本集合
通过预处理函数加载文本集合，需要注意的是，gensim:models.ldamodel 处理对象是一个文本集合而不是文本集，因此其输入应该为[[],``````,[]]结构而不是[]
```python 
    text_1 = u"Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. It is altogether fitting and proper that we should do this."
    text_2 = u'Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal.'
    text_3 = u"We are met on a great battle-field of that war. We have come to dedicate a portion of that field, as a final resting place for those who here gave their lives that nation might live. "
    text_data_1 = prepare_text_for_lda(text_1)
    text_data_2 = prepare_text_for_lda(text_2)
    text_data_3 = prepare_text_for_lda(text_3)
    text_data =[]
    text_data.append(text_data_1)
    text_data.append(text_data_2)
    text_data.append(text_data_3)
    print "text_data :",text_data
```
通过对于三个string的预处理并组合成为一个list集合，数据如下：

    [[u'engage', u'great', u'civil', u'testing', u'whether', u'nation', u'nation', u'conceive', u'dedicate', u'endure', u'altogether', u'fitting', u'proper'], [u'score', u'seven', u'years', u'father', u'bring', u'forth', u'continent', u'nation', u'conceive', u'liberty', u'dedicate', u'proposition', u'create', u'equal'], [u'great', u'battle', u'field', u'dedicate', u'portion', u'field', u'final', u'rest', u'place', u'life', u'nation', u'might']]


### 2.2 使用LDA算法提取主题词
需要注意的是，如下实现LDA算法的gensim.models.ldamodel.LdaModel()与生成的corpus、dictionary密切相关。
```python
    #加载gensim 
    #使用gensim.Dictionary从text_data中生成一个词袋（bag-of-words)
    dictionary = corpora.Dictionary(text_data)
    corpus = [dictionary.doc2bow(text) for text in text_data]

    #加载gensim，使用LDA算法求得前五的topic，
    #同时生成的topic在之后也会被使用到来定义文本所属主题
    
    NUM_TOPICS = 5#定义了生成的主题词的个数
    ldamodel = gensim.models.ldamodel.LdaModel(corpus,              
    	                                       num_topics = NUM_TOPICS,
    	                                       id2word=dictionary,
    	                                       passes=15)
    ldamodel.save('model5.gensim')
    topics = ldamodel.print_topics(num_words=4)
    for topic in topics:
        print(topic)
```
## 3. 附录遇到的问题及修改
### 3.1 来自spacy的报错
```python
import spacy
spacy.load('en')
```
    Traceback (most recent call last):
      File "topial_LDA.py", line 13, in <module>
        spacy.load('en')
      File "C:\Python27\lib\site-packages\spacy\__init__.py", line 15, in load
        return util.load_model(name, **overrides)
      File "C:\Python27\lib\site-packages\spacy\util.py", line 119, in load_model
        raise IOError(Errors.E050.format(name=name))
    IOError: [E050] Can't find model 'en'. It doesn't seem to be a shortcut link, a Python package or a valid path to a data directory.

这条报错是因为没有向spacy指明引入的english类型的语言包具体是那个，在spacy中我们发现了如下多个包：

修改代码代码，实现功能：
```python
import spacy
spacy.load('en_core_web_sm')
```
### 3.2 来自dictionary的报错
这个报错参考2.1

> C:\Python27\lib\site-packages\gensim\utils.py:1209: UserWarning:
> detected Windows; aliasing chunkize to chunkize_serial  
> warnings.warn("detected Windows; aliasing chunkize to
> chunkize_serial") Traceback (most recent call last):   File
> "topial_LDA.py", line 122, in <module>
>     dictionary = corpora.Dictionary(text_data_1)   File "C:\Python27\lib\site-packages\gensim\corpora\dictionary.py", line 81,
> in __init__
>     self.add_documents(documents, prune_at=prune_at)   File "C:\Python27\lib\site-packages\gensim\corpora\dictionary.py", line
> 198, in add_documents
>     self.doc2bow(document, allow_update=True)  # ignore the result, here we only care about updating token ids   File
> "C:\Python27\lib\site-packages\gensim\corpora\dictionary.py", line
> 236, in doc2bow
>     raise TypeError("doc2bow expects an array of unicode tokens on input, not a single string") TypeError: doc2bow expects an array of
> unicode tokens on input, not a single string

