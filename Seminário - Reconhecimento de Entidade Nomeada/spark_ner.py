# -*- coding: utf-8 -*-
from __future__ import print_function   
import os
import sys
from django.utils.encoding import smart_str, smart_unicode
import re

from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
from nltk.tokenize import word_tokenize
import mpld3
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE

from numpy import array, random, tile
import string
import unicodedata



reload(sys)
sys.setdefaultencoding("utf-8")


os.environ['SPARK_HOME'] = "/usr/local/spark/"

# # Append pyspark  to Python Path
sys.path.append("/usr/local/spark/python/")


n_samples = 2000
n_features = 5000
n_topics = 5
n_top_words = 5


#Create Pyspark import 
try:

    from pyspark import SparkContext
    from pyspark import SparkConf

    print("Successfully imported Spark Modules")
    sc = SparkContext("local", "Simple Language Model Computing")

except ImportError as e:
    print("Can not import Spark Modules", e)

    sys.exit(1)


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

######################################################################################################################################################
#Functions Natural Process Languange
def remove_hashtag(text):
    words = text.split()
    for i in words:
	if i.startswith('#'):	
		words.remove(i)

    text = ' '.join(words)
    return text	

def remove_URL(text):
    clean_tweet = re.match('(.*?)http.*?\s?(.*?)', text)
    if clean_tweet:
	return clean_tweet.group(1)	
    else:
	return text 



def remove_stopwords(text):

    regex = re.compile('[%s]' % re.escape(string.punctuation))
    
    a=[]
   
    words = text.split()
    for t in words:
        new_token = regex.sub(u'',t)
        if not new_token == u'':
            a.append(new_token)

    
    import nltk
    stopwords = nltk.corpus.stopwords.words('portuguese')
    content = [w for w in a if  w.lower().strip() not in stopwords]

    clean_text=[]
    for word in content:
        
        chars = ['http','´','kk','zz','”','https','ht','htt']
        nfkd = unicodedata.normalize('NFKD', word)
        palavraSemAcento = u"".join([c for c in nfkd if not unicodedata.combining(c)])
        
        q = re.sub('[^a-zA-Z0-9 \\\]', ' ', palavraSemAcento)

        for c in chars:
            rep = q.replace(c.decode("utf-8"), '').lower()
       
        if rep != "":  
            clean_text.append(rep.lower().strip())  
                  

    tokens = [t for t in clean_text if len(t) > 2 and not t.isdigit()]     
    ct = ' '.join(tokens)

    return ct





def remove_emoji(text):
    d = []

    d = re.compile("["u"\U0001F600-\U0001F64F"  # emoticons
                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                   "]+", flags=re.UNICODE)
    d = d.sub(r'', text)

    return d

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()


def remove_break_line(x):

    return x.replace('\n', ' ')

######################################################################################################################################################
# Finding Key Terms
def search_terms(x,y):
    x=x.lower()
    z=y.split()
    if z[0] in x and z[1] in x and z[2] in x and z[3] in x:
        
        return x

import polyglot
from polyglot.text import Text, Word

def find_entity(text):
    regex = re.compile('[%s]' % re.escape(string.punctuation))    
    words = text.split()
    a=[]	
    
    for t in words:
        new_token = regex.sub(u'',t)
        if not new_token == u'':
            a.append(new_token)
    b=[]
    for t in a:
   	 nfkd = unicodedata.normalize('NFKD', t)
   	 palavraSemAcento = u"".join([c for c in nfkd if not unicodedata.combining(c)])	   
         x = re.sub('[^a-zA-Z0-9 \\\]', ' ', palavraSemAcento)
	 b.append(x)

    text = ' '.join(b)
    entity = Text(smart_str(text))
    a = entity.language.code
    list_entity =[]
    
    if a=="pt":	
   	 for i in entity.entities:
		 #print(i)
       		 list_entity.append(str(i)[3:].replace("']",""))

   	 ct = ' '.join(list_entity)
    
   	 return ct    
    else:
	 return " "  

#Load Data Set. Use txt file.


t0 = time.time()
data=sc.textFile("Comentarios_contra_classificados.txt")

#When we use the function "map", the spark read line by line of data.
# words = data.map(tokenize)
words = data.map(remove_URL)
words = words.map(remove_emoji)
words = words.map(find_entity)

#Data clean 
documents=words.collect()
    
#   print(documents.shape())
arquivo_out =  open('entidades_contra','w')

for i in documents:
    arquivo_out.write(i)





