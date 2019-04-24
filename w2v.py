#%% imports
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import random
from sklearn.utils import shuffle
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import svm

#%% carregar dados
df = pd.read_csv("data/movie_review.csv")

#%%
df = df[["text","tag"]]
df = shuffle(df)
df = df.reset_index(drop=True)
#%%
df.head()

#%%
def TreatText(data):    
    stops = set(stopwords.words("english"))  
    data['text'] = [re.sub("[^a-zA-Z]", " ",data['text'][i]) for i in range(len(data))] 
    data['text'] = [word_tokenize(data['text'][i].lower()) for i in range(len(data))] 
    data['text'] = [[w for w in data['text'][i] if w not in stops]for i in range(len(data))]
    return(data)
    
#%%
df = TreatText(df)

#%%
labels = np.array(df["tag"])

#%%
def meanVector(model,phrase):
    vocab = model.wv.vocab
    phrase = " ".join(phrase)
    phrase = [x for x in word_tokenize(phrase) if x in vocab]    
    if phrase == []:
        vetor = [0.0]*300 
    else:         
        vetor = np.mean([model[word] for word in phrase],axis=0)
    return vetor

#%%
def createFeatures(base,model): 
    features = [meanVector(model,base['text'][i])for i in range(len(base))]
    return features

#%%
model_skip = Word2Vec(df["text"], sg=1, min_count=10, size = 300, window=4, workers=8)
model_cbow = Word2Vec(df["text"], sg=0, min_count=10, size = 300, window=4, workers=8)   

#%%
df_skip = createFeatures(df, model_skip)
df_cbow = createFeatures(df, model_cbow)

#%%
def train_test(base_df):
    X_train, X_test, y_train, y_test = train_test_split(base_df[0:100], labels[0:100], test_size=0.3, random_state=109)
    clf = svm.SVC(kernel="linear")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_test,y_pred   

#%%
sg_test,sg_pred = train_test(df_skip)
cb_test,cb_pred = train_test(df_cbow)

#%%
print(classification_report(sg_test,sg_pred))

#%%
print(classification_report(cb_test,cb_pred))