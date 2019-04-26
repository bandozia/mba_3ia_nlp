#%% imports
import pandas as pd
import numpy as np
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

#%%
df = pd.read_csv("data/movie_review.csv")
df = df[["text","tag"]]
df = shuffle(df)
df = df.reset_index(drop=True)

df = df[0:1000]
labels = np.array(df["tag"])

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
train_text, test_text, labels_train, labels_test = train_test_split(df, labels, test_size=0.3)

#%%
common_dic = Dictionary(train_text.text)
common_corpus = [common_dic.doc2bow(t) for t in train_text.text]

#%%
test_corpus = [common_dic.doc2bow(t) for t in test_text.text]

#%%
lda = LdaModel(common_corpus, num_topics=2)

#%%
#test = lda[test_corpus[0]]
#%%
cdf = pd.DataFrame(test_text["tag"])
cdf["lda_topic"] = [max(lda[x],key=lambda item: item[1])[0] for x in test_corpus]
cdf["lda_prob"] = [max(lda[x],key=lambda item: item[1])[1] for x in test_corpus]

#%%
cdf.head()