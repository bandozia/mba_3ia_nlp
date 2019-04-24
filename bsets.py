
#%% imports
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer

#%% carregar dados
corpus = pd.read_csv("data/movies.csv")
query_list = ['toy story', 'the lion king','alladin','beauty and the best','cinderella','little mermaid','hercules']
corpus.loc[:,"title"] = corpus.title.apply(lambda t : re.sub(r'\([^)]*\)',"", t))

corpus.head()

#%% criar dtms
vectorizer = CountVectorizer()
vectorizer.fit(corpus.title)
X = vectorizer.transform(corpus.title)
x = vectorizer.transform(query_list)

#%% ajustar parametros
c = 2
m = np.mean(X, 0) + 0.0000000001
N = x.shape[0]
xij = x.toarray()

alpha = c * m
beta = c * (1 - m)
alpha_t = alpha + np.sum(xij, 0)
beta_t = beta + N - np.sum(xij, 0)
nc = np.sum(np.log(alpha + beta) - np.log(alpha + beta +N) + np.log(beta_t) - np.log(beta),1)
q = np.log(alpha_t) - np.log(alpha) - np.log(beta_t) + np.log(beta)

#%% calcular o score
s = nc + np.sum(X.multiply(q),1)


#%% relacionar o score com items do dataset
s_flat = np.array(s).reshape((s.shape[0],))
c_indexes=s_flat.argsort()[::-1][:20]

#%%
result = pd.DataFrame(corpus.iloc[c_indexes]['title'])
result['score'] = s_flat[c_indexes]

#%%
result
