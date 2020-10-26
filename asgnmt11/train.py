import sys
import numpy as np
import pandas as pd

from nltk import RegexpTokenizer
from gensim.models import Word2Vec

tokenizer = RegexpTokenizer(r"\w+")

dataPath = sys.argv[1]
model_file_path = sys.argv[2]

data_df = pd.read_csv(dataPath)

text_data = data_df['text'].values
labels = data_df['label'].values

fake_texts = text_data[np.where(labels == 'FAKE')]
fake_texts = np.array(list(map(lambda x:x.lower(), fake_texts)))
fake_texts = np.array(list(map(tokenizer.tokenize, fake_texts)))


real_texts = text_data[np.where(labels == 'REAL')]
real_texts = np.array(list(map(lambda x:x.lower(), real_texts)))
real_texts = np.array(list(map(tokenizer.tokenize, real_texts)))


fake_model = Word2Vec(fake_texts, size=100, window=10, min_count=5, workers=5)
real_model = Word2Vec(real_texts, size=100, window=10, min_count=5, workers=5)

fake_model.save(model_file_path+'/fake.model')
real_model.save(model_file_path+'/real.model')
