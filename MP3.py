import gensim.downloader as api
import pandas as pd
from gensim.models import Word2Vec
from gensim.similarities import Similarity
from gensim.similarities import Similarity
import csv

##2.2 Task 1: Evaluation of the word2vec-google-news-300 Pre-trained Model

wv = api.load("word2vec-google-news-300")
df = pd.read_csv('synonyms.csv')
dicts = df.to_dict('records')
header = ['question-word', 'answer-word', 'guess-word', 'label']
f = open("word2vec-google-news-300-details.csv", "a")
writer = csv.writer(f)
writer.writerow(header)

c=0
v=0

for i, x in enumerate(dicts):
  if x['question'] in wv.key_to_index:
    similarity = []
    similarity.append(wv.similarity(x['question'], x['0']))
    similarity.append(wv.similarity(x['question'], x['1']))
    similarity.append(wv.similarity(x['question'], x['2']))
    similarity.append(wv.similarity(x['question'], x['3']))
    max_value = max(similarity)
    max_index = similarity.index(max_value)
    if(x[str(max_index)] == x['answer']):
      print ("{0}) question: {1}, guess-word: {2}, correct".format(i,x['question'],x[str(max_index)]))
      data=[x['question'],x['answer'],x[str(max_index)],'correct']
      c+=1
    else:
      print ("{0}) question: {1}, guess-word: {2}, wrong".format(i,x['question'],x[str(max_index)]))
      data=[x['question'],x['answer'],x[str(max_index)],'wrong']
    v+=1
  else:
    print('guess')
    data=[x['question'],x['answer'],'','guess']
  writer.writerow(data)
f.close()

f = open("analysis.csv", "a")
writer = csv.writer(f)
header = ['model name', 'vocabulary size', 'c', 'v','accuracy']
data = ['word2vec-google-news-300', str(len(wv)), str(c), str(v),str(c/v)]
writer.writerow(header)
writer.writerow(data)
f.close()