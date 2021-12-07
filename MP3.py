import pandas
import csv
from gensim import models, similarities, downloader

## Task 1: Evaluation of the word2vec-google-news-300 Pre-trained Model

word2vec = downloader.load('word2vec-google-news-300')
df = pandas.read_csv('synonyms.csv')
dicts = df.to_dict('records')
header = ['question-word', 'answer-word', 'guess-word', 'label']
f = open('word2vec-google-news-300-details.csv', 'w', newline='')
writer = csv.writer(f)
writer.writerow(header)

correct = 0
without_guess = 0

for i, x in enumerate(dicts):
  if x['question'] in word2vec.key_to_index:
    similarity = []
    similarity.append(word2vec.similarity(x['question'], x['0']))
    similarity.append(word2vec.similarity(x['question'], x['1']))
    similarity.append(word2vec.similarity(x['question'], x['2']))
    similarity.append(word2vec.similarity(x['question'], x['3']))
    max_value = max(similarity)
    max_index = similarity.index(max_value)
    if(x[str(max_index)] == x['answer']):
      print ('{0}) question: {1}, guess-word: {2}, correct'.format(i,x['question'],x[str(max_index)]))
      data=[x['question'],x['answer'],x[str(max_index)],'correct']
      correct+=1
    else:
      print ('{0}) question: {1}, guess-word: {2}, wrong'.format(i,x['question'],x[str(max_index)]))
      data=[x['question'],x['answer'],x[str(max_index)],'wrong']
    without_guess+=1
  else:
    print('guess')
    data=[x['question'],x['answer'],'','guess']
  writer.writerow(data)

f.close()

f = open('analysis.csv', 'w', newline='')
writer = csv.writer(f)
header = ['model name', 'vocabulary size', 'correct(c)', 'without_guess(V)','accuracy']
data = ['word2vec-google-news-300', str(len(word2vec)), str(correct), str(without_guess),str(correct/without_guess)]
writer.writerow(header)
writer.writerow(data)
f.close()