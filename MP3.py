import pandas
import csv
from gensim import downloader

def analyze(model, dicts, csv_writer):
  correct = 0
  without_guess = 0
  for i, x in enumerate(dicts):
    word_zero_in_vec = x['0'] in model.key_to_index
    word_one_in_vec = x['1'] in model.key_to_index
    word_two_in_vec = x['2'] in model.key_to_index
    word_three_in_vec = x['3'] in model.key_to_index
    if x['question'] in model.key_to_index:
      similarity = []
      if word_zero_in_vec:
        similarity.append(model.similarity(x['question'], x['0']))
      if word_one_in_vec:
        similarity.append(model.similarity(x['question'], x['1']))
      if word_two_in_vec:
        similarity.append(model.similarity(x['question'], x['2']))
      if word_three_in_vec:
        similarity.append(model.similarity(x['question'], x['3']))
      if len(similarity) == 0:
        print('guess')
        data=[x['question'],x['answer'],'','guess']
      else:
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
    csv_writer.writerow(data)
  return (correct, without_guess)

## Task 1: Evaluation of the word2vec-google-news-300 Pre-trained Model
word2vec = downloader.load('word2vec-google-news-300')
df = pandas.read_csv('synonyms.csv')
dicts = df.to_dict('records')
header = ['question-word', 'answer-word', 'guess-word', 'label']
f = open('word2vec-google-news-300-details.csv', 'w', newline='')
writer = csv.writer(f)
writer.writerow(header)

(correct, without_guess) = analyze(word2vec, dicts, writer)
f.close()
f = open('analysis.csv', 'w', newline='')
writer = csv.writer(f)
header = ['model name', 'vocabulary size', 'correct(C)', 'without_guess(V)','accuracy']
data = ['word2vec-google-news-300', str(len(word2vec)), str(correct), str(without_guess),str(correct/without_guess)]
writer.writerow(header)
writer.writerow(data)
f.close()

## Task 2: Comparison with Other Pre-trained Models

glove_twitter_25 = downloader.load('glove-twitter-25')
f = open('glove-twitter-25-details.csv', 'w', newline='')
writer = csv.writer(f)
(glove_twitter_25_correct, glove_twitter_25_without_guess) = analyze(glove_twitter_25, dicts, writer)
f.close()

glove_twitter_200 = downloader.load('glove-twitter-200')
f = open('glove-twitter-200-details.csv', 'w', newline='')
writer = csv.writer(f)
(glove_twitter_200_correct, glove_twitter_200_without_guess) = analyze(glove_twitter_200, dicts, writer)
f.close()

glove_wiki_gigaword_50 = downloader.load('glove-wiki-gigaword-50')
f = open('glove-wiki-gigaword-50-details.csv', 'w', newline='')
writer = csv.writer(f)
(glove_wiki_gigaword_50_correct, glove_wiki_gigaword_50_without_guess) = analyze(glove_wiki_gigaword_50, dicts, writer)
f.close()

glove_wiki_gigaword_300 = downloader.load('glove-wiki-gigaword-300')
f = open('glove-wiki-gigaword-300.csv', 'w', newline='')
writer = csv.writer(f)
(glove_wiki_gigaword_300_correct, glove_wiki_gigaword_300_without_guess) = analyze(glove_wiki_gigaword_300, dicts, writer)
f.close()

f = open('analysis.csv', 'a', newline='')
writer = csv.writer(f)

data = ['glove-twitter-25', str(len(glove_twitter_25)), str(glove_twitter_25_correct), str(glove_twitter_25_without_guess),str(glove_twitter_25_correct/glove_twitter_25_without_guess)]
writer.writerow(data)
data = ['glove-twitter-200', str(len(glove_twitter_200)), str(glove_twitter_200_correct), str(glove_twitter_200_without_guess),str(glove_twitter_200_correct/glove_twitter_200_without_guess)]
writer.writerow(data)
data = ['glove-wiki-gigaword-50', str(len(glove_wiki_gigaword_50)), str(glove_wiki_gigaword_50_correct), str(glove_wiki_gigaword_50_without_guess),str(glove_wiki_gigaword_50_correct/glove_wiki_gigaword_50_without_guess)]
writer.writerow(data)
data = ['glove-wiki-gigaword-300', str(len(glove_wiki_gigaword_300)), str(glove_wiki_gigaword_300_correct), str(glove_wiki_gigaword_300_without_guess),str(glove_wiki_gigaword_300_correct/glove_wiki_gigaword_300_without_guess)]
writer.writerow(data)

f.close()