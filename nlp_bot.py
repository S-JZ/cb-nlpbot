import nltk

#import speech_recognition as sr
#from playsound import playsound
#from gtts import gTTS 
import os 
import random
import string
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
#from sklearn.metrics.pairwise_distances import l1, l2
import numpy as np
#import pyaudio
#import speech_recognition as sr
import statistics 
# import matplotlib.pyplot as plt
import pandas as pd
import csv
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#from wordcloud import WordCloud

nltk.download('stopwords')

path = os.path.join(os.getcwd(),'Query form (Responses) - Questions.csv')
df = pd.read_csv(path)

pd.set_option('display.max_colwidth', None)
df.head()

def remove_punc(text):
    return text.translate(str.maketrans('','',string.punctuation))

df['wo_punc'] = df.Questions.apply(lambda text: remove_punc(text))
df.head()

comment_words = ""
stopwords = set(stopwords.words('english'))
# iterate through the csv file
for val in df.Questions:
     
    # typecaste each val to string
    val = str(val)
 
    # split the value
    tokens = val.split()
     
    # Converts each token into lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
     
    comment_words += " ".join(tokens) + " "
 
#wordcloud = WordCloud(width = 800, height = 800, background_color ='white', stopwords = stopwords, min_font_size = 10).generate(comment_words)
 
# plot the WordCloud image                      
# plt.figure(figsize = (8, 8), facecolor = None)
# plt.imshow(wordcloud)
# plt.axis("off")
# plt.tight_layout(pad = 0)
 
# plt.show()

#stopwords = set(stopwords.words('english'))
def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in stopwords])

df['wo_stopwords'] = df.wo_punc.apply(lambda text: remove_stopwords(text))
df.head()

stemmer = PorterStemmer()
def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])

df['stemmed'] = df.wo_stopwords.apply(lambda text: stem_words(text))
df.head()

comment_words = ""
#stopwords = set(stopwords.words('english'))
# iterate through the csv file
for val in df.stemmed:
     
    # typecaste each val to string
    val = str(val)
 
    # split the value
    tokens = val.split()
     
    # Converts each token into lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
     
    comment_words += " ".join(tokens) + " "
 
# wordcloud = WordCloud(width = 800, height = 800, background_color ='white', stopwords = stopwords, min_font_size = 10).generate(comment_words)
 
# # plot the WordCloud image                      
# plt.figure(figsize = (8, 8), facecolor = None)
# plt.imshow(wordcloud)
# plt.axis("off")
# plt.tight_layout(pad = 0)
 
# plt.show()

df = df.drop('wo_stopwords', axis=1)

df = df.drop('wo_punc', axis=1)

# df.head()

questions = df["stemmed"]
answers =  df["Answers"]

question_dict = {"who","what","where","when","why","how","which","?"}
length = len(questions)
list_index = list(range(0, length))

len(questions)

def greeting_response(text):
  text = remove_punc(text.lower())

  #Bots respnse
  bot_greetings = ["Hi","Hey","Hello","Hey there"]
  #Users Greeing
  user_greeting = ["hi","hello","greetings","wassup","hey","hey there"]
  info = ["how are you","how have you been"]

  for word in text.split():
    if word in user_greeting:
      return random.choice(bot_greetings)+", tell me how can I help you?"
  for word in text.split():
    if word in user_greeting:
      return random.choice(bot_greetings)+", tell me how can I help you?"

def bot_response(user_input):
  user_input = stem_words(remove_stopwords(remove_punc(user_input.lower())))
  lt1 = []
  lt1.append(user_input)
  questions_list = list(questions)
  questions_list.append(user_input)
  
  bot_response = ""
  cm = CountVectorizer().fit_transform(questions_list)
  #similarity_scores = pairwise_distances(cm, cm[-1], metric='manhattan')
  #similarity_scores = pairwise_distances(cm, cm[-1], metric='euclidean')
  similarity_scores = cosine_similarity(cm[-1], cm)
  similarity_scores_list = list(similarity_scores.flatten())
  index = similarity_scores_list.index(max(similarity_scores_list[:-1]))  #index_sort(similarity_scores_list) 
  response_flag = 0
  
  if similarity_scores_list[index] > 0.0:
      bot_response += answers[index]
      #print(index[i] , questions[index[i]] , answers[index[i]])
      #y.append(similarity_scores_list[index])
      response_flag = 1
      #scores.append(statistics.mean(y)) 

  if response_flag == 0:
    bot_response += "I apologise, I don't understand."
    #scores.append(0)
  questions_list.remove(user_input) 
    
  return bot_response

# print("NLP Bot")
# Question = []
# Answer = []
# scores = []
# lt = []
# while True:
#     y=[]
#     exit_list = ["exit", "see you later", "bye", "break", "quit"]
#     user_input = input("You: ")

#     if remove_punc(user_input.lower()) in exit_list:
#         res = ("Bye, Chat with you later")
#         language = 'en'
#         print("Bot:", res, end = '\n')
#         scores.append(1)
#         lt.append(user_input)
#         Question.append(user_input)
#         Answer.append(res)
#         break
    
#     else:
#         if greeting_response(user_input) != None:
#             res = (greeting_response(user_input))
#             print("Bot:", res, end = '\n')
#             scores.append(1)
#         else:
#             res = (bot_response(user_input))
#             print("Bot:", res, end = '\n')
#     language = 'en'
    
#     Question.append(user_input)
#     Answer.append(res)

# """### Cosine Similarity"""

def nlp(user_input):
        print("NLP Bot")
        Question = []
        Answer = []
        scores = []
        lt = []
        
        #y=[]
        exit_list = ["exit", "see you later", "bye", "break", "quit"]
        #user_input = input("You: ")

        if remove_punc(user_input.lower()) in exit_list:
            res = ("Bye, Chat with you later")
            language = 'en'
            # print("Bot:", res, end = '\n')
            #scores.append(1)
            lt.append(user_input)
            Question.append(user_input)
            Answer.append(res)
            #break
        
        else:
            if greeting_response(user_input) != None:
                res = (greeting_response(user_input))
                # print("Bot:", res, end = '\n')
                #scores.append(1)
            else:
                res = (bot_response(user_input))
                # print("Bot:", res, end = '\n')
        language = 'en'
        
        # Question.append(user_input)
        # Answer.append(res)

        return res

# Hdf = pd.DataFrame(list(zip(Question, Answer, scores)), 
#                columns =['Question', 'Answer', 'Similiarity Scores']) 
# Hdf 
# Hdf.to_csv('5.csv')

# Hdf #cosine similarity

# df #euclidean

# df # manhattan distance

# def plot():    
#     x=[]
#     for i in range(1, len(scores) + 1):
#         x.append(i)
#     plt.style.use('seaborn')
#     plt.title("Similiarity Score Graph")
#     plt.xlabel("Question No.")
#     plt.ylabel("Similiarity Between Question and Answer")
#     plt.plot(x,scores,color='black')
#     plt.scatter(x,scores,color='orange')

# plot() #cosine

# plot() #euclidean

# plot() #manhattan

# statistics.mean(scores) #manhattan

# statistics.mean(scores) #euclidean

# statistics.mean(scores) #cosine



# """## NOT REQUIRED SECTION:

# """

# def index_sort(list_var):
#   length = len(list_var)
#   list_index = list(range(0, length))
#   print(list_var)
#   x = list_var
#   for i in range(length):
#     for j in range(length):
#       if x[list_index[i]] > x[list_index[j]]:
#         temp = list_index[i]
#         list_index[i] = list_index[j]
#         list_index[j] = temp
#   print(list_index)
#   return list_index
