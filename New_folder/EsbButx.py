#import nltk.stem.wordnet
import numpy as np
import nltk
import string
import random


f= open('C:\Users\Admin\OneDrive\Documents\bai_tap\xu_ly_ngon_ngu_tu_nhien\baikiemtradiemB\New folder\rnKrAvN.txt','r', errors = 'ignore')
raw_doc = f.read()
raw_doc = raw_doc.lower()
nltk.dowload('punkt')
nltk.dowload('wordnet')
nltk.dowload('omw-1.4')
sentence_tokens= nltk.sent_tokenize(raw_doc)
word_tokens= nltk.word_tokenize(raw_doc)
sentence_tokens[:5]
word_tokens[:5]
lemmer = nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
    return[lemmer.lemmatize(token)for token in tokens]
remove_punc_dict = dict((ord(punct),None)for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punc_dict)))
greet_inputs =('hello','hi','whassup','how are you?')
greet_responses = ('hi','Hey','hey there!','There there!!')
def greet(sentence):
    for word in sentence.split():
        if word.lower() in greet_inputs:
            return random.choice(greet_responses)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import cosine_similarity

def response(user_response):
    robo1_resonse = ''
    TfidfVec =TfidfVectorizer(tokenizer=LemNormalize, stop_word = 'english')
    tfidf = TfidfVec.fit_transform(sentence_tokens)
    vals = cosine_similarity(tfidf[-1],tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if (req_tfidf == 0):
        robo1_resonse = robo1_resonse + "I am sorry. Unable to understand you!"
        return robo1_resonse
    else:
        robo1_resonse= robo1_resonse + sentence_tokens[idx]
        return robo1_resonse
flag = True
print("Hello! I am the Learning Bot. start typing your text after greeting to talk to me. For ending convo type bye!")
while(flag == True):
    user_response = input()
    user_response = user_response.lower()
    if(user_response != 'bye'):
        if(user_response == 'thank you' or user_response == 'thanks' ):
            flag = False
            print('Bot: you are Welcome..')
        else:
            if(greet(user_response) != None):
                print('Bot' + greet(user_response))
            else:
                sentence_tokens.append(user_response)
                word_tokens = word_tokens + nltk.word_tokenize(user_response)
                final_words = list(set(word_tokens))
                print('Bot: ', end = '')
                print(response(user_response))
                sentence_tokens.remove(user_response)
    else:
        flag = False
        print('Bot: Goodbye!')



