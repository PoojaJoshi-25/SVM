import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.stem.porter import PorterStemmer
#import seaborn as sns # histogram plotting

#data loading
df = pd.read_csv(r'C:\Users\Pooja Joshi\Desktop\SVM\SpamHam\Spam.csv')

#print(df.sample(5))  # first five lines will be printed of dataset
#print(df.info())   # provides only data information  for cleaning


##########    DATA    CLEANING  ###################
# label encoding
from sklearn.preprocessing import LabelEncoder  # for giving label ham as 0 and spam as 1
encoder=LabelEncoder()   # LabelEncoder() ->  used to convert categorical labels into numeric format 0 for ham 1 for spam
df['Label'] = encoder.fit_transform(df['Label'])
#print(df.head())  # df.head()  is used to display few lines after label encoding

#misssing values
print(df.isnull().sum())  # used for checking any value is null or not

#duplicate value
print(df.duplicated().sum())    # we get 403 as duplicate items now we have to remove them

# remove dupluactes
df=df.drop_duplicates(keep='first')  # drop_duplicates  removes all duplicates
print(df.duplicated().sum())     # 0 printed

# print(df.shape)

############                   EDA-> Exploritary data analysis       ##################

# checking how much percent is spam and how much is ham by value_counts()
print(df['Label'].value_counts())
# for graphical representation of spam and ham
plt.pie(df['Label'].value_counts(),labels=['ham','spam'], autopct="%0.2f")
plt.show()   # will show graphical representation of data of spam and ham

# how many alphabets are used and how many words for that we have imported ntlk library
nltk.download('punkt')

df['num_character'] = df['EmailText'].apply(len)  # we are counting length of emailtext and making a new coloumn as num_character and storing in it
#print(df.head)

df['num_words'] = df['EmailText'].apply(lambda x:len(nltk.word_tokenize(x)))  # splitting the text into words and counting its length

df['num_sentences']=df['EmailText'].apply(lambda x:len(nltk.sent_tokenize(x))) # counting sentences

print(df.sample(5))

###########################   DATA PRRPROCESSING ######################33
def transform_text(text):
    text=text.lower()   # change all data to lowercase
    text=text.word_tokenize(text)   # text into words

    y=[]
    for i in EmailText:
        if i.isalnum():    # removing special characters
            y.append(i)

    text=y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:  # removing stopwords  stopwords are like "you i my on "
            y.appned(i)

    text=y[:]
    y.clear()

    ps=PorterStemmer()  # used to change dancing danced as only word  dance
    for i in text:
        y.appned(ps.stem(i))

    return " ".join(y)

df['transformed_text'] = df['EmailText'].apply(transformed_text)

df.sample(5)

