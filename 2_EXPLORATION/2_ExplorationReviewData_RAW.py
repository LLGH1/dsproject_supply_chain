#!/usr/bin/env python
# coding: utf-8

# ## Project Supply Chain -- Data Exploration

# Possible Questions for the Project:
# 
# Are customers happier with digital products than with physical ones?
# 
# Are customers willing to switch to digital products from physical ones when they had a positive/negative experience
# 
# **Exploration of metadata (without Text Mining technics)**
# Some ideas (from Maelys):
# 
# - Response rate, influence of brand or source, verified_purchase or not
# 
# - Distribution of scores.
# 
# - Influence of the marketplace or the company on the distribution of notes (hypothesis testin could be used for  this kind of analysis )
# 
# - Information about the 10 most active users, with a small analysis on it (distribution of scores, response rate, company...).
# 
# **Goal 2 Analysis of text (and cleaning if necessary ). You will need to complete the text mining module to be able to do this part.**
# Some ideas :
# - Analyze the punctuation according to the note
# 
# - Analyze the length of the text (nb character, nb words...) according to the note.
# 
# - Analyze the frequency of email addresses, links, phone numbers...
# 
# - Occurrence of words, wordcloud...
# 
# - N-gram
# 
# - Occurrence of some words : delivery order, return order, delivery, SAV, customer service...

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm # color map
import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
# from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
nltk.download('stopwords')
nltk.download('punkt')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import os
project_path = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir, os.path.pardir)).replace('\\', '/')
project_path


# In[ ]:


data_hc = pd.read_csv(project_path + '/data/amazon_reviews_us_Video_Games_v1_00.tsv', sep="\t", error_bad_lines=False)
data_dc = pd.read_csv(project_path + '/data/amazon_reviews_us_Digital_Video_Games_v1_00.tsv', sep="\t", error_bad_lines=False)
data_raw = pd.concat([data_hc,data_dc], axis = 0)


# ### Data Cleansing

# In[ ]:


data_raw.shape


# In[ ]:


data_raw.info()


# In[ ]:


data_raw.head()


# In[ ]:


# check null data
print(data_raw.isnull().sum(axis = 0))


# In[ ]:


# drop missing data, since the amount of missing data is very low
data = data_raw.dropna(axis = 0)


# #### check data quality and clean data

# In[ ]:


# check duplicates
print(data["marketplace"].value_counts()) #Only data from us marketplace, so we can drop the row
print("number of duplicated customer ids = ", len(data[data['customer_id'].duplicated() == True])) #most likely because customers ordered multiple items
print("number of duplicated review ids = ", len(data[data['review_id'].duplicated() == True])) 
#has to be 0 in order to ensure the ID is unique, i think we can drop this row as well in this case
#placeholder for code that shows if product_id, product_title and product_parent are fully correlated
print("product_category: ", data["product_category"].value_counts())
print("star rating: ", data["star_rating"].value_counts())
print("vines: ",data["vine"].value_counts())
print("verified_purchases: ",data["verified_purchase"].value_counts())


# In[ ]:


# check corr
round(data.corr(),2)


# In[ ]:


# drop the rows we do not need for this analysis or our model
to_drop = ["marketplace"] # "review_id", "product_id", "product_parent"
data = data.drop(to_drop, axis=1)


# ### First exploration

# #### reviews per product

# In[ ]:


reviews_per_product = data.groupby(["product_id"])["review_id"].nunique().sort_values().reset_index().rename({"review_id":"num_of_reviews"}, axis=1)

reviews_per_product.tail(1000).plot()   


# In[ ]:


reviews_per_product["num_of_reviews"].quantile([0.01,0.1,0.25, 0.5,0.75,0.9,0.99])


# #### countplot of various data

# In[ ]:


sns.countplot(x=data["product_category"])


# In[ ]:


sns.countplot(x=data["star_rating"])


# In[ ]:


sns.countplot(x=data["verified_purchase"])
#there are a lot of non-verified purchases 
#let's look at how the rating distribution of these reviews looks like compared to the verified ones


# In[ ]:


data.groupby('verified_purchase').size()


# In[ ]:


# alternative pie chat of verified and non-verified purchase
def label_function(val):
    return f'{val / 100 * len(data):,.0f}\n{val:.2f}%'

figsize = (10, 5)

data.groupby('verified_purchase').size().plot(kind='pie', 
                                              autopct=label_function, 
                                              textprops={'fontsize': 10},
                                              cmap = cm.get_cmap('Pastel1'))

plt.ylabel('', fontsize=10)

plt.show()


# In[ ]:


sns.countplot(x=data[data["verified_purchase"]=="Y"]["star_rating"])


# In[ ]:


# alternative pie chat of verified and non-verified purchase
def label_function(val):
    return f'{val / 100 * len(data):,.0f}\n{val:.2f}%'

figsize = (10, 5)

data[data["verified_purchase"]=="Y"].groupby('star_rating').size().plot(kind='pie', 
                                              autopct=label_function, 
                                              textprops={'fontsize': 8},
                                              cmap = cm.get_cmap('Pastel1'))

plt.ylabel('', fontsize=10)

plt.show()


# In[ ]:


# alternative pie chat of verified and non-verified purchase
def label_function(val):
    return f'{val / 100 * len(data):,.0f}\n{val:.2f}%'

figsize = (10, 5)

data[data["verified_purchase"]=="N"].groupby('star_rating').size().plot(kind='pie', 
                                              autopct=label_function, 
                                              textprops={'fontsize': 8},
                                              cmap = cm.get_cmap('Pastel1'))

plt.ylabel('', fontsize=10)

plt.show()


# In[ ]:


sns.countplot(x=data[data["verified_purchase"]=="N"]["star_rating"])
#there are a lot more lower ratings in comparison. 
#It is possible that customers were so unhappy, that they created a 2nd account just to review the game negatively again


# In[ ]:


#there are definitely more hard-copy sales than digital sales, let's look at the ratings from the reviews for each one
sns.countplot(x=data[data["product_category"]=="Video Games"]["star_rating"])


# In[ ]:


sns.countplot(x=data[data["product_category"]=="Digital_Video_Games"]["star_rating"])
#we can see from this simple analysis, that there are a lot more 1-star reviews for digital products


# #### rating problems

# In[ ]:


data["rating_problems"] = data["star_rating"].apply(lambda x: False if x in [1,2,3,4,5,"1", "2", "3", "4", "5" ] else True) 
data["star_rating"].value_counts(normalize=1)
data["rating_problems"].value_counts(normalize=1)


# In[ ]:


data = data[data["rating_problems"] != True]
data["star_rating"] = data["star_rating"].astype("int")
sns.countplot(x=data["star_rating"]) # Data is very imbalanced across classes


# #### reviews over time

# In[ ]:


reviews_over_time = data.groupby("review_date").agg({"review_id":"count"}).plot(kind="line")


# In[ ]:


num_rev_prod_per_rating = data.groupby("star_rating").agg({"review_id":lambda x: x.nunique(), "product_id": lambda x: x.nunique()} )
num_rev_prod_per_rating["rev_per_prod"] = num_rev_prod_per_rating.apply(lambda x: x["review_id"] / x["product_id"], axis=1)
num_rev_prod_per_rating["rev_per_prod"].plot(kind="bar", title="Reviews per product")


# In[ ]:


data.groupby(["star_rating", "verified_purchase"])["review_id"].count().plot(kind="bar")


# #### 10 most rated titles

# In[ ]:


data["product_title"].value_counts().head(10).plot(kind="barh")


# #### Top 10 titles where the reviews recieved the most votes

# In[ ]:


data[["product_title", "total_votes"]].nlargest(10, ["total_votes"]).plot(x="product_title", y="total_votes", kind="barh")


# #### Top 10 titles have the most 5-star reviews

# In[ ]:


data[data["star_rating"] == 5]["product_title"].value_counts().head(10).plot(kind="barh")


# #### Top 10 titles with the lowest rated reviews

# In[ ]:


data[data["star_rating"] == 1]["product_title"].value_counts().head(10).plot(kind="barh")
#that's where SimCity went, reviewers did not like this game at all


# #### Top 10 games with the most helpful reviews

# In[ ]:


data[["product_title", "helpful_votes"]].nlargest(10, ["helpful_votes"]).plot(x="product_title", y="helpful_votes", kind="barh")
#Customers found reviews of SimCity the most helpful allthough the game also had a lot of bad reviews. this again shows that this title was highly controversial


# ### tokenization and stemming of review_body

# In[ ]:


from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk.stem.snowball import EnglishStemmer
stemmer = EnglishStemmer()


stop_words.update(["car", "work", "product", "install"])

def tokenization_and_stemming(text):
    tokens = []
    # exclude stop words and tokenize the document, generate a list of string 
    for word in word_tokenize(text):
        if word.lower() not in stop_words:
            tokens.append(word.lower())

    filtered_tokens = []
    
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if token.isalpha(): # filter out non alphabet words like emoji
            filtered_tokens.append(token)
            
    # stemming
    # Removes ing also in anything ...
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return_string = " ".join(stems)
    
    return return_string


# In[ ]:


df_review_body = data.dropna(subset=['review_body'])


# In[ ]:


df_review_body["processed_reviews"] = df_review_body["review_body"].apply(lambda x: tokenization_and_stemming(x))


# In[ ]:


df_review_body.head()


# ### word cloud of processed_reviews

# In[ ]:


df_processed_reviews = data.dropna(subset="processed_reviews")
processed_review_string = df_processed_reviews.groupby("star_rating").aggregate({"processed_reviews":lambda x: " \n ".join(x)})


# In[ ]:


def wc_for_rating(rating):
    wordcloud = WordCloud(collocations=True).generate(processed_review_string.loc[rating][0][1:5000000].replace("one", "").replace("use", "").replace(" br ", " ").replace("car", "").replace("work", ""))
    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


# In[ ]:


wc_for_rating(1)
wc_for_rating(2)
wc_for_rating(3)
wc_for_rating(4)
wc_for_rating(5)


# ## --- From here new 2023/01/12  ---

# ### 2nd Data Exploration

# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# check corr
round(data.corr(),2)


# In[ ]:


# drop the rows we do not need for this analysis or our model
to_drop = ["marketplace"] # "review_id", "product_id", "product_parent"
data = data.drop(to_drop, axis=1)


# In[ ]:


#converting to lower case
data['review_body'] = data['review_body'].astype(str).str.lower()
data['review_headline'] = data['review_headline'].astype(str).str.lower()
data.head()


# In[ ]:


#removing punctuation and converting to lower case
import re
import string

def text_clean1(text):
    text = text.lower()
    text = re.sub('[%s]' % re.escape(string.punctuation),'',text)
    return text

cleaned1 = lambda x: text_clean1(x)

#sample data to test code
data_subsample = data.iloc[0:100]

# using function on columns
data_subsample['cleaned_reviewheadline'] = data_subsample['review_headline'].apply(cleaned1)
data_subsample['cleaned_reviewbody'] = data_subsample['review_body'].apply(cleaned1)


#Another round of cleaning
def text_clean2(text):
    text = re.sub('\n','', text)
    return text

cleaned2 = lambda x: text_clean2(x)
data_subsample['cleaned_reviewheadline'] = data_subsample['cleaned_reviewheadline'].apply(cleaned2)
data_subsample['cleaned_reviewbody'] = data_subsample['cleaned_reviewbody'].apply(cleaned2)

data_subsample.head()


# In[ ]:


#Remove Stopwords
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stopwords = stopwords.words('english')
data_subsample['cleaned_reviewbody'] = data_subsample['cleaned_reviewbody'].apply(lambda words: ' '.join(word.lower() for word in words.split()if word not in stopwords))
data_subsample.head()


# In[ ]:


#Tokenization
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer


def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(w) for w in text.split(' ')])

data_subsample['tokenized_reviewbody'] = data_subsample['cleaned_reviewbody'].apply(lemmatize_text)
data_subsample.head()


# In[ ]:


from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'w+')
data_subsample['tokenizedbody'] = data_subsample['cleaned_reviewbody'].apply(tokenizer.tokenize)
data_subsample.head()


# In[ ]:


#Text Exploratory Analysis
data_subsample['word_length'] = data_subsample['cleaned_reviewbody'].str.split().str.len()
data_subsample.head()
def sentiment(row):
    if row['star_rating'] == 5:
        return "Positive"
    elif row['star_rating'] == 1:
        return "Negative"
    elif  1 < row['star_rating'] < 5:
        return "Inbetween"
    else:
        return "Undefined"
    

data_subsample['star_sentiment'] = data_subsample.apply(sentiment,axis =1)
data_subsample.head()


# In[ ]:


#Calculating text length
data_subsample[["word_length", "star_sentiment"]].hist(bins=20, figsize=(15, 10))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.set(color_codes=True)
plt.figure(figsize=(15,7))
cmap = ['red', 'green','yellow']
labels = ['Negative', 'Positive', 'Inbetween']

for label,clr in zip(labels,cmap):
    sns.kdeplot(data_subsample.loc[(data_subsample['star_sentiment'] == label),'word_length'], color=clr, shade=True, label=label)
    plt.xlabel('Text Length')
    plt.ylabel('Density')
    plt.legend()


# In[ ]:


#removing punctuation and converting to lower case
import re
import string

def text_clean1(text):
    text = text.lower()
    text = re.sub('[%s]' % re.escape(string.punctuation),'',text)
    return text

cleaned1 = lambda x: text_clean1(x)


# using function on columns
data['cleaned_reviewheadline'] = data['review_headline'].apply(cleaned1)
data['cleaned_reviewbody'] = data['review_body'].apply(cleaned1)


#Another round of cleaning
def text_clean2(text):
    text = re.sub('\n','', text)
    return text

cleaned2 = lambda x: text_clean2(x)
data['cleaned_reviewheadline'] = data['cleaned_reviewheadline'].apply(cleaned2)
data['cleaned_reviewbody'] = data['cleaned_reviewbody'].apply(cleaned2)

data.head()


# In[ ]:


#Remove Stopwords
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stopwords = stopwords.words('english')
data['cleaned_reviewbody'] = data['cleaned_reviewbody'].apply(lambda words: ' '.join(word.lower() for word in words.split()if word not in stopwords))
data.head()


# In[ ]:


#Tokenization
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer


def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(w) for w in text.split(' ')])

data['tokenized_reviewbody'] = data['cleaned_reviewbody'].apply(lemmatize_text)
data.head()


# In[ ]:


#Text Exploratory Analysis
data['word_length'] = data['cleaned_reviewbody'].str.split().str.len()
data.head()
def sentiment(row):
    if row['star_rating'] == 5:
        return "Positive"
    elif row['star_rating'] == 1:
        return "Negative"
    elif  1 < row['star_rating'] < 5:
        return "Inbetween"
    else:
        return "Undefined"
    

data['star_sentiment'] = data.apply(sentiment,axis =1)
data.head()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.set(color_codes=True)
fig, ax = plt.subplots()
cmap = ['red', 'green','yellow']
labels = ['Negative', 'Positive', 'Inbetween']

for label,clr in zip(labels,cmap):
    sns.kdeplot(data.loc[(data['star_sentiment'] == label),'word_length'], ax=ax, color=clr, shade=True, label=label)
    ax.set_xlim(0,199)
    plt.xlabel('Text Length')
    plt.ylabel('Density')
    plt.legend()
   


# ### Word Cloud

# In[ ]:


# Easy and approximate detection of english by scanning reviews for " the ", " and ", "but"
import numpy as np
english_words = ["great", "game", " and ", "the ", "but ", "good", " bad ", " this ", " my ", "awesome", "love", "have", " be ", " it ", " was ",     " an ", "they", "did", "not", "quality", "poor", "perfect", "it ", "for", "his", "work", "recommend", "nice", "excellent", "fantastic", "issues", "very", "my", "gift",        "with", "ok", "can", "just", "you", "memory", "thanks", "now", "uses", "fun", "too", "decent", "problems", "good", "want", "please", "want", "there", "are", "product", "believ",                "like", "again", "only", "much ", "please", " i ", "buy", "what", "money", "waste", " at ", "able", "worth", "try", "epic", "amazing",  "play", "brilliant", "don't",                        "better", " way", "about", "stupid", " run", "bring", "version", " of ", " as ", "it's", "ever", " me ", "how", " by "]
# Could use stopwords, but running time increases with every list member several seconds

data=data.dropna(subset="review_body")
data["language"] = data["review_body"].apply(lambda x: "EN" if (np.any([s in x.lower() for s in english_words])) else "OTHER")


# In[ ]:


data.value_counts("language")


# In[ ]:


data[data["language"]!="EN"]["review_body"].tail(40)


# In[ ]:


from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk.stem.snowball import EnglishStemmer
stemmer = EnglishStemmer()


stop_words.update(["car", "work", "product", "install"])

def tokenization_and_stemming(text):
    tokens = []
    # exclude stop words and tokenize the document, generate a list of string 
    for word in word_tokenize(text):
        if word.lower() not in stop_words:
            tokens.append(word.lower())

    filtered_tokens = []
    
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if token.isalpha(): # filter out non alphabet words like emoji
            filtered_tokens.append(token)
            
    # stemming
    # Removes ing also in anything ...
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return_string = " ".join(stems)
    
    return return_string


# In[ ]:


#data = data.dropna(subset=['review_body'])
data_en = data[data["language"]=="EN"]


# In[ ]:



processed_reviews = data_en["review_body"].apply(lambda x: tokenization_and_stemming(x))


# In[ ]:


processed_reviews.head(3)


# In[ ]:


data_en["processed_reviews"] = processed_reviews


# In[ ]:


import os
os.getcwd()
data_en.to_pickle(r"../../data/data_en.pickle")


# In[ ]:


tokens = processed_reviews.apply(lambda x: x.split(" "))
tokens.head()


# In[ ]:


token_list = [item for sublist in list(tokens) for item in sublist] 
token_string = " ".join(token_list)


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer

def return_word_occurences(token_string, stopwords_in = None):
    vectorizer = CountVectorizer(stop_words = stopwords_in)
    #vectorizer.fit_transform(token_list)
    X = vectorizer.fit_transform([token_string])
    #vocab = vectorizer.vocabulary_
    words = vectorizer.get_feature_names_out()
    word_occurences = X.toarray().sum(axis=0)
    word_occurences_dict = dict(zip(words, word_occurences))
    sorted_occurences = sorted(word_occurences_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_occurences


# In[ ]:


sorted_occurences_initial = return_word_occurences(token_string)
sorted_occurences_initial

pd.DataFrame(sorted_occurences_initial).to_csv(r"../../data/word_occurences.csv")


# In[ ]:


stop_words_cv = set(stopwords.words('english'))
stop_words_cv.update(["game", "br", "play", "get", "one", "would", "make", "first"])

#Update inspired by word cloud per rating class
stop_words_cv_enh = stop_words_cv
stop_words_cv_enh.update(["like", "control", "time"])

sorted_occurences_sw = return_word_occurences(token_string, stopwords_in=stop_words_cv)
sorted_occurences_sw


# In[ ]:


#data = data.dropna(subset="processed_reviews")
processed_review_string = data_en.groupby("star_rating").aggregate({"processed_reviews":lambda x: " ".join(x)})


# In[ ]:


processed_review_string.loc[1]


# In[ ]:


def wc_for_rating(rating, stopwords_in = None, max_words = 100, collocations = False):
    plt.figure(figsize = (20,20))
    wordcloud = WordCloud(collocations=collocations, stopwords = stopwords_in, max_words = max_words).generate(processed_review_string.loc[rating][0])
    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


# In[ ]:


wc_for_rating(1, stopwords_in= stop_words_cv)


# In[ ]:


#wc_for_rating(1, stopwords_in= stop_words_cv)
wc_for_rating(2, stopwords_in= stop_words_cv)
wc_for_rating(3, stopwords_in= stop_words_cv)
wc_for_rating(4, stopwords_in= stop_words_cv)
wc_for_rating(5, stopwords_in= stop_words_cv)


# In[ ]:


wc_for_rating(1, stopwords_in= stop_words_cv_enh)
wc_for_rating(2, stopwords_in= stop_words_cv_enh)
wc_for_rating(3, stopwords_in= stop_words_cv_enh)
wc_for_rating(4, stopwords_in= stop_words_cv_enh)
wc_for_rating(5, stopwords_in= stop_words_cv_enh)


# In[ ]:


if True:
    data.to_csv(r"D:\DSProject\VideoGames_transformed.csv")
else:
    data = pd.read_csv(r"D:\DSProject\amazon_reviews_us_Automotive_v1_00_transformed.csv")


# In[ ]:


data.head()


# In[ ]:


#data = data.dropna(subset="processed_reviews")
# Lots of foreign languages in the dataset, thus filter for marketplace = US
tokens = processed_reviews.apply(lambda x: x.split(" "))


# In[ ]:


tokens.head()


# In[ ]:


token_list = [item for sublist in list(tokens) for item in sublist] 
token_string = " ".join(token_list)


# In[ ]:


token_string = " ".join(token_list)


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer

def return_word_occurences(token_string, stopwords_in = None):
    vectorizer = CountVectorizer(stop_words = stopwords_in)
    #vectorizer.fit_transform(token_list)
    X = vectorizer.fit_transform([token_string])
    #vocab = vectorizer.vocabulary_
    words = vectorizer.get_feature_names_out()
    word_occurences = X.toarray().sum(axis=0)
    word_occurences_dict = dict(zip(words, word_occurences))
    sorted_occurences = sorted(word_occurences_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_occurences


# In[ ]:


sorted_occurences_initial = return_word_occurences(token_string)
sorted_occurences_initial


# In[ ]:


sorted_occurences_initial


# In[ ]:


stop_words_cv = set(stopwords.words('english'))
stop_words_cv.update(["game", "br", "play", "get", "one", "would", "make"])

sorted_occurences_sw = return_word_occurences(token_string, stopwords_in=stop_words_cv)
sorted_occurences_sw


# In[ ]:


sorted_occurences_sw


# In[ ]:


data.columns


# In[ ]:


import re
r = re.compile(r"[a-zA-Z]+")
test = r.match('çáêôæíô')
test2 = r.match('windshiield')

token_list_an = [item for item in token_list_us if not (r.match(item) is None)] 


# ### N-Gram

# #### pre-processing

# In[ ]:


# take the sample data for testing code (faster)
# data_test = data.sample(frac=.1)
data_test = data.copy()


# In[ ]:


# remove special characters
import re


# In[ ]:


# define function that removes special characters
def remove_special_characters(text):
    return re.sub("[^A-Za-z]+", " ", text).strip()


# In[ ]:


# remove special characters in the string
data_test['review_headline_wo_punctuations']= data_test['review_headline'].apply(lambda x:remove_special_characters(x))
data_test.head()


# In[ ]:


data_test[(data_test['review_headline_wo_punctuations']!= data_test['review_headline']) == True]


# #### remove stop words

# In[ ]:


import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


# In[ ]:


# define remove stopwords function
def remove_stopwords(df, col):
    s = df[col].str.lower() # stopwords in the package are all in lower case
    for word in stopwords.words('english'):
        s = s.str.replace("\\b" + word + "\\b", " ", regex=True)
#         print(word + " ", end="")
    return s.str.replace("\\s+", " ", regex=True) # .strip()

# # define function: remove word if the length = 1
# def remove_single_word(s):
#     if len(s) <= 1:
#         return ''
#     else:
#         return s

# define generate_N_grams function
def generate_N_grams(text, ngram):
    text = text.strip()
    words = text.split(" ")
#     print(words)
    if len(words) < ngram:                  
        return '_'.join(words) 
    else: 
        temp=zip(*[words[i:] for i in range(0,ngram)])
        ans=['_'.join(word) for word in temp]
        return " ".join(ans).strip()


# In[ ]:


# for word in stopwords.words('english'):
#     print(word)

Question: if text length is smaller than ngram, it will return empty string (See example below). How to include them into analyse? 
# In[ ]:


# test function
print(generate_N_grams("Horrible Gave a great trilogy a horrible ending",2))
# print(generate_N_grams("Horrible Gave a great trilogy a horrible ending",1))
# print(*ngrams("Horrible Gave a great trilogy a horrible ending".split(" "), 2))

# remove_stopwords(df_train[df_train.star_rating==5][:5], "review_headline_wo_punctuations")
# df_train[df_train.star_rating==5][:5]['review_headline_wo_punctuations'].str.replace("\\band\\b", "")
# print(generate_N_grams(" best",2)) # _best
# print(generate_N_grams("love ",2)) # love_
generate_N_grams("awesome text", 3)


# In[ ]:


# Remove stopwords 
data_test['review_headline_wo_punctuations'] = remove_stopwords(data_test, 'review_headline_wo_punctuations')


# In[ ]:


data_test.head(3).transpose()


# In[ ]:


# chec na in the data
print(data_test.review_headline.isna().sum())
print(data_test.star_rating.isna().sum())
# no na data in both columns


# In[ ]:


# only star rating 5 reviews contrain text "five stars" in review
df_train[df_train['review_headline'].str.contains("five stars")==True]['star_rating'].nunique()


# In[ ]:


stop


# #### split train/test data

# In[ ]:


y = data_test['star_rating'].values
y.shape


# In[ ]:


x = data_test['review_headline_wo_punctuations'].values
x.shape


# In[ ]:


from sklearn.model_selection import train_test_split

(x_train,x_test,y_train,y_test)=train_test_split(x,y,test_size=0.4)
print("x_train shape: ", x_train.shape)
print("y_train shape: ", y_train.shape)
print("x_test shape: ", x_test.shape)
print("y_test shape: ", y_test.shape)


# In[ ]:


df_x_train = pd.DataFrame(x_train)
df_x_train = df_x_train.rename(columns={0:'review_headline'})

df_y_train = pd.DataFrame(y_train)
df_y_train = df_y_train.rename(columns={0:'star_rating'})

df_train = pd.concat([df_x_train,df_y_train],axis=1)

df_train.head()


# In[ ]:


df_x_test = pd.DataFrame(x_test)
df_x_test = df_x_test.rename(columns={0:'review_headline'})

df_y_test = pd.DataFrame(y_test)
df_y_test = df_y_test.rename(columns={0:'star_rating'})

df_test = pd.concat([df_x_test,df_y_test],axis=1)

df_test.head()


# #### define functions for bigrams and trigrams

# In[ ]:


# define functions for bigrams and trigrams analysis
def generate_N_grams_df(df, column_y, value_y, column_x, n_grams, dict_result):
    for text in df[df[column_y] == value_y][column_x]:
        ngram = generate_N_grams(text, n_grams)
        if len(ngram) == 1 and ngram[0] == "": # skip empty string
            pass
        if len(generate_N_grams(text, n_grams)) < n_grams:
            dict_result[text]+=1
        else:
            for word in generate_N_grams(text, n_grams).split(" "):
                dict_result[word]+=1
    
    df_result = pd.DataFrame.from_dict(dict_result, orient = 'index')
    df_result = df_result.rename(columns={0:'# words'}).sort_values(by=['# words']).reset_index()
    
    return df_result


# #### create unigram

# In[ ]:


from collections import defaultdict

n_grams = 1

rating_5 = defaultdict(int)
rating_4 = defaultdict(int)
rating_3 = defaultdict(int)
rating_2 = defaultdict(int)
rating_1 = defaultdict(int)


# In[ ]:


# Create string with n-grams
# get the count of every word in both the columns of df_train and df_test dataframes where star_rating=5
for text in df_train[df_train.star_rating==5].review_headline:
    for word in generate_N_grams(text, n_grams).split(" "):
        rating_5[word]+=1


# In[ ]:


# convert dict to df
df_rating_5 = pd.DataFrame.from_dict(rating_5, orient = 'index')
df_rating_5 = df_rating_5.rename(columns={0:'# words'}).sort_values(by=['# words']).reset_index()
df_rating_5.tail(5)


# In[ ]:


# Create string with n-grams
# get the count of every word in both the columns of df_train and df_test dataframes where star_rating=1
for text in df_train[df_train.star_rating==1].review_headline:
    for word in generate_N_grams(text, n_grams).split(" "):
        rating_1[word]+=1

# convert dict to df
df_rating_1 = pd.DataFrame.from_dict(rating_1, orient = 'index')
df_rating_1 = df_rating_1.rename(columns={0:'# words'}).sort_values(by=['# words']).reset_index()
df_rating_1.tail(5)


# In[ ]:


for text in df_train[df_train.star_rating==2].review_headline:
    for word in generate_N_grams(text, n_grams).split(" "):
        rating_2[word]+=1

# convert dict to df
df_rating_2 = pd.DataFrame.from_dict(rating_2, orient = 'index')
df_rating_2 = df_rating_2.rename(columns={0:'# words'}).sort_values(by=['# words']).reset_index()
df_rating_2.tail(5)


# In[ ]:


for text in df_train[df_train.star_rating==3].review_headline:
    for word in generate_N_grams(text, n_grams).split(" "):
        rating_3[word]+=1

# convert dict to df
df_rating_3 = pd.DataFrame.from_dict(rating_3, orient = 'index')
df_rating_3 = df_rating_3.rename(columns={0:'# words'}).sort_values(by=['# words']).reset_index()
df_rating_3.tail(5)


# In[ ]:


for text in df_train[df_train.star_rating==4].review_headline:
    for word in generate_N_grams(text, n_grams).split(" "):
        rating_4[word]+=1

# convert dict to df
df_rating_4 = pd.DataFrame.from_dict(rating_4, orient = 'index')
df_rating_4 = df_rating_4.rename(columns={0:'# words'}).sort_values(by=['# words']).reset_index()
df_rating_4.tail(5)


# In[ ]:


#focus on more frequently occuring words for top and lowest ratings
#sort in DO wrt 2nd column in each of top and lowest ratings
df_rating_5['index'][-10:]


# In[ ]:


plt.figure(1,figsize=(16,4))
plt.bar(df_rating_5['index'][-10:], df_rating_5['# words'][-10:], color ='green', width = 0.4)
plt.xlabel("Words in star_rating 5 DF")
plt.ylabel("Count")
plt.title("Top 10 words in star_rating 5 DF-UNIGRAM ANALYSIS")
plt.show()


# In[ ]:


plt.figure(1,figsize=(16,4))
plt.bar(df_rating_4['index'][-10:], df_rating_4['# words'][-10:], color ='green', width = 0.4)
plt.xlabel("Words in star_rating 4 DF")
plt.ylabel("Count")
plt.title("Top 10 words in star_rating 4 DF-UNIGRAM ANALYSIS")
plt.show()


# In[ ]:


plt.figure(1,figsize=(16,4))
plt.bar(df_rating_3['index'][-10:], df_rating_3['# words'][-10:], color ='green', width = 0.4)
plt.xlabel("Words in star_rating 3 DF")
plt.ylabel("Count")
plt.title("Top 10 words in star_rating 3 DF-UNIGRAM ANALYSIS")
plt.show()


# In[ ]:


plt.figure(1,figsize=(16,4))
plt.bar(df_rating_2['index'][-10:], df_rating_2['# words'][-10:], color ='green', width = 0.4)
plt.xlabel("Words in star_rating 2 DF")
plt.ylabel("Count")
plt.title("Top 10 words in star_rating 2 DF-UNIGRAM ANALYSIS")
plt.show()


# In[ ]:


plt.figure(1,figsize=(16,4))
plt.bar(df_rating_1['index'][-10:], df_rating_1['# words'][-10:], color ='green', width = 0.4)
plt.xlabel("Words in star_rating 1 DF")
plt.ylabel("Count")
plt.title("Top 10 words in star_rating 1 DF-UNIGRAM ANALYSIS")
plt.show()

The result of unigram is similar to wordcloud 
# In[ ]:


# Alternative directly use ngrams package --> only works with single string
# get individual words
# tokenized = text.split()

# for text in df_train[df_train.star_rating==5].review_headline_wo_punctuations:
#     print(text)
#     tokenized = text.split()
#     print(tokenized)
#     for word in tokenized:
#         rating_5_v2['word']
#         print(ngrams(tokenized, 2))
#     stop


# In[ ]:


# test = ngrams(tokenized, 2)
# collections.Counter(test)


# #### create bigrams

# In[ ]:


from collections import defaultdict

rating_5 = defaultdict(int)
rating_4 = defaultdict(int)
rating_3 = defaultdict(int)
rating_2 = defaultdict(int)
rating_1 = defaultdict(int)


# In[ ]:


df_rating_5 = generate_N_grams_df(df = data_test, column_y = "star_rating", value_y = 5, 
                    column_x = "review_headline", n_grams = 2, 
                    dict_result = rating_5)


# In[ ]:


df_rating_5.tail(10)


# In[ ]:


plt.figure(1,figsize=(16,4))
plt.bar(df_rating_5['index'][-10:], df_rating_5['# words'][-10:], color ='green', width = 0.4)
plt.xlabel("Words in star_rating 5 DF")
plt.ylabel("Count")
plt.title("Top 10 words in star_rating 5 DF-UNIGRAM ANALYSIS")
plt.show()


# In[ ]:


df_rating_4 = generate_N_grams_df(df = data_test, column_y = "star_rating", value_y = 4, 
                    column_x = "review_headline", n_grams = 2, 
                    dict_result = rating_4)


# In[ ]:


df_rating_4.tail(10)


# In[ ]:


plt.figure(1,figsize=(16,4))
plt.bar(df_rating_4['index'][-10:], df_rating_4['# words'][-10:], color ='green', width = 0.4)
plt.xlabel("Words in star_rating 4 DF")
plt.ylabel("Count")
plt.title("Top 10 words in star_rating 4 DF-UNIGRAM ANALYSIS")
plt.show()


# In[ ]:


df_rating_3 = generate_N_grams_df(df = df_train, column_y = "star_rating", value_y = 3, 
                    column_x = "review_headline", n_grams = 2, 
                    dict_result = rating_3)


# In[ ]:


df_rating_3.tail(10)


# In[ ]:


plt.figure(1,figsize=(16,4))
plt.bar(df_rating_3['index'][-10:], df_rating_3['# words'][-10:], color ='green', width = 0.4)
plt.xlabel("Words in star_rating 3 DF")
plt.ylabel("Count")
plt.title("Top 10 words in star_rating 3 DF-UNIGRAM ANALYSIS")
plt.show()


# In[ ]:


df_rating_2 = generate_N_grams_df(df = df_train, column_y = "star_rating", value_y = 2, 
                    column_x = "review_headline", n_grams = 2, 
                    dict_result = rating_2)


# In[ ]:


plt.figure(1,figsize=(16,4))
plt.bar(df_rating_2['index'][-10:], df_rating_2['# words'][-10:], color ='green', width = 0.4)
plt.xlabel("Words in star_rating 2 DF")
plt.ylabel("Count")
plt.title("Top 10 words in star_rating 2 DF-UNIGRAM ANALYSIS")
plt.show()


# In[ ]:


df_rating_1 = generate_N_grams_df(df = df_train[df_train['star_rating'] == 1], 
                                  column_y = "star_rating", value_y = 1, 
                                  column_x = "review_headline", n_grams = 2, 
                                  dict_result = rating_1)


# In[ ]:


plt.figure(1,figsize=(16,4))
plt.bar(df_rating_1['index'][-10:], df_rating_1['# words'][-10:], color ='green', width = 0.4)
plt.xlabel("Words in star_rating 1 DF")
plt.ylabel("Count")
plt.title("Top 10 words in star_rating 1 DF-UNIGRAM ANALYSIS")
plt.show()


# #### create trigrams

# In[ ]:


from collections import defaultdict

rating_5 = defaultdict(int)
rating_4 = defaultdict(int)
rating_3 = defaultdict(int)
rating_2 = defaultdict(int)
rating_1 = defaultdict(int)


# In[ ]:


df_rating_5 = generate_N_grams_df(df = data_test, column_y = "star_rating", value_y = 5, 
                    column_x = "review_headline", n_grams = 3, 
                    dict_result = rating_5)


# In[ ]:


df_rating_4 = generate_N_grams_df(df = data_test, column_y = "star_rating", value_y = 4, 
                    column_x = "review_headline", n_grams = 3, 
                    dict_result = rating_4)


# In[ ]:


df_rating_3 = generate_N_grams_df(df = data_test, column_y = "star_rating", value_y = 3, 
                    column_x = "review_headline_wo_punctuations", n_grams = 3, 
                    dict_result = rating_3)


# In[ ]:


df_rating_2 = generate_N_grams_df(df = data_test, column_y = "star_rating", value_y = 2, 
                    column_x = "review_headline_wo_punctuations", n_grams = 3, 
                    dict_result = rating_2)


# In[ ]:


df_rating_1 = generate_N_grams_df(df = data_test, column_y = "star_rating", value_y = 1, 
                    column_x = "review_headline_wo_punctuations", n_grams = 3, 
                    dict_result = rating_1)


# In[ ]:


plt.figure(1,figsize=(16,4))
plt.bar(df_rating_5['index'][-10:], df_rating_5['# words'][-10:], color ='green', width = 0.4)
plt.xlabel("Words in star_rating 5 DF")
plt.ylabel("Count")
plt.title("Top 10 words in star_rating 5 DF-TRIGRAM ANALYSIS")
plt.show()


# In[ ]:


plt.figure(1,figsize=(16,4))
plt.bar(df_rating_4['index'][-10:], df_rating_4['# words'][-10:], color ='green', width = 0.4)
plt.xlabel("Words in star_rating 4 DF")
plt.ylabel("Count")
plt.title("Top 10 words in star_rating 4 DF-TRIGRAM ANALYSIS")
plt.show()


# In[ ]:


plt.figure(1,figsize=(16,4))
plt.bar(df_rating_3['index'][-10:], df_rating_3['# words'][-10:], color ='green', width = 0.4)
plt.xlabel("Words in star_rating 3 DF")
plt.ylabel("Count")
plt.title("Top 10 words in star_rating 3 DF-TRIGRAM ANALYSIS")
plt.show()


# In[ ]:


df_rating_3[(df_rating_3['index'].str.strip() == '') == True]


# In[ ]:


df_rating_3 = df_rating_3[(df_rating_3['index'].str.strip() == '') == False]


# In[ ]:


df_rating_3.tail(10)


# In[ ]:


# in the top 10 contains empty space -- > drop it (need to modify the n_gram function)
plt.figure(1,figsize=(16,4))
plt.bar(df_rating_3['index'][-10:], df_rating_3['# words'][-10:], color ='green', width = 0.4)
plt.xlabel("Words in star_rating 3 DF")
plt.ylabel("Count")
plt.title("Top 10 words in star_rating 3 DF-TRIGRAM ANALYSIS")
plt.show()


# In[ ]:


plt.figure(1,figsize=(16,4))
plt.bar(df_rating_2['index'][-10:], df_rating_2['# words'][-10:], color ='green', width = 0.4)
plt.xlabel("Words in star_rating 2 DF")
plt.ylabel("Count")
plt.title("Top 10 words in star_rating 2 DF-TRIGRAM ANALYSIS")
plt.show()


# In[ ]:


df_rating_2 = df_rating_2[(df_rating_2['index'].str.strip() == '') == False]
df_rating_2.tail(10)


# In[ ]:


plt.figure(1,figsize=(16,4))
plt.bar(df_rating_2['index'][-10:], df_rating_2['# words'][-10:], color ='green', width = 0.4)
plt.xlabel("Words in star_rating 2 DF")
plt.ylabel("Count")
plt.title("Top 10 words in star_rating 2 DF-TRIGRAM ANALYSIS")
plt.show()


# In[ ]:


plt.figure(1,figsize=(16,4))
plt.bar(df_rating_1['index'][-10:], df_rating_1['# words'][-10:], color ='green', width = 0.4)
plt.xlabel("Words in star_rating 1 DF")
plt.ylabel("Count")
plt.title("Top 10 words in star_rating 1 DF-TRIGRAM ANALYSIS")
plt.show()


# In[ ]:


df_rating_1 = df_rating_1[(df_rating_1['index'].str.strip() == '') == False]
df_rating_1.tail(10)


# In[ ]:


plt.figure(1,figsize=(16,4))
plt.bar(df_rating_1['index'][-10:], df_rating_1['# words'][-10:], color ='green', width = 0.4)
plt.xlabel("Words in star_rating 1 DF")
plt.ylabel("Count")
plt.title("Top 10 words in star_rating 1 DF-TRIGRAM ANALYSIS")
plt.show()


# In[ ]:


STOP


# #### To-Do <br>
# the result of trigram makes most sense. <br>
# As next step, a matrix for further ML-analysis (for example naive bayes [Link](https://towardsdatascience.com/text-classification-using-naive-bayes-theory-a-working-example-2ef4b7eb7d5a))
# 
# 

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


df_test.head()


# In[ ]:


df_test['review_headline_ngrams'] = df_test['review_headline_wo_punctuations'].apply(lambda x: generate_N_grams(x, 3))


# In[ ]:


df_test.head()


# In[ ]:


df_test_final = df_test[['star_rating', 'review_headline_ngrams']]


# In[ ]:


corpus = df_test_final.review_headline_ngrams.values.tolist()
corpus[0:5]


# In[ ]:


vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names())


# In[ ]:


len(vectorizer.get_feature_names())


# In[ ]:


# define functions for bigrams and trigrams analysis
def generate_N_grams_df_all(df, column_x, n_grams, dict_result):
    for text in df[column_x]:
        ngram = generate_N_grams(text, n_grams)
        if len(ngram) == 1 and ngram[0] == "": # skip empty string
            pass
        if len(generate_N_grams(text, n_grams)) < n_grams:
            dict_result[text]+=1
        else:
            for word in generate_N_grams(text, n_grams).split(" "):
                dict_result[word]+=1
    
    df_result = pd.DataFrame.from_dict(dict_result, orient = 'index')
    df_result = df_result.rename(columns={0:'# words'}).sort_values(by=['# words']).reset_index()
    
    return df_result 


# In[ ]:


dict_result_all = defaultdict(int)
df_result_all = generate_N_grams_df_all(df = df_test, column_x = "review_headline_wo_punctuations", n_grams = 3, dict_result = dict_result_all)


# In[ ]:


df_result_all.describe()


# In[ ]:


df_result_all[df_result_all['# words']<=5].count()


# In[ ]:


# drop the tokens which appear fewer than &including 5 times
df_result_all = df_result_all[df_result_all['# words'] > 5]


# In[ ]:


df_result_all.describe()


# In[ ]:


df_result_all.head()


# In[ ]:


feature_filter = df_result_all['index'].values.tolist()


# In[ ]:


vectorizer = CountVectorizer(vocabulary = feature_filter)
X = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names()[0:5])
print(len(vectorizer.get_feature_names()))


# In[ ]:


# problem > too many features --> not enough memory --> drop featurew with number of appearance in the db fewer than 5 times
print(X.toarray())


# In[ ]:


X.shape


# In[ ]:


df_test_final.shape


# In[ ]:


# # test matrix
# for row in X.toarray():
#     print(sum(row))


# In[ ]:


# count = 0
# for i in X.toarray()[0]:
#     count+=1
#     if i != 0:
#         print(count)
#         break


# In[ ]:


# feature_filter[2302]


# ##### train model

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score


# In[ ]:


X = df_test_final


# In[ ]:


# Build the model
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model using the training data
model.fit(X, df_test_final.star_rating)


# In[ ]:


# test
from sklearn.datasets import fetch_20newsgroups
data = fetch_20newsgroups()
text_categories = data.target_names


# In[ ]:


data


# ## Reference
# 1. https://www.analyticsvidhya.com/blog/2021/09/what-are-n-grams-and-how-to-implement-them-in-python/ <br>
# 2. https://www.kaggle.com/code/rtatman/tutorial-getting-n-grams/notebook <br>
# 3. https://towardsdatascience.com/text-classification-using-naive-bayes-theory-a-working-example-2ef4b7eb7d5a

# In[ ]:




