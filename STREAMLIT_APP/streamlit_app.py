#########################################################################
### define packages used in the app
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd 
import numpy as np
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import spacy
from joblib import dump, load
from sklearn.feature_extraction.text import CountVectorizer
from util import prepare_data,draw_correlation_with_target, get_score ,get_predictions
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import os
from os.path import dirname
from PIL import Image
# plotly
#import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff


#streamlit run STREAMLIT_APP\streamlit_app.py
#########################################################################
### define the project path
path = dirname(os.getcwd()) # path of the parent folder of repo, where data folder is located
path_repo = os.getcwd() # path of the repo-folder. STREAMLT_APP is a subfolder of the repo-folder

#########################################################################
### define side bar > table of content
st.sidebar.title("Contents")
pages =["Introduction","Data Exploration and cleansing", "Data Pre-processing, Modelling and Interpretation", "Visualization", "Interactive part - Get the Sentiment", "Conclusion"] # "Data Visualizations"
page = st.sidebar.radio("Click the page",options = pages)

#########################################################################
### Part introduction
if page == pages[0]: 
    st.title('Supply Chain- Customer Satisfaction')
    st.caption("Amazon US Customer Reviews -- Video Games ")
    st.caption("Author: Ipsita Ranjan, Philipp Reber, Sebastian Willutzky, Ling Zhu")
    st.text("\n") 

    st.header("Introduction")

    st.write("The purpose of this study is to investigate how companies can conduct sentiment analysis based on Amazon.com reviews to gain more insights into customer experience. The dataset used in this study is the Amazon US Customer Reviews Dataset. Here is the [link](https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset) towards the original dataset"
    )
    st.write("The reviews are split into two parts, there is a star-rating system and customer review text. The star rating system is informative; however, it provides only a quantitative analysis on whether a product is popular or not. The review text provides qualitative information on why the product was popular or not."
    )
    st.write("Once a sentiment analysis is completed and a model is trained, companies that sell products on Amazon or sell similar products on other platforms can gain more understanding on top-rated products, what customers value, maintain positive engagement and improve neutral/negative experiences. This can encourage innovative product development and enhanced customer service."
    )
    st.write("The objective of this study is to build a model that can predict the star rating of reviews with a high accuracy and thus determine if the review is associated with a positive or negative sentiment. As mentioned above, the study will involve sentiment analysis and the process relies on machine learning (ML) algorithms and natural language processing (NLP)."
    )
    #st.write(path_repo)
    
    image = Image.open(path_repo + "/STREAMLIT_APP/pic/smilelys.jpg")
    st.image(image)

#########################################################################
### part Data exploration and cleansing   
if page  == pages[1]: 
    st.header("Data exploration and cleansing")

    st.subheader("Explore the raw data")
    st.write("The dataset used is the Amazon US Customer Reviews found on Kaggle.com [link](https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset). The data is freely available and is public. The content of the dataset is as follows: "
    )
    df = pd.read_excel(path_repo + "/STREAMLIT_APP/input/dataexploration_datacolumns.xlsx")
    st.dataframe(df)
    st.write("The collection of reviews were written in the Amazon.com marketplace and associated metadata from 1995 until 2015. Examples of products in the dataset include but are not limited to apparel, books, furniture, musical instruments, toys etc. "
    )
    st.write("For the purpose of this study, the dataset based on reviews for Hardcopy and Digital videogames were used. It is approximately 1,2 GB in size with 15 columns and 1.924.992 entries. You can find the first 5 lines of the raw data below: "
    )
    # data_raw = pd.read_csv(path + '/data/raw/data_raw.csv')
    # st.dataframe(data_raw.head())
    st.write("The explaining variables are review_headline and review_body since it is the text in these columns that the model will analyse and ultimately draw conclusions from. The target variable is star_rating since it provides a clear output on the sentiment (i.e., positive, neutral, or negative along 5 rating classes) of the review."
    )

    st.subheader("Data cleansing")
    st.write("There was a low amount of missing data therefore what was found in the dataset was dropped. The column marketplace was dropped since the dataset only included reviews from the US Marketplace therefore deemed irrelevant. The customer_id column had a high number of duplicate entries since a customer with a unique customer id most likely ordered multiple items There were no duplicate entries found in the review_id column since the id is truly unique to the review made. After cleansing, 1.924.825 from 1.924.992 entries are left.")

    # df_clean = pd.read_csv(path + '/data/raw/data_firstclean.csv') # data is not loaded due to long loading time

    # pic "Number of video games & digital video games
    image = Image.open(path_repo + "/STREAMLIT_APP/pic/fig_no_video_digital_games.jpg")
    st.image(image)

    # pic "Number of reviews per star rating"
    image = Image.open(path_repo + "/STREAMLIT_APP/pic/fig_no_reviews_per_rating.jpg")
    st.image(image)

    # pic "Number of reviews by reivew date"
    image = Image.open(path_repo + "/STREAMLIT_APP/pic/fig_no_reviews_by_date.jpg")
    st.image(image)

    st.write("In general, the dataset is imbalanced, since there are more 5 star reviews compared to the other ratings. This could be an issue with the machine learning classifiers since the algorithms expect an equal number of entries/examples per class to perform adequately. ")
        
#########################################################################
### part Modelling    
if page == pages[2]:
    st.header("Data Pre-processing, Modelling and Interpretation")
    st.write("The following base level pre-processing steps were completed during the data analysis stage:")
    st.write("1. Tokenization")
    st.write("2. Lower Casing")
    st.write("3. Removing Punctuation")
    st.write("4. Standard Stop Words Removal")
    st.write("To ensure a high performing model, additional pre-processing steps are required. It is important to note that in the upcoming chapters, if the nomenclature states basic pre-processing, the above steps were taken, and if it states advanced pre-processing, the below steps were taken.")

    st.subheader("Stemming and Lemmatization")
    st.write("Stemming and lemmatization are algorithms used to normalize text and prepare words for further processing. For example, in a review, the model should be able to acknowledge that the words 'like' and 'liked' are the same word in different tenses. Therefore, it is necessary to reduce both words to a common root word, which can be done by stemming or lemmatization. It should be noted that lemmatization is a development of stemming, so it is logical to first preprocess through stemming.")
    st.write("Stemming removes suffixes from words and provides a word stem. To continue with the above example, the words 'likes', 'likely', and 'liked' all result in the common word stem 'like'. The algorithm used was the snowball stemmer, also known as the Porter2 stemming algorithm.")
    st.write("Next, words were lemmatized, which is the process of grouping together the different inflected forms of a word so they can be analyzed as a single item. This gives context to the words. Lemmatization is also a prerequisite for POS-Tagging, which will be discussed later. The WordNet Lemmatizer from the Python library was used.")

    st.subheader("Customized Stop Words")
    st.write("To further simplify the model input and remove excessive noise, the stop words that were removed were customized. Words analyzed through the word cloud (Figure 5.0) were added above and beyond the standard English stop words and subsequently removed.")
    image = Image.open(path_repo + "/STREAMLIT_APP/pic/customized_stopwords.png")
    st.image(image, caption="Figure 5.0: Customized Stop Word Cloud")

    st.subheader("POS-Tagging")
    st.write("Part-of-speech (POS) tagging is a preprocessing technique that refers to categorizing words in a text in correspondence with a particular part of speech. It considers the definition of the word and its context. The POS tags are used to describe the lexical terms in the English language: noun, pronoun, verb, adjective, adverb, preposition, conjunction, and interjection. During the experimental modeling phase, the accuracy of models was compared with and without POS-Tagging to see what kind of impact this pre-processing step has.")

    st.subheader("NER")
    st.write("Named Entity Recognition (NER) is another form of natural language processing and specifically a subset of artificial intelligence. The NER technique is a two-step process. First, it detects a named entity, and second, it categorizes the entity. This was implemented using the open-source Spacy library.")

    st.subheader("Feature Extraction")
    st.write("The integral component of this sentiment analysis is that natural language processing requires computers to understand human language. Therefore, it is a prerequisite for the textual data to be converted into a numerical value prior to feeding it into a machine learning model.")
    st.write("This will be analyzed as a classification problem. Classification is a supervised machine learning method where the method tries to predict the correct label of a given input data. The data used and analyzed above comes with numerical rating data, ranging from 1 to 5. These numerical indicators will be used as labels that represent the sentiment of the review text. Thus, this problem will be viewed as a multi-class classification problem.")
    st.write("The three feature extraction techniques analyzed were CountVectorizer, TF-IDF, and Word2Vec.")

    st.subheader("CountVectorizer")
    st.write("CountVectorizer converts a collection of text documents to a matrix of token counts, in other words, it counts the occurrences of tokens in each document. Each unique word is represented by a column of the matrix, and each text sample is a row in the matrix.")

    st.subheader("TF-IDF")
    st.write("Term Frequency-Inverse Document Frequency (TF-IDF) is a statistical measure that evaluates how relevant a word is to a document in a collection of documents. CountVectorizer simply counts the number of times a word appears in a document, whereas TF-IDF considers not only how many times a word appears in a document but also how important that word is in the whole corpus. Essentially, TF-IDF is an extension of CountVectorizer, and therefore it is a good idea to try both.")

    st.subheader("Word2Vec")
    st.write("Word embeddings are words mapped to real numbers of vectors such that it can capture the semantic meanings of the words. TF-IDF does not capture the meaning between the words; it considers the words as separate features.")
    st.write("Word2Vec was created by a team of researchers led by Tomas Mikilov at Google. It is an unsupervised learning algorithm, and it works by predicting its context words by applying a two-layer neural network. Word2Vec vectors are generated for each review in the train data by traversing the X_train dataset. Prior to vectorizing and training the dataset, the dataset is balanced by undersampling the minority classes. There are two approaches when implementing Word2Vec: there is a pre-trained model, or the model can be self-trained.")
    st.write("Google published a pre-trained Word2Vec model that is trained with a Google News dataset that is approximately 100 billion words. The model contains 300-dimensional vectors for 3 million words and phrases. The pre-trained model is downloaded and loaded using the gensim package.")

    st.subheader("Modelling")
    st.write("The modelling phase was split into two phases.")
    st.write("The first model exploration tested various models with basic pre-processing steps. All pre-processing steps are outlined in Chapter 5.0 in more detail. In summary, the basic pre-processing steps included the following:")
    st.write("1. Tokenization")
    st.write("2. Lower Casing")
    st.write("3. Removing Punctuation")
    st.write("4. Standard Stop Words Removal")
    st.write("After the first model exploration, it was concluded that further pre-processing steps were necessary and other vectorization methods will also be experimented. In summary, the advanced pre-processing steps included the following:")
    st.write("1. Stemming and Lemmatization")
    st.write("2. Customized Stop Words")
    st.write("3. POS-Tagging")
    st.write("4. NER")
    st.write("Furthermore, labels were recoded to have three classes instead of five. The three classes will be positive, neutral, and negative. It is noted that even though the numerical accuracy will increase, there is less information about the model. The increased accuracy is a mathematical consequence of having fewer neighboring classes.")
    st.write("In the second exploration specific stop-words were manually updated from the reviews and the Word2Vec tokenization method was attempted. During the first exploration it was noted that logistic regression was the only model that had the best ratio of performance and time to run. Logistic regression had an average run time of 9 minutes. Therefore, the logistic regression model was the only one tested during the second exploration. CPU run time will also be measured during the second exploration to justify the cost of running logistic regression over other models.")
    
    st.subheader("Logistic Regression")
    st.write("Logistic regression was the first algorithm tested after the data was vectorized due to cost. In this case, cost is measured by time. Logistic regression tends to be a standard baseline model due to its simplicity and aids in the initial exploratory data analysis. Results shown in the Table 7.0 below for models where advanced pre-processing steps were used, the logistic regression model was already fine-tuned using gridsearch.")
    st.write("Figure 7.0 illustrates the hyperparameters obtained through the gridsearch.")
    image = Image.open(path_repo +"/STREAMLIT_APP/pic/Gridsearch.png")
    st.image(image, caption="Figure 7.0: Hyperparameters obtained through gridsearch")
    st.write("From the above table, the model that had the highest accuracy score (84%) had 3 classes as opposed to 5 and TF-IDF was used as the feature extraction technique. The advanced pre-processing steps include POS-Tagging was required to develop a strong model. According to the research, Word2Vec should have had strong results as well, however the results are worse than TF-IDF. It is assumed this is because of a lack of training data.")
    image = Image.open(path_repo +"/STREAMLIT_APP/pic/LR_results.png")
    st.image(image, caption="Table 7.0: Summary of Logistic Regression Results")
    st.write("Figure 7.1 shows this model interpretability. (0 stands for negative/rating 1-2, 1 stands for neutral/rating 3 and 2 stands for positive/rating 4-5).")
    image = Image.open(path_repo +"/STREAMLIT_APP/pic/best_model_interpretability.png")
    st.image(image, caption="Figure 7.1: Best model interpretability.")

    st.subheader("Decision Tree and Random Forest")
    st.write("The two algorithms tested on a CountVectorized dataset with basic pre-processing steps were Decision Tree and Random Forest. Decision tree is a single model that makes predications based on a series of if-then rules and Random Forest is an ensemble learning technique. Unfortunately, due to computational limitations the dataset had to be reduced to 10,000 lines to produce any results. The results were poor at accuracy rates less than 30%.")
    image = Image.open(path_repo +"/STREAMLIT_APP/pic/DT_summary.png")
    st.image(image, caption="Table 7.1: Summary of Decision Tree and Random Forest results")

    st.subheader("XG and Cat Boost")
    st.write("Extreme Gradient Boosting, also known as XG-Boost is a top gradient boosting framework. The algorithm uses regression trees for the base learner and is powerful due to its accuracy, efficiency, and cost. To further leverage the power of XG-Boost the hyperparameters were tuned.")
    st.write("CatBoost is a depth-wise gradient boosting library. It grows a balanced tree using oblivion decision trees. This is a powerful algorithm due to its ability to handle categorical features natively, it reduces parameter tuning time by providing good results with default parameters and it can be used for both regression and classification problems.")
    st.write("Both these algorithms were tested with the basic pre-processing steps and therefore did not include POS-Tagging.")
    image = Image.open(path_repo +"/STREAMLIT_APP/pic/XG_CB_summary.png")
    st.image(image, caption="Table 7.2: Summary of XG and Cat Boost Results")

    st.subheader("Model Interpretability")
    st.write("A machine learning algorithm’s interpretability refers to how easy it is for humans to understand the processes the model uses to arrive at a result. Model interpretability is extremely important when the outcome of a model is being used for a business decision and what features the model did consider. In this case, once a model is trained, companies that sell products on Amazon or sell similar products on other platforms can gain more understanding on top-rated products, what customers value, maintain positive engagement and improve neutral/negative experiences. This can encourage innovative product development and enhanced customer service. Model interpretability was done on the logistic regression model with TF-IDF and pos-tagging as baseline")
    
    st.subheader("N-Gram Analysis")
    st.write("For every n-gram in the vocabulary, the absolute value of the difference of the corresponding coefficients of the best and the worst class was considered to identify the most important n-grams. The resulting n-grams intuitively are very reasonable. Figure 8.0 a & b outline the most positive and negative words.")
    image = Image.open(path_repo +"/STREAMLIT_APP/pic/important_words.png")
    st.image(image, caption="Figure 8.0a: Most important positive words - Figure 8.0b: Most important negative words")

    st.subheader("SHAP analysis")
    st.write("The Shapley Additive exPlanations (SHAP) is inspired by several methods on model interpretability. It is proposed that SHAP value is a united approach to explaining the output of any machine learning model [7]. The benefits of using SHAP for model interpretability are the following:")
    st.write("1. Global Interpretability: the collective SHAP values can show how much each predictor contributes, either positively or negatively, to the target variable. This is like the variable importance plot, but it can show the positive or negative relationship for each variable with the target.")
    st.write("2. Local Interpretability: each observation gets its own set of SHAP values. This greatly increases its transparency. This explains why a case receives its prediction and the contributions of the predictors. Traditional variable importance algorithms only show the results across the entire population but not on each individual case. The local interpretability enables us to pinpoint and contrast the impacts of the factors")
    st. write("The best and worst predictions are illustrated the figures below.")
    image = Image.open(path_repo +"/STREAMLIT_APP/pic/SHAP.png")
    st.image(image)


#########################################################################
### part Visualization (plotly)
if page == pages[3]:

    st.header("Visualization")

    # import data
    st.write("Loading pre-processed data from "+ path_repo)  
    try:
        df_in = pd.read_pickle(path + "\data\processed\data_en3_3sentiments.pickle")
        df_in = df_in.iloc[0:10000]
        st.write("Data loaded")
        st.write("==============================================================")
    except:
        st.write("No data found!")
        st.write("==============================================================")
   
    st.subheader("Data exploration after pre-processing")
    # graph: number of review per rating class
    df_in["myear"] = df_in.review_date.apply(lambda x: datetime.strptime(x, "%Y-%m-%d").strftime("%Y-%m"))
    df_in["star_rating"] = df_in["star_rating"].apply(lambda x: int(x))

    fig = px.line(df_in.groupby(["myear","star_rating"])["review_id"].count().reset_index().sort_values("myear"), x='myear', y="review_id", color="star_rating", title = "Reviews per rating class")
    st.plotly_chart(fig)

    # graph: distribution of word length and rating class
    # st.write("Loading graph: ")
    df_in['word_length'] = df_in['lem_pos_ner_rem'].str.split().str.len()

    @st.cache_data
    def sentiment(row):
        if (row['star_rating'] == 5) | (row['star_rating'] == 4):
            return "Positive"
        elif (row['star_rating'] == 1) | (row['star_rating'] == 2):
            return "Negative"
        elif row['star_rating'] == 3:
            return "Inbetween"
        else:
            return "Undefined"

    st.write("Converting 5 stars rating into 3 groups: ")
    df_in['star_sentiment'] = df_in.apply(sentiment,axis =1)

    positive = df_in[df_in['star_sentiment']=='Positive']['word_length']
    negative = df_in[df_in['star_sentiment']=='Negative']['word_length']
    neutral = df_in[df_in['star_sentiment']=='Inbetween']['word_length']
    st.write("Finished converting! ")

    histdata = [positive, neutral, negative]
    # st.dataframe(histdata)
    group_labels = ['Positifve', 'Neutral', 'Negative']

    # Create distplot with custom bin_size
    fig = ff.create_distplot(histdata, group_labels, show_hist=False)
    fig.update_layout(title_text = "Distribution plot of text length and star sentiment")
    st.plotly_chart(fig)
    # try:
    #     st.plotly_chart(fig)          
    #     st.write("Plotly-graph loaded")
    #     st.write("==============================================================") 
    # except:
    #     st.write("Plotly-graph failed to load!")
    #     st.write("==============================================================")

    # graph: interactive wordcloud
    st.subheader("Word Cloud")
    st.write("Loading graph: ")
    processed_review_string = df_in.groupby("star_rating").aggregate({"processed_reviews":lambda x: " ".join(x)})
    def wc_for_rating(rating):
        wordcloud = WordCloud(collocations=True).generate(processed_review_string.loc[rating][0][1:5000000].replace("one", "")\
                                                          .replace("use", "").replace(" br ", " ").replace("car", "").replace("work", "")\
                                                            .replace("game", "").replace("br", " ").replace("play", " ").replace("control", " ")\
                                                                .replace("even", " "))
        # Display the generated image:
        return wordcloud
    
        #plt.imshow(wordcloud, interpolation='bilinear')
        #plt.axis("off")
        #plt.show()

    wci = st.text_input("Rating Class base")    

    if wci:
        fig, ax = plt.subplots(figsize = (12, 8))
        ax.imshow(wc_for_rating(int(wci)), interpolation = "bilinear")
        plt.axis("off")
        st.pyplot(fig)

    wci2 = st.text_input("Rating Class comparison")    

    if wci2:
        fig, ax = plt.subplots(figsize = (12, 8))
        ax.imshow(wc_for_rating(int(wci2)), interpolation = "bilinear")
        plt.axis("off")
        st.pyplot(fig)

#########################################################################
### part interactive part    

@st.cache_data
def load_spacy(mdl):
    nlp = spacy.load(mdl)
    return nlp

nlp = load_spacy("en_core_web_lg")

@st.cache_data
def lemmatize_and_pos_tag(review):
    doc = nlp(review)
    lst = []
    for tok in doc:
        if (tok.pos != 97 ) and (tok.pos_ != "SPACE") and (tok.is_alpha or tok.pos_== "PART") and (tok.ent_type_ == ''):
           lst.append(tok.lemma_ + "_" + tok.pos_)
    return " ".join(lst)

@st.cache_data
def load_pipeline(location):
    pipeline=load(location)
    return pipeline

if page == pages[4]:
    st.header("Interactive part - Get the sentiment")

    st.write("Loading model from "+ path_repo)    
    try:
        pipeline = load_pipeline(path_repo + '\logreg_model.joblib')
        st.write("Model loaded")
        st.write("==============================================================")
    except:
        st.write("No model found!")
        st.write("==============================================================")

    st.write("Please enter a review below, for which you want sentiment to be detected")
    input_review = st.text_input("Enter Review here")
    st.write("==============================================================")

    if input_review:
        st.write("You entered: ")
        st.write(input_review)
        st.write("==============================================================")

        st.write("We have seperated the review into tokens, and added a POS Tag:")
        lpos = lemmatize_and_pos_tag(input_review)
        st.write(lpos)
        st.write("==============================================================")

        st.write(" ")
        st.write("We will now build n-grams and predict the sentiment")
        sentiment = pipeline.predict([lpos])
        st.write("Sentiment class (1 - very bad to 5 - very good): - " + str(sentiment[0]) + " - ")
        sentiment_p = pipeline.predict_proba([lpos])
        st.write("Probality for this class: " + str(np.round(100*sentiment_p[0][sentiment[0]-1],1)) + "%")        
        st.write("==============================================================")

        st.write("All sentiment class probabilities:")
        prframe = pd.DataFrame({"Class Probability" : sentiment_p[0]})
        prframe["Sentiment Class"] = [1,2,3,4,5]
        st.bar_chart(data=prframe, x= "Sentiment Class", y="Class Probability",  width=0, height=0, use_container_width=True)              
        st.write("==============================================================")
        
        st.write("Loading impacting n-grams")

        tok_coef = pd.DataFrame(
            {
                "token": pipeline["vect"].get_feature_names_out()
                , "CTR_Class_1": pipeline["clf"].coef_[0,:]
                , "CTR_Class_2": pipeline["clf"].coef_[1,:]
                , "CTR_Class_3": pipeline["clf"].coef_[2,:]
                , "CTR_Class_4": pipeline["clf"].coef_[3,:]
                , "CTR_Class_5": pipeline["clf"].coef_[4,:]
            }
        )

        tok_coef = tok_coef.set_index("token")

        b = CountVectorizer(ngram_range=(1, 3))#pipeline["vect"]
        b.fit([lpos.lower()])
        all_toks = b.get_feature_names_out()

        rel_toks = []
        for x in all_toks:
            if x in pipeline["vect"].get_feature_names_out():
                rel_toks.append(x)

        st.write("Top 10 n-grams with highest impact on sentiment prediction")
        coefs = tok_coef[["CTR_Class_"+ str(sentiment[0])]].loc[rel_toks]
        coefs["abs_val"] = np.abs(coefs)

        coefs=coefs.sort_values(by = "abs_val", ascending=False)
        st.write(coefs[["CTR_Class_"+ str(sentiment[0])]].head(10))

        st.write("==============================================================")
        st.write("Now let's have a look at all individual classes, and which tokens are contributing most to their probabilities")
        st.write("==============================================================")
        st.write("Top 10 n-grams for all classes")

        for x in [1,2,3,4,5]:
            coefs = tok_coef[["CTR_Class_"+ str(x)]].loc[rel_toks]
            coefs["abs_val"] = np.abs(coefs)
            coefs=coefs.sort_values(by = "abs_val", ascending=False)
            st.write(coefs[["CTR_Class_"+ str(x)]].head(10))
            #st.write("Probability of this class")
            st.write(np.sum(coefs[["CTR_Class_"+ str(x)]]))
            st.write("==============================================================")

###########################################################
### part conclusion 
if page == pages[4]:
    st.header("Conclusion")
    st.write("This sentiment analysis projects’ most significant constraint was computational power. This limitation affected all members of the team. Since the modelling portion of the project was so computationally expensive, the dataset had to be significantly reduced. This reduction affected the accuracy of the models analyzed. ")

    st.write("Other challenges in a sentiment analysis are tone and polarity. Tone is difficult interpret and with large amounts of data, it can be difficult to differentiate objective and subjective tone. Polarity was also a challenge when analyzing the reviews. Words such as “love” and “hate” are clear in their intention, however a statement such as “not so bad” is harder to determine whether it is a positive or negative review.")

    st.write("The final challenge was the dataset itself. Amazon reviews is a five-star system and the multiclassification approach needed a complex pre-processing method and required a neutral label as well. Therefore, to simplify the problem, the labels were recoded, this changed the scope of the project. After reencoding the 5 starts labels into positive, negative and neutral, the results get improved. ")