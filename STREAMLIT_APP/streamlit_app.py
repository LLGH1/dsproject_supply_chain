#########################################################################
### define packages used in the app
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd 
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from util import prepare_data,draw_correlation_with_target, get_score ,get_predictions

import os
from os.path import dirname

#########################################################################
### define the project path
path = dirname(os.getcwd()) # path of the parent folder of repo, where data folder is located
path_repo = os.getcwd() # path of the repo-folder. STREAMLT_APP is a subfolder of the repo-folder

#########################################################################
### define side bar > table of content
st.sidebar.title("Contents")
pages =["Introduction","Data Exploration and cleansing","Data Visualizations", "Modelling and Intrepretation", "Interactive part - Get the Sentiment", "Conclusion"]
page = st.sidebar.radio("Click the page",options = pages)

#########################################################################
### Part introduction
if page == pages[0]: 
    st.title('Supply Chain- Customer Satisfaction')
    st.caption("Amazon US Customer Reviews -- Video Games ")
    st.caption("Author: Ipsita Ranjan, Philipp Reber, Sebastian Willutzky, Ling Zhu")
    st.text("\n") 

    st.subheader("Introduction")

    st.write("The purpose of this study is to investigate how companies can conduct sentiment analysis based on Amazon.com reviews to gain more insights into customer experience. The dataset used in this study is the Amazon US Customer Reviews Dataset. Here is the [link](https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset) towards the original dataset"
    )
    st.write("The reviews are split into two parts, there is a star-rating system and customer review text. The star rating system is informative; however, it provides only a quantitative analysis on whether a product is popular or not. The review text provides qualitative information on why the product was popular or not."
    )
    st.write("Once a sentiment analysis is completed and a model is trained, companies that sell products on Amazon or sell similar products on other platforms can gain more understanding on top-rated products, what customers value, maintain positive engagement and improve neutral/negative experiences. This can encourage innovative product development and enhanced customer service."
    )
    st.write("The objective of this study is to build a model that can predict the star rating of reviews with a high accuracy and thus determine if the review is associated with a positive or negative sentiment. As mentioned above, the study will involve sentiment analysis and the process relies on machine learning (ML) algorithms and natural language processing (NLP)."
    )
    st.image(path_repo + "/STREAMLIT_APP/pic/smilelys.jpg") # TBD: error while opening image. needs to be fixed

#########################################################################
### part Data exploration and cleansing   
if page  == pages[1]: 
    st.subheader("Data exploration and cleansing")

    st.write("The dataset used is the Amazon US Customer Reviews found on Kaggle.com [link](https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset). The data is freely available and is public. The content of the dataset is as follows: "
    )
    df = pd.read_excel(path_repo + "/STREAMLIT_APP/input/dataexploration_datacolumns.xlsx")
    st.dataframe(df)
    st.write("The collection of reviews were written in the Amazon.com marketplace and associated metadata from 1995 until 2015. Examples of products in the dataset include but are not limited to apparel, books, furniture, musical instruments, toys etc. "
    )
    st.write("For the purpose of this study, the dataset based on reviews for Hardcopy and Digital videogames were used. It is approximately 1,2 GB in size with 15 columns and 1.924.992 entries."
    )
    st.write(":red[placeholder for the head of raw data] ")
    st.write("The explaining variables are review_headline and review_body since it is the text in these columns that the model will analyse and ultimately draw conclusions from. The target variable is star_rating since it provides a clear output on the sentiment (i.e., positive, neutral, or negative along 5 rating classes) of the review."
    )
    # df = pd.read_excel(path + "/data/raw/xxx.csv")

    # df = pd.read_csv(path + "/data/raw/speeddating.csv") # tbd our data
    # df_clean = prepare_data(df)
    # features_list = ["gender","age","age_o","attractive_o","sinsere_o","funny_o","intelligence_o",
    #     "funny_partner","attractive_partner","sincere_partner","intelligence_partner"]
    # X_train,X_test,y_train, y_test = train_test_split(df_clean [
    #     features_list ],df_clean['match'])
    
    # st.dataframe(df_clean.head())
    # st.image("nouvelle_image.jpg",caption="A cool picture about the dataset")
    # st.video("https://www.youtube.com/watch?v=WKNRM2xVRJo")
    # st.write("The columns' names are",df_clean.columns)

    st.latex(r'''
    p = \dfrac{1}{1+e^{-{b_{0} + b_{1} \times x }}}
    ''')

   # fig,ax = plt.subplots()

    sns.set_theme(style="whitegrid")

    # Draw a nested barplot by species and sex
    g = sns.catplot(
        data=df_clean, kind="bar",
        x="gender", y="funny", hue="match",
    palette="dark", alpha=.6, height=6
    )
    g.despine(left=True)
    g.set_axis_labels("Gender", "Funny ")
    plt.xticks(rotation=45)
    g.legend.set_title("Match ? ")

    st.pyplot(g)

    fig,ax = plt.subplots()
    sns.boxplot(x="gender", y="funny_o",
            hue="match", palette=["m", "g"],
            data=df_clean)
    sns.despine(offset=10, trim=True)
    st.pyplot(fig)


    fig3 = plt.figure()
    sns.displot(
    df, x="age", col="race", row="gender",
    binwidth=3, height=3, facet_kws=dict(margin_titles=True),
    )
    st.pyplot(fig3)
    number_start = st.number_input('Insert a number for first feature ',0,len(df_clean.columns)-1,1)
    number_end= st.slider('Number of features?',min_value= int(number_start), max_value=int(len(df_clean.columns)) ,step= 1)
    st.write('The current number is ', number_start)
    st.write('The current number is ', number_end)
    if number_start < number_end:
        st.pyplot(draw_correlation_with_target(df_clean,nb_features=[number_start,number_end]))

#########################################################################
### part Visualization    
if page == pages[2]:
    st.subheader("Data Visualization")
    model_name =st.selectbox("Choose a ML Model to train",options=["KNN","Logistic Regression","Random Forest"])
    st.write(f"The performance of the ML model is {get_score(model_name,X_train,X_test,y_train,y_test)}")

    features_range =[list(df_clean[f].unique()) for f in features_list]
    options ={}
    for i in range(len(features_list))  : 
        options[features_list[i]]= st.selectbox(
        f'What is your {features_list[i]} ?',
      features_range[i]
        )

    st.write('You selected:', options)

    inputs = pd.DataFrame(options,index=[1])
    b =st.button("Click to see your predictions !")
    if b :
        pred =get_predictions(model_name,inputs,X_train,y_train)
        if pred :
         
            st.success('Congratulations !! You have suceedeed!', icon="✅")
           # st.write(f'The prediction of the ML model is ',pred)
        else :
            st.error('tough Luck !! You are out, Next !', icon="🚨")
            #st.write(f'The prediction of the ML model is ',pred)
           # st.write( ' "Cogito ergo sum" Descartes')

#########################################################################
### part Modelling    
if page == pages[3]:
    st.subheader("Data Modelling and Interpretation")


#########################################################################
### part interactive part    
import spacy
from joblib import dump, load
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
nlp = spacy.load("en_core_web_lg")



def lemmatize_and_pos_tag(review):
    doc = nlp(review)
    lst = []
    for tok in doc:
        if (tok.pos != 97 ) and (tok.pos_ != "SPACE") and (tok.is_alpha or tok.pos_== "PART") and (tok.ent_type_ == ''):
           lst.append(tok.lemma_ + "_" + tok.pos_)
    return " ".join(lst)


if page == pages[4]:
    st.subheader("Interactive part - Get the sentiment")
    st.write("Loading model from "+ path_repo)    
    try:
        pipeline=load(path_repo + '\logreg_model.joblib') 
        st.write("Model loaded")
        st.write("==============================================================")
    except:
        st.write("No model found!")
        st.write("==============================================================")
        
    #df = pd.read_pickle(r"..\data\data_en3.pickle")
    
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
#########################################################################
### part conclusion 
if page == pages[5]:
    st.subheader("Conclusion")