import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd 
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import kmeans_plusplus 

st.sidebar.title("Différents Onglets")
pages =["Presentation","Visualisation","Modelisation"]
page = st.sidebar.radio("Choisir une page",options = pages)
df = pd.read_csv("speeddating.csv")


if page == pages[0]: 
    st.title('An introduction on streamlit')
    st.header("By Datascientest")
    st.image("dataset-cover.jpg",caption="A cool picture about the dataset")
    st.video("https://www.youtube.com/watch?v=WKNRM2xVRJo")
    st.write("The columns' names are")
    st.markdown("Here is the [link](https://www.kaggle.com/datasets/annavictoria/speed-dating-experiment) towards the original dataset")
    

if page  == pages[1]: 
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
    

    
if page == pages[2]:
    model_name =st.selectbox("Choose a ML Model to train",options=["KNN","Logistic Regression","Random Forest"])
    st.write(f"The performance of the ML model is")

    