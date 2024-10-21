import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

df = pd.read_csv("streamlit/MovieReview.csv")
print(df.shape)
df = df.drop('sentiment', axis=1)
nltk.download()
stop_words = stopwords.words('english')

# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!]+", " ", w)
    w = re.sub(r'\b\w{0,2}\b', '', w)

    # remove stopword
    mots = word_tokenize(w.strip())
    mots = [mot for mot in mots if mot not in stop_words]
    return ' '.join(mots).strip()

df.review = df.review.apply(lambda x :preprocess_sentence(x))
df.head()
import tensorflow as tf
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000)
tokenizer.fit_on_texts(df.review)


st.title("Projet de classification binaire Titanic")
st.sidebar.title("Sommaire")
pages=["Exploration", "DataVizualization", "Modélisation"]
page=st.sidebar.radio("Aller vers", pages)

if page == pages[0] : 
    st.header("Data set")
    st.write("### Extrait")
    st.dataframe(df.head(10))
    st.write(df.shape)
    st.subheader("Describe")
    st.dataframe(df.describe())
    if st.checkbox("Afficher les NA") :
        st.dataframe(df.isna().sum())

if page == pages[1] :
    st.write("### DataVizualization")
    fig = plt.figure()
    sns.countplot(x = 'Survived', data = df)
    plt.title("Répartition des survivants")
    st.pyplot(fig)
    
    fig = plt.figure()
    sns.countplot(x = 'Sex', data = df)
    plt.title("Répartition du genre des passagers")
    st.pyplot(fig)
    
    fig = plt.figure()
    sns.countplot(x = 'Pclass', data = df)
    plt.title("Répartition des classes des passagers")
    st.pyplot(fig)
    
    fig=sns.displot(x = 'Age', data = df)
    plt.title("Distribution de l'âge des passagers")
    st.pyplot(fig)
    
    fig = plt.figure()
    sns.countplot(x = 'Survived', hue='Sex', data = df)
    st.pyplot(fig)

    fig = sns.catplot(x='Pclass', y='Survived', data=df, kind='point')
    st.pyplot(fig)

    fig = sns.lmplot(x='Age', y='Survived', hue="Pclass", data=df)
    st.pyplot(fig)
    
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), ax=ax)
    st.write(fig)
    
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
y = df['Survived']
X_cat = df[['Pclass', 'Sex',  'Embarked']]
X_num = df[['Age', 'Fare', 'SibSp', 'Parch']]   
for col in X_cat.columns:
  X_cat[col] = X_cat[col].fillna(X_cat[col].mode()[0])
for col in X_num.columns:
  X_num[col] = X_num[col].fillna(X_num[col].median())
X_cat_scaled = pd.get_dummies(X_cat, columns=X_cat.columns)
X = pd.concat([X_cat_scaled, X_num], axis = 1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train[X_num.columns] = scaler.fit_transform(X_train[X_num.columns])
X_test[X_num.columns] = scaler.transform(X_test[X_num.columns])

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
def prediction(classifier):
    if classifier == 'Random Forest':
        clf = RandomForestClassifier()
    elif classifier == 'SVC':
        clf = SVC()
    elif classifier == 'Logistic Regression':
        clf = LogisticRegression()
    clf.fit(X_train, y_train)
    return clf
def scores(clf, choice):
    if choice == 'Accuracy':
        return clf.score(X_test, y_test)
    elif choice == 'Confusion matrix':
        return confusion_matrix(y_test, clf.predict(X_test))
if page == pages[2] : 
    st.write("### Modélisation")
    choix = ['Random Forest', 'SVC', 'Logistic Regression']
    option = st.selectbox('Choix du modèle', choix)
    st.write('Le modèle choisi est :', option)
    
    clf = prediction(option)
    display = st.radio('Que souhaitez-vous montrer ?', ('Accuracy', 'Confusion matrix'))
    if display == 'Accuracy':
      st.write(scores(clf, display))
    elif display == 'Confusion matrix':
      st.dataframe(scores(clf, display))
    
