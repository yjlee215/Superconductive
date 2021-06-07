import numpy as np
import pandas as pd

# visualization
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
from sklearn import neighbors
from itertools import combinations

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB # Gaussian Naive Bayes
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

import statsmodels.formula.api as smf
import statsmodels.formula.api as sm

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler

# import Beautiful Soup, re,
import bs4 as bs
import re
import hashlib

# download NLTK classifiers - these are cached locally on your machine
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

# import ml classifiers
from nltk.tokenize import sent_tokenize # tokenizes sentences
from nltk.stem import PorterStemmer     # parsing/stemmer
from nltk.tag import pos_tag            # parts-of-speech tagging
from nltk.corpus import wordnet         # sentiment scores
from nltk.stem import WordNetLemmatizer # stem and context
from nltk.corpus import stopwords       # stopwords
from nltk.util import ngrams            # ngram iterator

# import word2vec
from gensim.test.utils import datapath
from gensim import utils
from gensim.models import Word2Vec
from gensim.test.utils import common_texts

# import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize, FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# import datetime
import datetime as dt
from datetime import *
import time

import streamlit as st

########################################################################
st.title("1. Upload CSV File")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(df)

    st.write("""

    ### Check the missing values

    """)
    # Automatically visualize the missing Values on dataset
    fig_missing = plt.figure()
    sns.heatmap(df.isnull(), yticklabels = False, cbar = False)
    st.pyplot(fig_missing)

    # Print the missing proportion for each column
    df_ = df.isnull()
    st.write("""#### The missing Proportion for each column""")

    for col_name in df_.columns:
        st.write(col_name + ": %f " % df_[col_name].mean())

    # + automatically drop the column if its missing proportion is higher than 50%
    # + automatically replce the missing values with median

    Train = df.copy(deep = True)
    for col_name in df_.columns:
        if df_[col_name].mean() >= 0.5:  # Drop the column if its missing proportion > 50%
            Train = Train.drop(columns = [col_name])
            st.write(col_name + " is dropped because its missing proportion is > 50%", "\n")

        elif df_[col_name].mean() > 0:   # Handle missing values
            Train.fillna(Train.median(), inplace = True)

        # Re-check the missing Values on dataset - should be no missing proportion
        #st.write("""#### Check the missing values after pre-processing""")
        #fig_missing_2 = plt.figure()
        #sns.heatmap(Train.isnull(), yticklabels = False, cbar = False)
        #st.pyplot(fig_missing_2)


##################################################################
st.write("""
# 2. Data Pre-processing


### Which data pre-processing method do you want to apply?

""")
# Convert Data Type
def astype_inplace(df, dct):
    df[list(dct.keys())] = df.astype(dct)[list(dct.keys())]
    return df
convert_type = st.checkbox("Convert Data Type")
if convert_type:
    col_names = st.text_input("Enter the column name and the targeted data type for convert data type (e.g. Fare int)")
    col_names_ = col_names.split()
    if col_names:
        Train = astype_inplace (Train, {col_names_[0]: col_names_[1]})
        st.dataframe(Train.head())

# Extract numbers from a string(letter + numbers)
def extract_numbers_from_string(df, col_name):
    new_list = []
    list = df[col_name]
    for string in list:
        new_data = ''.join(filter(str.isdigit, string))
        new_list.append(new_data)
        pd.Series(new_list)
    df[col_name + "_w_number"] = new_list
    df[col_name + "_w_number"] = pd.to_numeric(df[col_name + "_w_number"])
    return df
extract_numbers = st.checkbox("Extract Numbers From A String")
if extract_numbers:
    col_names = st.text_input("Enter the column name for extracting numbers from string, e.g. Ticket")
    if col_names:
        Train = extract_numbers_from_string(Train, col_names)
        st.dataframe(Train.head())

# Extract letters from a string(letter + numbers)
def extract_letters_from_string(df, col_name):
    new_list = []
    list = df[col_name]
    for string in list:
        new_data = " ".join(re.split("[^a-zA-Z]*", string))
        new_list.append(new_data)
        pd.Series(new_list)
    df[col_name + "_w_letter"] = new_list
    return df
extract_letters = st.checkbox("Extract Letters From A String")
if extract_letters:
    col_names = st.text_input("Enter the column name for extracting letters from a string, e.g. Ticket")
    if col_names:
        Train = extract_letters_from_string(Train, col_names)
        st.dataframe(Train.head())

#clean up string columns by removing html tags, emoticons, and all special characters
def string_cleaner(data, col_name):
    #1. Remove HTML tags
    data[col_name] = data[col_name].apply(lambda x: bs.BeautifulSoup(x).text)
    #2. Use regex to find emoticons
    emoticons = data[col_name].apply(lambda x: re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', x))
    #3. Remove punctuation (keep only letters and digits)
    data[col_name] = data[col_name].apply(lambda x: re.sub("[^a-zA-Z0-9]", "", x))
    return data
string_clean = st.checkbox("String Clean-up")
if string_clean:
    col_names = st.text_input("Enter the column name for string cleaner, e.g. Ticket")
    if col_names:
        Train = string_cleaner(Train, col_names)
        st.dataframe(Train.head())

#allows users to drop multiple columns.
def drop_columns(df, col):
    return df.drop(col, axis = 1)
drop = st.checkbox("Drop Columns")
if drop:
    col_names = st.text_input("Enter the column name for drop column, e.g. Ticket PassengerId")
    col_names_ = col_names.split()
    if col_names:
        Train = drop_columns(Train, col_names_)
        st.dataframe(Train.head())

##################################################################

st.write("""
# 3. Feature transformation


### Which feature transformation method do you want to apply?

""")


#Fourier Transform
from scipy.fftpack import fft
def fourier_transformation(data):
    try:
        ft = fft(data.values)
        ft_real = ft.real
        ft_imag = ft.imag
        ft_abs = abs(ft)
        ft_real_norm = (ft_real - np.min(ft_real)) / (np.max(ft_real) - np.min(ft_real))
        ft_imag_norm = (ft_imag - np.min(ft_imag)) / (np.max(ft_imag) - np.min(ft_imag))
        return ft_real_norm, ft_imag_norm
    except:
        print("FFT ERROR!")

fourier = st.checkbox("fourier transformation")
if fourier:
    col_name = st.text_input("Enter the column name for fourier tranformation")
    if col_name:
        Train[col_name + "_FFT_real"], Train[col_name + "_FFT_imag"] = fourier_transformation(Train[col_name])
        st.dataframe(Train.head())




#get dummies
#Use for nominal data
def get_dummy(data, column):
    new_data = pd.get_dummies(data, columns = column)
    return new_data

dummies = st.checkbox("get dummies")
if dummies:
    col_names = st.text_input("Enter the column name for get dummy")
    col_names = col_names.split()
    Train = get_dummy(Train, column = col_names)
    st.dataframe(Train.head())




# LabelBinarizer
# Binary nominal data
# convert categorical features with binary values to integer values of 0 or 1
from sklearn.preprocessing import LabelBinarizer
def BinaryColumnConvertToInt(data, column):
    if data[column].nunique() != 2: #Another approach: use assert
        print("The column is not binary")
        return
    # binarizer object for each binary categorical variable
    binarizer = LabelBinarizer()
    # Fit and transform each respective binary variable to their respective binarizer objects
    data[column] = binarizer.fit_transform(data[column])
    return data

labelbinary = st.checkbox("label binarizer")
if labelbinary:
    col_names = st.text_input("Enter the column name for label binarizer")
    if col_names:
        Train = BinaryColumnConvertToInt(Train, col_names)
        st.dataframe(Train.head())



#Ordinal Encoding for ordinal data
#Encoder ordinal value in ascending order eg. Highschool, college, masters
#Result: Highschool->0, college->1, masters->2
import category_encoders as ce
def ordinal_encode(df, col, name):
    # create object of Ordinalencoding
    col_dict = {}
    for i in np.arange(0, len(name)):
        col_dict[name[i]] = i + 1

    #print(col_dict)
    encoder = ce.OrdinalEncoder(cols = [col], return_df = True,
                               mapping = [{'col' : col, 'mapping' : col_dict}])
    df_train_transformed = encoder.fit_transform(df)
    return df_train_transformed

ordinal_encoder = st.checkbox("ordinal encoder")
if ordinal_encoder:
    col_name = st.text_input("Enter the column name for ordinal encoder")
    ordinal_val = st.text_input("input ordinal value in ascending order")
    lst = []
    ordinal_val = ordinal_val.split()
    for word in ordinal_val:
        lst.append(word)
    if col_name and ordinal_val:
        Train = ordinal_encode(Train, col_name, lst)
        st.dataframe(Train.head())





#Log transformation; data and column will be passed into function
def log_transformation(data, col):
    data["log_" + col] = np.log(data[col])
    data.drop([col], axis = 1, inplace = True)
    return data
log_transform = st.checkbox("log transformation")
if log_transform:
    col_names = st.text_input("Enter the column name for log transform")
    if col_names:
        Train = log_transformation(Train, col_names)
        log_transformed = True #To generate evaluation method later
        st.dataframe(Train.head())




#A bag of word
# Clean, tokenize and lemmatize a text.
ps = PorterStemmer()
wnl = WordNetLemmatizer()
eng_stopwords = set(stopwords.words("english"))

def text_cleaner(text, lemmatize = True, stem = False):
    if lemmatize == True and stem == True:
        raise RuntimeError("May not pass both lemmatize and stem flags")

    #1. Remove HTML tags
    text = bs.BeautifulSoup(text).text

    #2. Use regex to find emoticons
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)

    #3. Remove punctuation
    text = re.sub("[^a-zA-Z]", " ", text)

    #4. Tokenize into words (all lower case)
    text = text.lower().split()

    #5. Remove stopwords, Lemmatize, Stem
    eng_stopwords = set(stopwords.words("english"))
    text = [w for w in text if not w in eng_stopwords]
    text_wnl = []
    wnl = WordNetLemmatizer()
    for w in text:
        text_wnl.append(wnl.lemmatize(w))
    #6. Join the review to one sentence
    text_processed = ' '.join(text_wnl + emoticons)

    return text_processed

# Use text_cleaner function to clean, tokenize and lemmatize texts in the table .
def table_text_clean(df, col_name):
    for row_name in df.index:
        text_clean = text_cleaner(text = df.at[row_name, col_name], lemmatize = True, stem = False)
        df.at[row_name, col_name] = text_clean
    return df

# Vectorize the text using a bag of words model
def get_vectorizer(ngram, max_features):
    return CountVectorizer(ngram_range = (1, ngram),
                             analyzer = "word",
                             tokenizer = None,
                             preprocessor = text_cleaner,
                             stop_words = None,
                             max_features = max_features)

def bag_of_word(df, col_name):
    vectorizer = get_vectorizer(ngram = 2, max_features = 100)  #set 100 as maximum features (identify maximum 100 different word, and count frequency)
    # Get feature names(each different word),and turn features into an array of frequency
    df_cleaned = table_text_clean(df, col_name)
    v = vectorizer.fit_transform(df_cleaned[col_name])
    bag_of_word_features = pd.DataFrame(data = v.toarray(), columns = vectorizer.get_feature_names())
    # Combine the created features with Train dataset
    df_w_features = pd.concat([df_cleaned, bag_of_word_features], axis = 1, join = 'inner')
    return df_w_features

bag_word = st.checkbox("A bag of word")
if bag_word:
    col_name = st.text_input("Enter the column name for a bag of word")
    if col_name:
        Train = bag_of_word(Train, col_name)
        st.dataframe(Train.head())



#Word2Vec
# Given a set of texts (each one a list of words), calculate the average feature vector for each one
def get_avg_feature_vecs(df, col_name):
    Table_cleaned = table_text_clean(df, col_name) # Use table_text_cleaner function to clean, tokenize and lemmatize texts in table
    model = Word2Vec(sentences = Table_cleaned[col_name], size = 100, window = 5, min_count = 1, workers = 4) #Initialize the model

    # Index2word is a list that contains the names of the words in the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.wv.index2word)

    textFeatureVecs = []
    # Loop through the texts
    for counter, text in enumerate(Table_cleaned[col_name]):

        # Function to average all of the word vectors in a given paragraph
        featureVec = []

        # Loop over each word in the text and, if it is in the model's vocaublary, add its feature vector to the total
        for n,word in enumerate(text):
            if word in index2word_set:
                featureVec.append(model.wv[word])

        # Average the word vectors
        featureVec = np.mean(featureVec, axis = 0).reshape(1, -1)

        textFeatureVecs.append(featureVec)
        word2vec_features = pd.DataFrame(data = np.concatenate(textFeatureVecs, axis = 0))
        df_w_features = pd.concat([Table_cleaned, word2vec_features], axis = 1, join = 'inner')
    return df_w_features
word_vec =  st.checkbox("Word2Vec")
if word_vec:
    col_name = st.text_input("Enter the column name for word 2 vec")
    if col_name:
        Train = get_avg_feature_vecs(Train, col_name)
        st.dataframe(Train.head())


##################################################################
st.write("""
# 4. Feature Engineering and selection

### Which feature engineering and selection do you want to apply?

""")


#####################################
# Select top 20 features and visualize

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace = True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


def best_features(df):
    list_f = list(df.columns)
    #Independent Column
    x = df.iloc[:, df.columns != list_f[1]]

    #Target Column
    y = df.iloc[:, df.columns.get_loc(list_f[1])]

    #Apply SelectKBest class to extract top 5 best features
    bestfeatures = SelectKBest(score_func = chi2, k = 3)
    fit = bestfeatures.fit(x,y)
    df_scores = pd.DataFrame(fit.scores_)
    df_columns = pd.DataFrame(x.columns)


    #Concat two dataframes for better visualization
    featureScores = pd.concat([df_columns,df_scores],axis=1)

    #Naming the dataframe columns
    featureScores.columns = ['Specs','Score']
    data_twenty = featureScores.nlargest(20,'Score')
    print(data_twenty)
    return (data_twenty['Specs'])

top_feature = st.checkbox("Choosing top features")
if top_feature:
    st.write("""
    ### Here is your top 20 feature scores
    """)
    Train_ = Train.copy()
    a = Train_.select_dtypes(exclude = 'object')
    b = clean_dataset(a)
    best_col = best_features(b)
    st.dataframe(best_col)

#View the correlation between features in our processed dataset
colormap = plt.cm.viridis
fig_corr = plt.figure(figsize=(12,12))
plt.title('Pearson Correlation of top 20 Features', y=1.05, size=15)
sns.heatmap(Train[best_col].corr().round(2)\
            ,linewidths=0.1,vmax=1.0, square=True, cmap=colormap, \
            linecolor='white', annot=True);
st.pyplot(fig_corr)

#Covariance Heatmap
fig_covar = plt.figure(figsize = (14,10))
plt.title("Covariance of top 20 features")
sns.set(font_scale = 2)
sns.heatmap(Train[best_col].cov())
st.pyplot(fig_covar)

#########################################################################


#plot distribution function
def plot_distribution(data, column):
    fig, axs = plt.subplots(nrows = 2)

    sns.distplot(
        data[column],
        ax = axs[0]
    )
    sns.boxplot(
        data[column],
        width = 0.3,
        ax = axs[1],
        showfliers = False,
    )

    # Align axes
    add_space = np.max(data[column]) * 0.05
    min_val = np.min(data[column]) - add_space
    max_val = np.max(data[column]) + add_space
    axs[0].set_xlim((min_val, max_val))
    axs[1].set_xlim((min_val, max_val))

    # Put the two plots together
    plt.subplots_adjust(hspace = 0)

    # Adjust boxplot fill to be white
    axs[1].artists[0].set_facecolor('white')
plot_dist = st.checkbox("Plot Distribution")
if plot_dist:
    st.write("""
    ### Here is your outlier plot distribution
    """)
    col_name = st.text_input("Please enter the column name that you want to plot distribution")
    if col_name:
        outlier_graph = plot_distribution(Train, col_name)
        st.pyplot(outlier_graph)


#remove outliers function
def remove_outliers(data, column, lower=-np.inf, upper=np.inf):
    lower = int(lower)
    upper = int(upper)
    print("Initial numeric data size is : ", data.shape)
    num = len(data)
    data = data[(data[column] >= lower) & (data[column] <= upper) ]
    print(column, "removed", num - len(data))
    print("The data size after removing outliers : ", data.shape)
    return data
outlier = st.checkbox("Remove Outliers")
if outlier:
    st.write("""
    ### Here is your outlier plot distribution
    """)
    col_name = st.text_input("Please enter the column name that you want to remove outlier")
    lower = st.text_input("Input lower boundary")
    upper = st.text_input("Input upper boundary")
    if col_name and lower and upper:
        Train = remove_outliers(Train, col_name, lower, upper)
        st.dataframe(Train)



#top_interaction function
def top_interaction(data, numeric_col, number, y_train):
    data_numeric = data[numeric_col]
    interactions = list(combinations(data_numeric.columns, 2))
    print(interactions)

    interaction_dict = {}
    for interaction in interactions:
        data_int = data_numeric.copy()
        data_int['interaction'] = data_int[interaction[0]] * data_int[interaction[1]]

        lr_int = LinearRegression()
        lr_int.fit(data_int, y_train)
        interaction_dict[lr_int.score(data_int, y_train)] = interaction
    print(interaction_dict)

    top = sorted(interaction_dict.keys(), reverse = True)
    interaction_top_dict = []
    for interaction in top:
        interaction_top_dict.append(interaction_dict.get(interaction))
        if str(len(interaction_top_dict)) == number:
            break

    # Add the top interaction features (which exclude a categorical variable)
    data_final = data.copy()
    for interaction in interaction_top_dict:
        col_name = interaction[0] + "*" + interaction[1]
        data_final[col_name] = data_final[interaction[0]] * data_final[interaction[1]]

    return data_final

top_interact = st.checkbox("Top Interaction")
if top_interact:
    st.write("""
    ### Here is your Top Interaction
    """)
    col_names = st.text_input("Please input numerical column names for top interaction")
    num = st.text_input("Input number of features to be added")
    y_train = st.text_input("Input response(predict) variable")
    col_names = list(col_names.split())
    if col_names and num and y_train:
        Train = top_interaction(Train, col_names, num, Train[y_train])
        st.dataframe(Train)

##################################################################
st.write("""
# 5. Train Test Split



""")

#Train Test Split function
def train_test(col_name):
    X = Train.drop([col_name] , axis = 1)
    Y = Train[col_name]

    np.random.seed(1337)
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.2, random_state = 0)
    return X_train, X_val, Y_train, Y_val

splitted = st.checkbox("train-test split")
if splitted:
    col_name = st.text_input("Please enter the column name that you want as the output(label) eg. Survived") # Should be "Survived" in this dataset
    if col_name:
        X_train, X_val, Y_train, Y_val = train_test(col_name)
        st.write(X_train.shape, Y_train.shape)
        st.write(X_val.shape, Y_val.shape)
        st.write(X_train.head(), Y_train.head())
        st.write(X_val.head(), Y_val.head())


        #This feature scaling section will only run after train test split is done
        st.write("""
        # 6. Feature Scaling



        """)

        ################################
        #Feature Scaling - normalization
        from sklearn.preprocessing import MinMaxScaler
        # fit scaler on training data
        def normalization(X_train, X_val):
            norm = MinMaxScaler().fit(X_train)

            # transform training data
            X_train_norm = norm.transform(X_train)

            # transform testing dataabs
            X_val_norm = norm.transform(X_val)

            return X_train_norm, X_val_norm

        normalizate = st.checkbox("Feature scaling - Normalization")
        if normalizate:
            X_train, X_val = normalization(X_train, X_val)
            st.write(X_train.head(), Y_train.head())
            st.write(X_val.head(), Y_val.head())

        ###################################
        #Feature Scaling for numeric columns in x train and x test
        from sklearn.preprocessing import StandardScaler
        def standard_scaler(X_train, X_val):
            X_train = X_train.copy()
            X_val = X_val.copy()

            sc = StandardScaler()

            X_train = pd.DataFrame(sc.fit_transform(X_train))
            X_val = pd.DataFrame(sc.transform(X_val))

            return X_train, X_val

        standard_scaling = st.checkbox("Feature scaling - Standard Scaler")
        if standard_scaling:
            X_train, X_val = standard_scaler(X_train, X_val)
            st.write(X_train.head(), Y_train.head())
            st.write(X_val.head(), Y_val.head())



#############################################################################
###############################################################

st.write("""
# 7. Models & Evaluations


### Which models do you want to apply? Select Classification or Regression

""")

data_type = st.selectbox("Select Regression or Classification", ["Select", "Regression", "Classification"])
st.write(f"Selected Option: {data_type!r}")

# the class for drawing comparison metrics
class Storage:
    def __init__(self):
        self.store = pd.DataFrame()
        self.num_of_algorithms = 0
        self.store_regression = pd.DataFrame()
        self.num_of_algorithms_regression = 0
    def add_one_algorithm(self, acc, prec, recall, fl, algorithm = "logistic"):
        self.store[algorithm] = [acc, prec, recall, fl]
        self.num_of_algorithms += 1
        self.store.rename(index = {0 : 'acc', 1 : 'prec', 2 : 'recall', 3 : 'f1'}, inplace = True)
    def set_num(self, num):
        self.num_of_algorithms = num
    def draw_statistic(self):
        name_list = list(self.store.columns)
        x = list(range(len(name_list)))
        total_width, n = 0.8, 4
        width = total_width / n

        fig_acc = plt.bar(x,np.array(self.store.loc['acc']), width = width, label = 'acc', fc = 'y')
        for i in range(len(x)):
            x[i] = x[i] + width
        fig_prec =plt.bar(x, np.array(self.store.loc['prec']), width = width, tick_label = name_list, label = 'prec',fc = 'r')
        for i in range(len(x)):
            x[i] = x[i] + width
        plt.bar(x, np.array(self.store.loc['recall']), width = width, label = 'recall',fc = 'b')
        for i in range(len(x)):
            x[i] = x[i] + width
        plt.bar(x, np.array(self.store.loc['f1']), width = width, label = 'f1',fc = 'g')
        plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0)
        st.pyplot()
    def add_one_algorithm_re(self, mae, mse, rmse, algorithm = "logistic"):
        self.store_regression[algorithm] = [mae, mse, rmse]
        self.num_of_algorithms_regression += 1
        self.store_regression.rename(index = {0 : 'MAE', 1 : 'MSE', 2 : 'RMSE'}, inplace = True)

    def draw_statistic_re(self):
        name_list = list(self.store_regression.columns)
        x =list(range(len(name_list)))
        total_width, n = 0.8, 3
        width = total_width / n

        plt.bar(x,np.array(self.store_regression.loc['MAE']), width = width, label = 'MAE', fc = 'y')
        for i in range(len(x)):
            x[i] = x[i] + width
        plt.bar(x, np.array(self.store_regression.loc['MSE']), width = width, tick_label = name_list, label = 'MSE',fc = 'r')
        for i in range(len(x)):
            x[i] = x[i] + width
        plt.bar(x, np.array(self.store_regression.loc['RMSE']), width = width, label = 'RMSE', fc = 'b')
        plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0)
        plt.show()


if (data_type == "Classification"):
    #####################################
    #Logistic Regression model
    logreg = LogisticRegression(max_iter = 10000)                  # instantiate

    # tuning params
    kfold = 5
    logreg_param_grid = {'C':[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
                         'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}

    st.write("Logistic Regression Running...")
    logreg_grid_search = GridSearchCV(logreg, logreg_param_grid, n_jobs = 1, verbose = 10, cv = kfold)
    logreg_grid_search.fit(X_train, Y_train)

    # best params
    logreg_best_parameters = logreg_grid_search.best_estimator_.get_params()
    st.write(logreg_best_parameters)
    logreg = LogisticRegression(C = logreg_best_parameters['C'], solver = logreg_best_parameters['solver'], max_iter = 10000).fit(X_train, Y_train)
    logreg.fit(X_train, Y_train)                                 # fit
    Y_pred_log = logreg.predict(X_val)                           # predict

    #Logistic Regression Evaluation
    acc_log = logreg.score(X_val, Y_val)                       # evaluate accuracy
    st.write("\n",'Logistic Regression accuracy:', str(round(acc_log * 100, 2)), '%')

    prec_log = precision_score(Y_val, Y_pred_log, zero_division = 1)   # evaluate precision
    st.write('Logistic Regression precision score:',str(round(prec_log * 100, 2)), '%')

    recall_log = recall_score(Y_val, Y_pred_log, zero_division = 1)   # evaluate recall
    st.write('Logistic Regression recall score:', str(round(recall_log * 100, 2)), '%')

    f1_log = f1_score(Y_val, Y_pred_log, zero_division = 1)    # evaluate F1 score
    st.write('Logistic Regression F1 score:', str(round(f1_log * 100,2)), '%', "\n")

    c_log = confusion_matrix(Y_val, Y_pred_log)
    st.write('CONFUSION MATRIX:',"\n")
    st.write('           Predicted')
    st.write('           neg pos')
    st.write('   Actual')
    st.write('     neg  ', c_log[0])
    st.write('     pos  ', c_log[1])

    # Logistic instantiation
    store = Storage()
    store.add_one_algorithm(acc_log, prec_log, recall_log, f1_log, algorithm = "logistic")

    #####################################
    #SVM model
    svc = SVC(max_iter = 10000)                                   # instantiate
    #svc.fit(X_train, Y_train)                                   # fit
    #Y_pred_svc = svc.predict(X_val)                            # predict

    # tuning params
    kfold = 2
    svm_param_grid = {'kernel':['rbf', 'linear', 'poly', 'sigmoid'],
                      'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                      'gamma': [0.001, 0.0001]}

    st.write("SVM Running...")
    svm_grid_search = GridSearchCV(svc, svm_param_grid, n_jobs = 1, verbose = 10, cv = kfold)
    svm_grid_search.fit(X_train, Y_train)

    # best params
    svm_best_parameters = svm_grid_search.best_estimator_.get_params()
    svc = SVC(kernel = svm_best_parameters['kernel'], C = svm_best_parameters['C'], gamma = svm_best_parameters['gamma'], probability = True).fit(X_train, Y_train)

    svc.fit(X_train, Y_train)                                   # fit
    Y_pred_svc = svc.predict(X_val)

    acc_svc = svc.score(X_val, Y_val)                            # evaluate accuracy

    #SVM Evaluation
    st.write('Support Vector Machines labeling accuracy:', str(round(acc_svc * 100, 2)), '%')

    prec_svc = precision_score(Y_val, Y_pred_svc, zero_division = 1)   # evaluate precision
    st.write('Support Vector Machines labeling precision score:',str(round(prec_svc * 100, 2)), '%')

    recall_svc = recall_score(Y_val, Y_pred_svc, zero_division = 1)   # evaluate recall
    st.write('Support Vector Machines labeling recall score:', str(round(recall_svc * 100, 2)), '%')

    f1_svc = f1_score(Y_val, Y_pred_svc, zero_division = 1)    # evaluate F1 score
    st.write('Support Vector Machines labeling F1 score:', str(round(f1_svc * 100, 2)), '%', "\n")

    c_svc = confusion_matrix(Y_val, Y_pred_log)
    st.write('CONFUSION MATRIX:',"\n")
    st.write('           Predicted')
    st.write('           neg pos')
    st.write('   Actual')
    st.write('     neg  ',c_svc[0])
    st.write('     pos  ',c_svc[1])

   # SVC instantiation
    store.add_one_algorithm(acc_svc, prec_svc, recall_svc, f1_svc, algorithm = "SVC")

    #####################################
    # KNN classifier
    knn = KNeighborsClassifier(n_neighbors = 3)                  # instantiate

    # tuning params
    kfold = 5
    knn_param_grid = {'n_neighbors':[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                      'leaf_size': [1, 2, 5, 10, 20, 30, 40, 50],
                      'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

    st.write("KNN classifier Running...")
    knn_grid_search = GridSearchCV(knn, knn_param_grid, n_jobs = 1, verbose = 10, cv = kfold)
    knn_grid_search.fit(X_train, Y_train)

    # best params
    knn_best_parameters = knn_grid_search.best_estimator_.get_params()
    knn = KNeighborsClassifier(n_neighbors = knn_best_parameters['n_neighbors'],
                               leaf_size = knn_best_parameters['leaf_size'],
                               algorithm = knn_best_parameters['algorithm']).fit(X_train, Y_train)

    knn.fit(X_train, Y_train)                                   # fit
    Y_pred_knn = knn.predict(X_val)                            # predict

    #KNN Evaluations
    acc_knn = knn.score(X_val, Y_val)                            # predict + evaluate
    st.write('K-Nearest Neighbors labeling accuracy:', str(round(acc_knn * 100, 2)), '%')

    prec_knn = precision_score(Y_val, Y_pred_knn, zero_division = 1)   # evaluate precision
    st.write('K-Nearest Neighbors labeling precision score:',str(round(prec_knn * 100, 2)), '%')

    recall_knn = recall_score(Y_val, Y_pred_knn, zero_division = 1)   # evaluate recall
    st.write('K-Nearest Neighbors labeling recall score:', str(round(recall_knn * 100, 2)), '%')

    f1_knn = f1_score(Y_val, Y_pred_knn, zero_division = 1)    # evaluate F1 score
    st.write('K-Nearest Neighbors labeling F1 score:', str(round(f1_knn * 100, 2)), '%', "\n")

    c_knn = confusion_matrix(Y_val, Y_pred_knn)
    st.write('CONFUSION MATRIX:',"\n")
    st.write('           Predicted')
    st.write('           neg pos')
    st.write('   Actual')
    st.write('     neg  ',c_knn[0])
    st.write('     pos  ',c_knn[1])

    # KNN instantiation
    store.add_one_algorithm(acc_knn, prec_knn, recall_knn, f1_knn, algorithm = "KNN")

    #####################################
    #Random Forest Classifier
    from sklearn.ensemble import RandomForestClassifier
    rf_cla = RandomForestClassifier(n_estimators = 500, random_state = 0)

    # tuning params
    kfold = 2
    random_forest_param_grid = {'n_estimators':[1, 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                                'max_features': ['auto', 'sqrt', 'log2']}

    st.write("Random Forest Classifier Running...")
    random_forest_grid_search = GridSearchCV(rf_cla, random_forest_param_grid, n_jobs = 1, verbose = 10, cv = kfold)
    random_forest_grid_search.fit(X_train, Y_train)

    # best params
    random_forest_best_parameters = random_forest_grid_search.best_estimator_.get_params()
    rf_cla = RandomForestClassifier(n_estimators = random_forest_best_parameters['n_estimators'], max_features = random_forest_best_parameters['max_features']).fit(X_train, Y_train)
    rf_cla.fit(X_train, Y_train)
    rf_cla_pred = rf_cla.predict(X_val)

    #Random Forest Classifier Evaluation
    acc_rf_cla = rf_cla.score(X_val, Y_val)                            # predict + evaluate
    st.write('Random Forest Classifier labeling accuracy:', str(round(acc_rf_cla * 100, 2)), '%')

    prec_rf_cla = precision_score(Y_val, rf_cla_pred, zero_division = 1)   # evaluate precision
    st.write('Random Forest Classifier labeling precision score:',str(round(prec_rf_cla * 100, 2)), '%')

    recall_rf_cla = recall_score(Y_val, rf_cla_pred, zero_division = 1)   # evaluate recall
    st.write('Random Forest Classifier recall score:', str(round(recall_rf_cla * 100, 2)), '%')

    f1_rf_cla = f1_score(Y_val, rf_cla_pred, zero_division = 1)    # evaluate F1 score
    st.write('Random Forest Classifier labeling F1 score:', str(round(f1_rf_cla * 100, 2)), '%', "\n")

    c_rf_cla = confusion_matrix(Y_val, rf_cla_pred)
    st.write('CONFUSION MATRIX:',"\n")
    st.write('           Predicted')
    st.write('           neg pos')
    st.write('   Actual')
    st.write('     neg  ',c_rf_cla[0])
    st.write('     pos  ',c_rf_cla[1])

    # RF Classifier instantiation
    store.add_one_algorithm(acc_rf_cla, prec_rf_cla, recall_rf_cla, f1_rf_cla, algorithm = "RF")

    #####################################
    #Decision Tree Classifier
    decision_tree_cla = DecisionTreeClassifier(max_depth = 5)
    #decision_tree_cla.fit(X_train, Y_train)
    #dt_pred = decision_tree.predict(X_val)

    # tuning params
    kfold = 10
    decision_tree_param_grid = {'max_depth':[1, 5, 10, 50, 100, 200],
                                'criterion': ['gini', 'entropy']}

    st.write("Decision Tree Classifier Running...")
    decision_tree_grid_search = GridSearchCV(decision_tree_cla, decision_tree_param_grid, n_jobs = 1, verbose = 10, cv = kfold)
    decision_tree_grid_search.fit(X_train, Y_train)

    # best params
    decision_tree_best_parameters = decision_tree_grid_search.best_estimator_.get_params()
    decision_tree_cla = DecisionTreeClassifier(max_depth = decision_tree_best_parameters['max_depth'], criterion = decision_tree_best_parameters['criterion']).fit(X_train, Y_train)
    decision_tree_cla.fit(X_train, Y_train)
    dt_cla_pred = decision_tree_cla.predict(X_val)

    #Decision Tree Evaluation
    acc_dt_cla = decision_tree_cla.score(X_val, Y_val)                            # predict + evaluate
    st.write('Decision Tree Classifier labeling accuracy:', str(round(acc_dt_cla * 100, 2)), '%')

    prec_dt_cla = precision_score(Y_val, dt_cla_pred, zero_division = 1)   # evaluate precision
    st.write('Decision Tree Classifier labeling precision score:',str(round(prec_dt_cla * 100, 2)), '%')
    recall_dt_cla = recall_score(Y_val, dt_cla_pred, zero_division = 1)   # evaluate recall
    st.write('Decision Tree Classifier recall score:', str(round(recall_dt_cla * 100, 2)), '%')
    f1_dt_cla = f1_score(Y_val, dt_cla_pred, zero_division = 1)    # evaluate F1 score
    st.write('Decision Tree Classifier labeling F1 score:', str(round(f1_dt_cla * 100, 2)), '%', "\n")
    c_dt_cla = confusion_matrix(Y_val, dt_cla_pred,)
    st.write('CONFUSION MATRIX:',"\n")
    st.write('           Predicted')
    st.write('           neg pos')
    st.write('   Actual')
    st.write('     neg  ',c_dt_cla[0])
    st.write('     pos  ',c_dt_cla[1])
    # Decision Tree Classifier instantiation
    store.add_one_algorithm(acc_dt_cla, prec_dt_cla, recall_dt_cla, f1_dt_cla, algorithm = "DT")
    store.draw_statistic()



elif (data_type == "Regression"):

    #####################################
    #KNN Regressor model
    rmse_val = []
    smallest_error = np.inf
    for k in range(20):
        k = k+1
        model = neighbors.KNeighborsRegressor(n_neighbors = k)

        model.fit(X_train, Y_train)  #fit the model
        pred_reg = model.predict(X_val) #make prediction on test set
        error = np.sqrt(mean_squared_error(Y_val, pred_reg)) #calculate rmse
        if error < smallest_error:
            smallest_error = error
            best_model = model
            Y_pred_knn_re = pred_reg
        rmse_val.append(error) #store rmse values
        st.write('RMSE value for k= ', k ,'is:', error)

    #KNN Regressor Evaluation
    st.write('KNN regressor Mean Absolute Error:', mean_absolute_error(Y_val, Y_pred_knn_re))
    st.write('KNN regressor Mean Squared Error:', mean_squared_error(Y_val, Y_pred_knn_re ))
    st.write('KNN regressor Root Mean Squared Error:', np.sqrt(mean_squared_error(Y_val, Y_pred_knn_re)))

    #Draw statistic for regression models
    store = Storage()
    #SVC instantiation
    store.add_one_algorithm_re(mean_absolute_error(Y_val, Y_pred_svc), mean_squared_error(Y_val, Y_pred_svc), np.sqrt(mean_squared_error(Y_val, Y_pred_svc)), algorithm = "SVC")

    #####################################
    #Random Forest Regressor
    from sklearn.ensemble import RandomForestRegressor

    rf_reg = RandomForestRegressor(n_estimators = 200, random_state = 0)

    # tuning params
    kfold = 2
    random_forest_param_grid = {'n_estimators':[1, 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                                'max_features': ['auto', 'sqrt', 'log2']}

    st.write("Ramdom_forest running...")
    random_forest_grid_search = GridSearchCV(rf_reg, random_forest_param_grid, n_jobs = 1, verbose = 10, cv = kfold)
    random_forest_grid_search.fit(X_train, Y_train)

    # best params
    random_forest_best_parameters = random_forest_grid_search.best_estimator_.get_params()
    rf_reg = RandomForestRegressor(n_estimators = random_forest_best_parameters['n_estimators'], max_features = random_forest_best_parameters['max_features']).fit(X_train, Y_train)
    rf_reg.fit(X_train, Y_train)
    rf_reg_pred = rf_reg.predict(X_val)

    #Random Forest evaluation
    st.write('Random Forest Mean Absolute Error:', mean_absolute_error(Y_val, rf_reg_pred))
    st.write('Mean Squared Error:', mean_squared_error(Y_val, rf_reg_pred))
    st.write('Root Mean Squared Error:', np.sqrt(mean_squared_error(Y_val, rf_reg_pred)))

    # Random Forest instantiation
    store.add_one_algorithm_re(mean_absolute_error(Y_val, rf_reg_pred), mean_squared_error(Y_val, rf_reg_pred), np.sqrt(mean_squared_error(Y_val, rf_reg_pred)), algorithm = "RF")


    #####################################
    #Decision Tree Regressor
    decision_tree = DecisionTreeRegressor(max_depth = 5)

    # tuning params
    kfold = 10
    decision_tree_param_grid = {'max_depth':[1, 5, 10, 50, 100, 200],
                                'criterion': ['mse', 'friedman_mse', 'mae']}

    st.write("Decision Tree Regressor Running...")
    decision_tree_grid_search = GridSearchCV(decision_tree, decision_tree_param_grid, n_jobs = 1, verbose = 10, cv = kfold)
    decision_tree_grid_search.fit(X_train, Y_train)

    # best params
    decision_tree_best_parameters = decision_tree_grid_search.best_estimator_.get_params()
    decision_tree = DecisionTreeRegressor(max_depth = decision_tree_best_parameters['max_depth'], criterion = decision_tree_best_parameters['criterion']).fit(X_train, Y_train)
    decision_tree.fit(X_train, Y_train)
    dt_pred = decision_tree.predict(X_val)

    #Decision Tree Evaluation
    st.write('Decision Tree Mean Absolute Error:', mean_absolute_error(Y_val, dt_pred))
    st.write('Decision Tree Mean Squared Error:', mean_squared_error(Y_val, dt_pred))
    st.write('Decision Tree Root Mean Squared Error:', np.sqrt(mean_squared_error(Y_val, dt_pred)))

    # Decision Tree instantiation
    store.add_one_algorithm_re(mean_absolute_error(Y_val, dt_pred), mean_squared_error(Y_val, dt_pred), np.sqrt(mean_squared_error(Y_val, dt_pred)), algorithm = "DT")

    #####################################
    # Instantiate a regression object
    lr = LinearRegression()

    # tuning params
    kfold = 10
    lr_param_grid = {'normalize':[True, False]}

    st.write("Linear Regression Running...")
    lr_grid_search = GridSearchCV(lr, lr_param_grid, n_jobs = 1, verbose = 10, cv = kfold)
    lr_grid_search.fit(X_train, Y_train)

    # best params
    lr_best_parameters = lr_grid_search.best_estimator_.get_params()
    lr = LinearRegression(normalize = lr_best_parameters['normalize']).fit(X_train, Y_train)
    lr.fit(X_train, Y_train)
    lr_pred = lr.predict(X_val)

    #Linear Regression Evaluation
    st.write('Linear Regression Mean Absolute Error:', mean_absolute_error(Y_val, lr_pred))
    st.write('Linear Regression Mean Squared Error:', mean_squared_error(Y_val, lr_pred))
    st.write('Linear Regression Root Mean Squared Error:', np.sqrt(mean_squared_error(Y_val, lr_pred)))

    store.add_one_algorithm_re(mean_absolute_error(Y_val, lr_pred), mean_squared_error(Y_val, lr_pred), np.sqrt(mean_squared_error(Y_val, lr_pred)), algorithm = "Linear_Regression")
    store.draw_statistic_re()
