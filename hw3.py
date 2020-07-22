#%%
import numpy as np
import pandas as pd
import nltk
import sklearn 
import string
import re # helps you filter urls
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
#%%
#Whether to test your Q9 for not? Depends on correctness of all modules
def test_pipeline():
    return True # Make this true when all tests pass

# Convert part of speech tag from nltk.pos_tag to word net compatible format
# Simple mapping based on first letter of return tag to make grading consistent
# Everything else will be considered noun 'n'
posMapping = {
# "First_Letter by nltk.pos_tag":"POS_for_lemmatizer"
    "N":'n',
    "V":'v',
    "J":'a',
    "R":'r'
}

#%%
def process(text, lemmatizer=nltk.stem.wordnet.WordNetLemmatizer()):
    """ Normalizes case and handles punctuation
    Inputs:
        text: str: raw text
        lemmatizer: an instance of a class implementing the lemmatize() method
                    (the default argument is of type nltk.stem.wordnet.WordNetLemmatizer)
    Outputs:
        list(str): tokenized text
    """
    print(text)
    mapping = {
    "N" : "n",
    "V" : "v",
    "J" : "a",
    "R" : "r"
    }
    # all punctuation
    punc = string.punctuation
    # no punctuation string
    no_punc = ""
    # remove URL
    #text = re.sub("[\w]*://[\S]*", "" ,text)
    text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', "" ,text)
    # remove all punctuation
    text = text.replace("'s","")
    for i in text:
        if i not in punc:
            no_punc += i.lower()
        elif i == "-":
            no_punc += " "
        elif i =="'":
            no_punc += ""
        else:
            no_punc += " "
    token = nltk.word_tokenize(no_punc)
    token = nltk.pos_tag(token)
    word = []
    for x in token:
        word.append(lemmatizer.lemmatize(x[0],pos=mapping.get(x[1][0],"n")))
    return word
    print(word)
#%%
def process_all(df, lemmatizer=nltk.stem.wordnet.WordNetLemmatizer()):
    """ process all text in the dataframe using process function.
    Inputs
        df: pd.DataFrame: dataframe containing a column 'text' loaded from the CSV file
        lemmatizer: an instance of a class implementing the lemmatize() method
                    (the default argument is of type nltk.stem.wordnet.WordNetLemmatizer)
    Outputs
        pd.DataFrame: dataframe in which the values of text column have been changed from str to list(str),
                        the output from process_text() function. Other columns are unaffected.
    """
    for i in df.index:
        df['text'][i] = process(df['text'][i])
    return df
    
#%%
def create_features(processed_tweets, stop_words):
    """ creates the feature matrix using the processed tweet text
    Inputs:
        tweets: pd.DataFrame: tweets read from train/test csv file, containing the column 'text'
        stop_words: list(str): stop_words by nltk stopwords
    Outputs:
        sklearn.feature_extraction.text.TfidfVectorizer: the TfidfVectorizer object used
            we need this to tranform test tweets in the same way as train tweets
        scipy.sparse.csr.csr_matrix: sparse bag-of-words TF-IDF feature matrix
    """
    #[YOUR CODE HERE]
    tweets = []
    for x in processed_tweets.index:
#         for word in processed_tweets['text'][x]:
#             if word in stop_words:
#                 processed_tweets['text'][x].remove(word)
        tweets.append(' '.join(processed_tweets['text'][x]))
    #print(tweets)
    vec = TfidfVectorizer(min_df=2,stop_words=stop_words)
    mat = vec.fit_transform(tweets)
    return vec, mat

#%%
def create_labels(processed_tweets):
    """ creates the class labels from screen_name
    Inputs:
        tweets: pd.DataFrame: tweets read from train file, containing the column 'screen_name'
    Outputs:
        numpy.ndarray(int): dense binary numpy array of class labels
    """
    #[YOUR CODE HERE]
    zero = ['realDonaldTrump','mike_pence','GOP']
    num = []
    for i in processed_tweets.index:
        if processed_tweets['screen_name'][i] in zero:
            num.append(0)
        else:
            num.append(1)
    return np.array(num)
#%%
class MajorityLabelClassifier():
  """
  A classifier that predicts the mode of training labels
  """  
  def __init__(self):
    """
    Initialize
    """
    self.mode = -1
    #[YOUR CODE HERE]
  
  def fit(self, X, y):
    """
    Implement fit by taking training data X and their labels y and finding the mode of y
    """
    one = 0
    two = 0
    val1 = y[0]
    val2 = None 
    for x in y:
        if x != val1:
            val2 = x
            break
    for x in y:
        if x == val1:
            one +=1
        else:
            two +=1
    if one > two:
        self.mode = val1
    else:
        self.mode = val2
#sdsd
  def predict(self, X):
    """
    Implement to give the mode of training labels as a prediction for each data instance in X
    return labels
    """
    result = []
    for val in X:
        result.append(self.mode)
    return result
    #[YOUR CODE HERE]

#%%
def learn_classifier(X_train, y_train, kernel):
    """ learns a classifier from the input features and labels using the kernel function supplied
    Inputs:
        X_train: scipy.sparse.csr.csr_matrix: sparse matrix of features, output of create_features()
        y_train: numpy.ndarray(int): dense binary vector of class labels, output of create_labels()
        kernel: str: kernel function to be used with classifier. [linear|poly|rbf|sigmoid]
    Outputs:
        sklearn.svm.classes.SVC: classifier learnt from data
    """  
    #[YOUR CODE HERE]
    classifier = SVC(kernel=kernel)
    classifier.fit(X_train,y_train)
    return classifier

#%%
def evaluate_classifier(classifier, X_validation, y_validation):
    """ evaluates a classifier based on a supplied validation data
    Inputs:
        classifier: sklearn.svm.classes.SVC: classifer to evaluate
        X_validation: scipy.sparse.csr.csr_matrix: sparse matrix of features
        y_validation: numpy.ndarray(int): dense binary vector of class labels
    Outputs:
        double: accuracy of classifier on the validation data
    """
    #[YOUR CODE HERE]
    #count = 0
    #pred = []
    #for x in range(y_validation.size):
    #    pred.append(classifier.predict(X_validation[x]))
    #    if  pred[x] == y_validation[x]:
    #        count +=1
    #return count/y_validation.size
    #return accuracy_score(pred,y_validation)
    return accuracy_score(classifier.predict(X_validation),y_validation)

#%%
def classify_tweets(tfidf, classifier, unlabeled_tweets):
    """ predicts class labels for raw tweet text
    Inputs:
        tfidf: sklearn.feature_extraction.text.TfidfVectorizer: the TfidfVectorizer object used on training data
        classifier: sklearn.svm.classes.SVC: classifier learnt
        unlabeled_tweets: pd.DataFrame: tweets read from tweets_test.csv
    Outputs:
        numpy.ndarray(int): dense binary vector of class labels for unlabeled tweets
    """
    modify = process_all(unlabeled_tweets)
    list_of_tweets = []
    for x in modify.index:
        list_of_tweets.append(' '.join(modify['text'][x]))
    td_transform = tfidf.transform(list_of_tweets)
#     result = []
#     for x in unlabeled_tweets:
#         result.append(classifier.predict(x))
    return classifier.predict(td_transform)
    #[YOUR CODE HERE]