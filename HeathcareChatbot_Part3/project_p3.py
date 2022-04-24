# CS421: Natural Language Processing
# University of Illinois at Chicago
# Fall 2021
# Project Part 3
#
# Do not rename/delete any functions or global variables provided in this template and write your solution
# in the specified sections. Use the main function to test your code when running it from a terminal.
# Avoid writing that code in the global scope; however, you should write additional functions/classes
# as needed in the global scope. These templates may also contain important information and/or examples
# in comments so please read them carefully. If you want to use external packages (not specified in
# the assignment) then you need prior approval from course staff.
# This part of the assignment will be graded automatically using Gradescope.
# =========================================================================================================


# Add your import statements here:
import operator

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
import re
import pickle as pkl
import string

import csv
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# ***** New in Project Part 3! *****
import nltk

# Before running code that makes use of Word2Vec, you will need to download the provided w2v.pkl file
# which contains the pre-trained word2vec representations from Blackboard
#
# If you store the downloaded .pkl file in the same directory as this Python
# file, leave the global EMBEDDING_FILE variable below as is.  If you store the
# file elsewhere, you will need to update the file path accordingly.
EMBEDDING_FILE = "w2v.pkl"


# Function: load_w2v
# filepath: path of w2v.pkl
# Returns: A dictionary containing words as keys and pre-trained word2vec representations as numpy arrays of shape (300,)
def load_w2v(filepath):
    with open(filepath, 'rb') as fin:
        return pkl.load(fin)


# Function: load_as_list(fname)
# fname: A string indicating a filename
# Returns: Two lists: one a list of strings, and the other a list of integers
#
# This helper function reads in the specified, specially-formatted CSV file
# and returns a list of documents (lexicon) and a list of binary values (label).
def load_as_list(fname):
    df = pd.read_csv(fname)
    lexicon = df['Lexicon'].values.tolist()
    label = df['Label'].values.tolist()
    return lexicon, label


# Function: w2v(word2vec, token)
# word2vec: The pretrained Word2Vec representations as dictionary
# token: A string containing a single token
# Returns: The Word2Vec embedding for that token, as a numpy array of size (300,)
#
# This function provides access to 300-dimensional Word2Vec representations
# pretrained on Google News.  If the specified token does not exist in the
# pretrained model, it should return a zero vector; otherwise, it returns the
# corresponding word vector from the word2vec dictionary.
def w2v(word2vec, token):
    word_vector = np.zeros(300, )

    # [YOUR CODE HERE]
    word2vec = load_w2v(EMBEDDING_FILE)
    if token in word2vec:
        word_vector = word2vec[token]

    return np.array(word_vector)


def checkUserName():
    return r"^[A-Z][A-Za-z\.\&\-\']*$"


def checkDate():
    return r"^(0[1-9]|1[012])[- /.](0[1-9]|[12][0-9]|3[01])[- /.]\d\d$"


# Function: extract_user_info(user_input)
# user_input: A string of arbitrary length
# Returns: Two strings (a name, and a date of birth formatted as MM/DD/YY)
#
# This function extracts a name and date of birth, if available, from an input
# string using regular expressions.  Names are assumed to be UTF-8 strings of
# 2-4 consecutive camel case tokens, and dates of birth are assumed to be
# formatted as MM/DD/YY.  If a name or a date of birth can not be found in the
# string, return an empty string ("") in its place.
def extract_user_info(user_input):
    name = ""
    dob = ""

    # [YOUR CODE HERE]
    inputToken = get_tokens(user_input)
    print(inputToken)
    # first to check the name has:
    # first character of each string is capital
    # Both first and last name is there along with date
    # All string containing name has letters and the accepted punctuation

    regex = checkUserName()
    name_regex = re.compile(regex)
    for i in range(len(inputToken)):
        if i < len(inputToken) and name_regex.match(inputToken[i]):
            # print(inputToken[i])
            if (i + 1) < len(inputToken) and name_regex.match(inputToken[i + 1]):
                # print(inputToken[i + 1])
                name = inputToken[i] + " " + inputToken[i + 1]
                # print(name)
                if (i + 2) < len(inputToken) and name_regex.match(inputToken[i + 2]):
                    # print(inputToken[i + 2])
                    name = name + " " + inputToken[i + 2]
                    # print(name)
                    if (i + 3) < len(inputToken) and name_regex.match(inputToken[i + 3]):
                        # print(inputToken[i + 3])
                        name = name + " " + inputToken[i + 3]
                        # print(name)
                        break
                    else:
                        break
                else:
                    break

    regex = checkDate()
    date_regex = re.compile(regex)
    for j in range(len(inputToken)):
        # print(inputToken[j])
        date_match = date_regex.match(inputToken[j])
        if date_match is None:
            dob = ""
        else:
            dob = inputToken[j]
            break

    # print(name)
    # print(dob)
    return name, dob


# Function to convert a given string into a list of tokens
# Args:
#   inp_str: input string 
# Returns: token list, dtype: list of strings
def get_tokens(inp_str):
    return inp_str.split()


# Function: preprocessing(user_input), see project statement for more details
# user_input: A string of arbitrary length
# Returns: A string of arbitrary length
def preprocessing(user_input):
    # Initialize modified_input to be the same as the original user input
    modified_input = user_input

    # Write your code here:
    tokens = get_tokens(user_input)
    no_punctuation = []
    for t in tokens:
        if t not in string.punctuation:
            no_punctuation.append(t.lower())
    modified_input = ' '.join(no_punctuation)
    return modified_input


# Function: string2vec(word2vec, user_input)
# word2vec: The pretrained Word2Vec model
# user_input: A string of arbitrary length
# Returns: A 300-dimensional averaged Word2Vec embedding for that string
#
# This function preprocesses the input string, tokenizes it using get_tokens, extracts a word embedding for
# each token in the string, and averages across those embeddings to produce a
# single, averaged embedding for the entire input.
def string2vec(word2vec, user_input):
    embedding = np.zeros(300, )

    # [YOUR CODE HERE]
    # Preprocessing the String
    user_input = preprocessing(user_input)

    # Tokenizing the string
    tokenized_input = get_tokens(user_input)

    # storing all the word vector of each of the
    # tokenized string in a list of arrays
    all_word_vectors = np.zeros(shape=(len(tokenized_input), 300))
    for i in range(len(tokenized_input)):
        all_word_vectors[i] = w2v(word2vec, tokenized_input[i])

    # Adding up all the word embeddings array
    embedding_sum = np.sum(all_word_vectors, axis=0)

    # Dividing word embeddings array
    # After this step we should have the embeddings array
    embedding = embedding_sum / len(tokenized_input)
    embedding = np.array(embedding)

    return embedding


# Function: vectorize_train(training_documents)
# training_documents: A list of strings
# Returns: An initialized TfidfVectorizer model, and a document-term matrix, dtype: scipy.sparse.csr.csr_matrix
def vectorize_train(training_documents):
    # Initialize the TfidfVectorizer model and document-term matrix
    vectorizer = TfidfVectorizer()
    tfidf_train = None
    # [YOUR CODE HERE]
    tfidf_train = vectorizer.fit_transform(training_documents)

    return vectorizer, tfidf_train


# Function: vectorize_test(vectorizer, user_input)
# vectorizer: A trained TFIDF vectorizer
# user_input: A string of arbitrary length
# Returns: A sparse TFIDF representation of the input string of shape (1, X), dtype: scipy.sparse.csr.csr_matrix
#
# This function computes the TFIDF representation of the input string, using
# the provided TfidfVectorizer.
def vectorize_test(vectorizer, user_input):
    # Initialize the TfidfVectorizer model and document-term matrix
    tfidf_test = None

    # [YOUR CODE HERE]
    preprocessedUserInput = preprocessing(user_input)
    tfidf_test = vectorizer.transform([preprocessedUserInput])
    return tfidf_test


# Function: instantiate_models()
# This function does not take any input
# Returns: Three instantiated machine learning models
#
# This function instantiates the three imported machine learning models, and
# returns them for later downstream use.  You do not need to train the models
# in this function.
def instantiate_models():
    logistic = None
    svm = None
    mlp = None

    # [YOUR CODE HERE]
    # [YOUR CODE HERE]
    logistic = LogisticRegression()
    svm = LinearSVC()
    mlp = MLPClassifier()

    return logistic, svm, mlp


# Function: train_model(training_documents, training_labels)
# training_data: A sparse TfIDF document-term matrix, dtype: scipy.sparse.csr.csr_matrix
# training_labels: A list of integers (0 or 1)
# Returns: A trained model
def train_nb_model(training_data, training_labels):
    # Initialize the GaussianNB model and the output label
    naive = GaussianNB()

    # Write your code here.  You will need to make use of the GaussianNB fit()
    # function.  Make sure that your training data is formatted as a dense
    # NumPy array:
    # [YOUR CODE HERE]
    training_data_array = training_data.toarray()
    naive.fit(training_data_array, training_labels)

    return naive


# Function: train_model(model, word2vec, training_documents, training_labels)
# model: An instantiated machine learning model
# word2vec: A pretrained Word2Vec model
# training_data: A list of training documents
# training_labels: A list of integers (all 0 or 1)
# Returns: A trained version of the input model
#
# This function trains an input machine learning model using averaged Word2Vec
# embeddings for the training documents.
def train_model(model, word2vec, training_documents, training_labels):
    # [YOUR CODE HERE]
    training_documents_array = np.array(training_documents)
    training_documents_trained = []
    for i in range(len(training_documents_array)):
        training_documents_trained.append(string2vec(word2vec, training_documents[i]))
    model.fit(training_documents_trained, training_labels)
    return model


# Function: test_model(model, word2vec, training_documents, training_labels)
# model: An instantiated machine learning model
# word2vec: A pretrained Word2Vec model
# test_data: A list of test documents
# test_labels: A list of integers (all 0 or 1)
# Returns: Precision, recall, F1, and accuracy values for the test data
#
# This function tests an input machine learning model by extracting features
# for each preprocessed test document and then predicting an output label for
# that document.  It compares the predicted and actual test labels and returns
# precision, recall, f1, and accuracy scores.
def test_model(model, word2vec, test_documents, test_labels):
    precision = None
    recall = None
    f1 = None
    accuracy = None

    # Write your code here:

    return precision, recall, f1, accuracy


# Function: test_nb_model(naive, vectorizer, test_documents, test_labels)
# naive: A trained GaussianNB model
# vectorizer: A trained TFIDF vectorizer
# test_documents: A list of test documents
# test_labels: A list of integers (all 0 or 1)
# Returns: Precision, recall, F1, and accuracy values for the test data
#
# This function uses the trained GaussianNB model to predict binary labels for
# (1 = unhealthy; 0 = healthy) for the test data, and then computes precision,
# recall, F1, and accuracy scores using the predicted labels.
def test_nb_model(naive, vectorizer, test_documents, test_labels):
    training_documents_array = np.array(test_documents)
    training_documents_trained = []
    for i in range(len(training_documents_array)):
        training_documents_trained.append(string2vec(vectorizer, test_documents[i]))

    pred = naive.predict(training_documents_trained)
    print(pred)
    precision = precision_score(test_labels, pred)
    recall = recall_score(test_labels, pred)
    f1 = f1_score(test_labels, pred)
    accuracy = accuracy_score(test_labels, pred)

    return round(precision, 2), round(recall, 2), round(f1, 2), round(accuracy, 2)


# ***** New in Project Part 3! *****
# Function: count_words(user_input)
# user_input: A string of arbitrary length
# Returns: An integer value
#
# This function counts the number of words in the input string.
def count_words(user_input):
    num_words = 0

    # [YOUR CODE HERE]
    no_punctuation = []
    word_count = nltk.tokenize.word_tokenize(user_input)
    print(word_count)
    for t in word_count:
        if t not in string.punctuation:
            no_punctuation.append(t.lower())
    num_words = len(no_punctuation)
    return num_words


# ***** New in Project Part 3! *****
# Function: words_per_sentence(user_input)
# user_input: A string of arbitrary length
# Returns: A floating point value
#
# This function computes the average number of words per sentence
def words_per_sentence(user_input):
    wps = 0.0

    # [YOUR CODE HERE]
    count = 0
    num_sentences = nltk.tokenize.sent_tokenize(user_input)
    for i in num_sentences:
        count += count_words(i)
    wps = count / len(num_sentences)
    return wps


# ***** New in Project Part 3! *****
# Function: get_pos_tags(user_input)
# user_input: A string of arbitrary length
# Returns: A list of (token, POS) tuples
#
# This function tags each token in the user_input with a Part of Speech (POS) tag from Penn Treebank.
def get_pos_tags(user_input):
    tagged_input = []

    # [YOUR CODE HERE]
    tokenized = nltk.tokenize.word_tokenize(user_input)
    tagged_input = nltk.pos_tag(tokenized)
    return tagged_input


# ***** New in Project Part 3! *****
# Function: get_pos_categories(tagged_input)
# tagged_input: A list of (token, POS) tuples
# Returns: Seven integers, corresponding to the number of pronouns, personal
#          pronouns, articles, past tense verbs, future tense verbs,
#          prepositions, and negations in the tagged input
#
# This function counts the number of tokens corresponding to each of six POS tag
# groups, and returns those values.  The Penn Treebag tags corresponding that
# belong to each category can be found in Table 2 of the project statement.
def get_pos_categories(tagged_input):
    num_pronouns = 0
    num_prp = 0
    num_articles = 0
    num_past = 0
    num_future = 0
    num_prep = 0

    for i in tagged_input:
        tok, pos = i
        if pos == 'PRP' or pos == 'PRP$' or pos == 'WP' or pos == 'WP$':
            num_pronouns += 1
        if pos == 'PRP':
            num_prp += 1
        if pos == 'DT':
            num_articles += 1
        if pos == 'VBD' or pos == 'VBN':
            num_past += 1
        if pos == 'MD':
            num_future += 1
        if pos == 'IN':
            num_prep += 1
    return num_pronouns, num_prp, num_articles, num_past, num_future, num_prep


# ***** New in Project Part 3! *****
# Function: count_negations(user_input)
# user_input: A string of arbitrary length
# Returns: An integer value
#
# This function counts the number of negation terms in a user input string
def count_negations(user_input):
    num_negations = 0

    # [YOUR CODE HERE]
    tokenized = nltk.word_tokenize(user_input)
    for i in tokenized:
        if "no" in i or "not" in i or "never" in i or "n't" in i:
            num_negations += 1
    return num_negations


# ***** New in Project Part 3! *****
# Function: summarize_analysis(num_words, wps, num_pronouns, num_prp, num_articles, num_past, num_future, num_prep, num_negations)
# num_words: An integer value
# wps: A floating point value
# num_pronouns: An integer value
# num_prp: An integer value
# num_articles: An integer value
# num_past: An integer value
# num_future: An integer value
# num_prep: An integer value
# num_negations: An integer value
# Returns: A list of three strings
#
# This function identifies the three most informative linguistic features from
# among the input feature values, and returns the psychological correlates for
# those features.  num_words and/or wps should be included if, and only if,
# their values exceed predetermined thresholds.  The remainder of the three
# most informative features should be filled by the highest-frequency features
# from among num_pronouns, num_prp, num_articles, num_past, num_future,
# num_prep, and num_negations.
def summarize_analysis(num_words, wps, num_pronouns, num_prp, num_articles, num_past, num_future, num_prep,
                       num_negations):
    informative_correlates = []
    iterator = 0
    # Creating a reference dictionary with keys = linguistic features, and values = psychological correlates.
    # informative_correlates should hold a subset of three values from this dictionary.
    # DO NOT change these values for autograder to work correctly
    psychological_correlates = {}
    psychological_correlates["num_words"] = "Talkativeness, verbal fluency"
    psychological_correlates["wps"] = "Verbal fluency, cognitive complexity"
    psychological_correlates["num_pronouns"] = "Informal, personal"
    psychological_correlates["num_prp"] = "Personal, social"
    psychological_correlates["num_articles"] = "Use of concrete nouns, interest in objects/things"
    psychological_correlates["num_past"] = "Focused on the past"
    psychological_correlates["num_future"] = "Future and goal-oriented"
    psychological_correlates["num_prep"] = "Education, concern with precision"
    psychological_correlates["num_negations"] = "Inhibition"

    # Set thresholds
    num_words_threshold = 100
    wps_threshold = 20
    if num_words > num_words_threshold:
        iterator += 1
        informative_correlates.append(psychological_correlates["num_words"])

    if wps > wps_threshold:
        iterator += 1
        informative_correlates.append(psychological_correlates["wps"])

    parameters = {"num_pronouns": num_pronouns, "num_prp": num_prp, "num_articles": num_articles, "num_past": num_past,
                  "num_future": num_future, "num_prep": num_prep, "num_negations": num_negations}

    parameters = sorted(parameters.items(), key=operator.itemgetter(1), reverse=True)
    print(parameters)

    for i in parameters:
        if len(informative_correlates) < 3:
            informative_correlates.append(psychological_correlates[i[0]])
            iterator += 1

    return informative_correlates


# This is your main() function.  Use this space to try out and debug your code
# using your terminal.  The code you include in this space will not be graded.
if __name__ == "__main__":
    lexicon, labels = load_as_list("dataset.csv")

    # Load the Word2Vec representations so that you can make use of it in your project.
    word2vec = load_w2v(EMBEDDING_FILE)

    # Instantiate and train the machine learning models
    logistic, svm, mlp = instantiate_models()
    logistic = train_model(logistic, word2vec, lexicon, labels)
    svm = train_model(svm, word2vec, lexicon, labels)
    mlp = train_model(mlp, word2vec, lexicon, labels)

    # Uncomment the line below to test out the w2v() function.  Make sure to
    # try a few words that are unlikely to exist in its dictionary (e.g.,
    # "covid") to see how it handles those.
    # print("Word2Vec embedding for {0}:\t{1}".format("vaccine", w2v(word2vec, "vaccine")))

    # Test the machine learning models to see how they perform on the small
    # test set provided.  Write a classification report to a CSV file with this
    # information.
    test_data, test_labels = load_as_list("test.csv")

    models = [logistic, svm, mlp]
    model_names = ["Logistic Regression", "SVM", "Multilayer Perceptron"]
    outfile = open("classification_report.csv", "w")
    outfile_writer = csv.writer(outfile)
    outfile_writer.writerow(["Name", "Precision", "Recall", "F1", "Accuracy"])  # Header row
    i = 0
    while i < len(models):  # Loop through other results
        p, r, f, a = test_model(models[i], word2vec, test_data, test_labels)
        if models[i] == None:  # Models will be null if functions have not yet been implemented
            outfile_writer.writerow([model_names[i], "N/A"])
        else:
            outfile_writer.writerow([model_names[i], p, r, f, a])
        i += 1
    outfile.close()

    # For reference, let us also compute the accuracy for the Naive Bayes model from Project Part 1
    # Fill in the code templates from your previous submission and uncomment the code below
    # vectorizer, tfidf_train = vectorize_train(lexicon)
    # lexicon = [preprocessing(d) for d in test_data]
    # tfidf_test = vectorizer.transform(lexicon)
    # naive = train_nb_model(tfidf_train, labels)
    # predictions = naive.predict(tfidf_test.toarray())
    # acc = np.sum(np.array(test_labels) == predictions) / len(test_labels)
    # print("Naive Bayes Accuracy:", acc)

    # Display a welcome message to the user, and accept a user response of
    # arbitrary length
    user_input = input(
        "Welcome to the CS 421 healthcare chatbot!  What is your name and date of birth? Enter this information in the form: First Last MM/DD/YY\n")

    # Extract the user's name and date of birth
    name, dob = extract_user_info(user_input)

    # Check the user's current health
    user_input = input(
        "Thanks {0}! I'll make a note that you were born on {1}.  How are you feeling today?\n".format(name, dob))

    # Predict whether the user is healthy or unhealthy
    w2v_test = string2vec(word2vec, user_input)

    label = None
    # label = mlp.predict(w2v_test.reshape(1, -1)) # Use this if you select one of the other models (swap mlp for svm, etc.)

    if label == 0:
        print("Great!  It sounds like you're healthy.")
    elif label == 1:
        print("Oh no!  It sounds like you're unhealthy.")
    else:
        print("Hmm, that's weird.  My classifier predicted a value of: {0}".format(label))

    # ***** New in Project Part 3! *****
    user_input = input("I'd also like to do a quick stylistic analysis. What's on your mind today?\n")

    # ***** New in Project Part 3! *****
    num_words = count_words(user_input)
    wps = words_per_sentence(user_input)
    pos_tags = get_pos_tags(user_input)
    num_pronouns, num_prp, num_articles, num_past, num_future, num_prep = get_pos_categories(pos_tags)
    num_negations = count_negations(user_input)

    # ***** New in Project Part 3! *****
    # Uncomment the code below to view your output from each individual function
    # print("num_words:\t{0}\nwps:\t{1}\npos_tags:\t{2}\nnum_pronouns:\t{3}\nnum_prp:\t{4}"
    #      "\nnum_articles:\t{5}\nnum_past:\t{6}\nnum_future:\t{7}\nnum_prep:\t{8}\nnum_negations:\t{9}".format(
    #    num_words, wps, pos_tags, num_pronouns, num_prp, num_articles, num_past, num_future, num_prep, num_negations))

    # ***** New in Project Part 3! *****
    # Generate a stylistic analysis of the user's input
    informative_correlates = summarize_analysis(num_words, wps, num_pronouns,
                                                num_prp, num_articles, num_past,
                                                num_future, num_prep, num_negations)
    print(
        "Thanks!  Based on my stylistic analysis, I've identified the following psychological correlates in your response:")
    for correlate in informative_correlates:
        print("- {0}".format(correlate))
