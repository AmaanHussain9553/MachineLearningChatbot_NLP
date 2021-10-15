# CS421: Natural Language Processing
# University of Illinois at Chicago
# Fall 2021
# Project Part 1
#
# Do not rename/delete any functions or global variables provided in this template and write your solution
# in the specified sections. Use the main function to test your code when running it from a terminal.
# Avoid writing that code in the global scope; however, you should write additional functions/classes
# as needed in the global scope. These templates may also contain important information and/or examples
# in comments so please read them carefully. If you want to use external packages (not specified in
# the assignment) then you need prior approval from course staff.
# This part of the assignment will be graded automatically using Gradescope.
# =========================================================================================================


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
import string
import re


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


def checkUserName():
    return r"^[A-Z][A-Za-z\.\&\-\']*$"


def checkDate():
    return r"^(0[1-9]|1[012])[- /.](0[1-9]|[12][0-9]|3[01])[- /.]\d\d$"


# Function: extract_user_info(user_input), see project statement for more details
# user_input: A string of arbitrary length
# Returns: Two strings (a name, and a date of birth formatted as MM/DD/YY)
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
# Args:
#   user_input: A string of arbitrary length
# Returns: A string of arbitrary length
def preprocessing(user_input):
    modified_input = ""
    # [YOUR CODE HERE]
    # creates a token of the user input
    tokenized_string = get_tokens(user_input)
    # print("Printing tokenized string: ")
    # print(tokenized_string)

    # sets the punctuations possible and removes them from list of tokens
    for i in range(len(tokenized_string)):
        if tokenized_string[i] in string.punctuation and len(tokenized_string[i]) == 1:
            tokenized_string[i] = ""

    try:
        while True:
            tokenized_string.remove("")
    except ValueError:
        pass

    # print("Printing tokenized string after removing punctuations: ")
    # print(tokenized_string)

    # Converting all tokens to lowercase
    for i in range(len(tokenized_string)):
        tokenized_string[i] = tokenized_string[i].lower()
    # print("Printing tokenized string after lowercase: ")
    # print(tokenized_string)

    # Adding the preprocessed list of strings into a single string
    for i in range(len(tokenized_string)):
        if modified_input == "":
            modified_input = modified_input + tokenized_string[i]
        else:
            modified_input = modified_input + " " + tokenized_string[i]

    # print("Printing final string: " + modified_input)

    return modified_input


# Function: vectorize_train(training_documents)
# training_documents: A list of strings
# Returns: An initialized TfidfVectorizer model, and a document-term matrix, dtype: scipy.sparse.csr.csr_matrix
def vectorize_train(training_documents):
    # Initialize the TfidfVectorizer model and document-term matrix
    vectorizer = TfidfVectorizer()
    tfidf_train = None
    # [YOUR CODE HERE]
    tfidf_train = vectorizer.fit_transform(training_documents)
    # print(vectorizer.vocabulary_)
    # print(tfidf_train)
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


# Function: train_model(training_documents, training_labels)
# training_data: A sparse TfIDF document-term matrix, dtype: scipy.sparse.csr.csr_matrix
# training_labels: A list of integers (0 or 1)
# Returns: A trained model
def train_model(training_data, training_labels):
    # Initialize the GaussianNB model and the output label
    naive = GaussianNB()

    # Write your code here.  You will need to make use of the GaussianNB fit()
    # function.  Make sure that your training data is formatted as a dense
    # NumPy array:
    # [YOUR CODE HERE]
    # print(training_data)
    training_data_array = training_data.toarray()
    naive.fit(training_data_array, training_labels)
    # print(naive)
    return naive


# Function: get_model_prediction(naive, tfidf_test)
# naive: A trained GaussianNB model
# tfidf_test: A sparse TFIDF representation of the input string of shape (1, X), dtype: scipy.sparse.csr.csr_matrix
# Returns: A predicted label for the provided test data (int)
def get_model_prediction(naive, tfidf_test):
    # Initialize the GaussianNB model and the output label
    label = 0

    # Write your code here.  You will need to make use of the GaussianNB
    # predict() function.  Make sure that your training data is formatted as a
    # dense NumPy array
    # [YOUR CODE HERE]
    tfidf_test_array = tfidf_test.toarray()
    label = naive.predict(tfidf_test_array)

    return label


# This is your main() function.  Use this space to try out and debug your code
# using your terminal.
if __name__ == "__main__":

    # Display a welcome message to the user, and accept a user response of
    # arbitrary length
    user_input = input(
        "Welcome to the CS 421 healthcare chatbot!  What is your name and date of birth? Enter this information in the "
        "form: First Last MM/DD/YY\n")

    # Extract the user's name and date of birth
    name, dob = extract_user_info(user_input)

    while name == "":
        # User did not enter name and date correctly
        user_input = input(
            "Name and date of birth not entered correctly!! "
            "Enter this information in the form: First Last MM/DD/YY\n")
        # Extract the user's name and date of birth
        name, dob = extract_user_info(user_input)

    # Check the user's current health
    user_input = input("Thanks {0}! I'll make a note that you were born on {1}.  "
                       "How are you feeling today?\n".format(name, dob))

    preprocessing(user_input)

    # Set things up ahead of time by training the TfidfVectorizer and Naive
    # Bayes model
    lexicon, labels = load_as_list("dataset.csv")
    vectorizer, tfidf_train = vectorize_train(lexicon)
    naive = train_model(tfidf_train, labels)

    # Predict whether the user is healthy or unhealthy

    tfidf_test = vectorize_test(vectorizer, user_input)

    label = get_model_prediction(naive, tfidf_test)
    if label == 0:
        print("Great!  It sounds like you're healthy.")
    elif label == 1:
        print("Oh no!  It sounds like you're unhealthy.")
    else:
        print("Hmm, that's weird.  My classifier predicted a value of: {0}".format(label))
