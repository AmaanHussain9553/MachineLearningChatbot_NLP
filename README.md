# Healthcare Chatbot

This is a chatbot that creates four different kinds of machine learning models (Logistic Regression, GaussianNB, Support-Vector-Machine (SVM), and Multilayer Perceptron (MLP) and uses a word2vec .pkl that 
for any word creates the corresponding word embeddings in the form of a vector. It then trains the model against training documents and training labels of key medical vocabulary and tells us if 
the user is potentially sick or not. Furthermore, it tests all three models by calculating their precision_score, recall_score, f1_score and accuracy_score and stores it in the "classification.csv"


## Healthcare Chatbot Part 1

### Libraries and software used:
* Used Pycharm to run the program. 
* Imported Libraries: Scikit-Learn (TFidf Vectorizer, GaussianNB), Numpy, Pandas

### Methods
* Two regex methods to extract user's name and DOB at the welcome message
* Preprocessing method that lowercases and remove punctuations from a string
* Using Tfidf Vectorizer creates a document-term matrix which is fitted and transformed based on the training documents
* Train model and get model methods that trains based on the Gaussian NB model and then predicts label or "healthy" or "unhealthy" based on user's input

## Healthcare Chatbot Part 2

### Libraries and software used:
* Used Pycharm to run the program. 
* Imported Libraries: Scikit-Learn (Logistic Regression, SVM, MLP), Numpy, Pandas, csv, pickle

### Methods
* word2Vec method that takes a string and runs it against the entire data corpus to produce a word embedding
* string2Vec uses workd2vec to doing for the string, requires preprocessing and tokeninzing the string
* Method the instantiate the three different models and a train model function that trains based on the model parameter and the training and label documents provided
* Test model function that computes the precision_score, accuracy_score, recall_score, and f1_score



### This is an ongoing implementation that will be further developed in the coming months as of October 2021
