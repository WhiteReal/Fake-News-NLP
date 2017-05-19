from utils.dataset import DataSet
from utils.generate_test_splits import split
from utils.score import report_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model

dataset = DataSet()
data_splits = split(dataset)

training_data = data_splits['training']
dev_data = data_splits['dev']
test_data = data_splits['test']
#print(training_data[1].keys())
#print(training_data[1]["Headline"])

def stop_list(filepath):
# load stop list
    stoplist = []
    with open(filepath, "r") as stop_file:
        for line in stop_file:
            stoplist.append(line.strip())
    return stoplist


def tf_idf(text, stop_list = None):
# calculate the tf-idf matrix, a stop list could be added
    count_vect = CountVectorizer(stop_words=stop_list)
    X_train_counts = count_vect.fit_transform(text)
    tfidf_transformer = TfidfTransformer().fit(X_train_counts)
    X_train_tfidf = tfidf_transformer.transform(X_train_counts)
    return X_train_tfidf

def similarity(training_data,stoplist):
# calculate a cosine_similarity matrix between headlines and bodies(articles)
# extract headlines
    headline = []
    for stance in training_data:
        if stance["Headline"] not in headline:
            headline.append(stance["Headline"])
    len_head = len(headline)
# extract articles
    articles = []
    for body_id in dataset.articles:
        articles.append(dataset.articles[body_id])
    len_body = len(articles)
# merge the headlines and articles into one list
    headline.extend(articles)

# calculate the tf-idf matrix
    tfidf_matrix = tf_idf(headline, stoplist)
    
    tfidf_array = tfidf_matrix.toarray()
# split the array into head and body
    tfidf_head = tfidf_array[0:len_head]
    tfidf_body = tfidf_array[len_head:len_head+len_body]
# calculate cosine similarity
    similar_matrix = cosine_similarity(tfidf_head, tfidf_body)
    return similar_matrix, headline, articles


def train_threshold(training_data, similar_matrix, headline, articles):
# a key function in baseline, find a best threshold of similarity which could get a a good
# classification between related and unrelated
    thre = 0.5
    for i in range(10):
        for stance in training_data:
            test_head = stance["Headline"]
            test_body = dataset.articles[stance["Body ID"]]
            index_row = headline.index(test_head)
            index_column = articles.index(test_body)
            simil = similar_matrix[index_row, index_column]
            if simil < thre and stance["Stance"] != "unrelated":
                thre -= 0.01
            elif simil > thre and stance["Stance"] == "unrelated":
                thre += 0.01
    return thre


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
# This is a function of plotting learning curve
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def train_related(training_data, stoplist):
# This is a process to train a neural network model to divide the related data
# The unrelated data we could get by similarity, now we focus on the related data
# extract the related data (agree, disagree, dicuss) from training_data, combine their head and body
    related = []
    label = []
    for stance in training_data:
        if stance["Stance"] != "unrelated":
            related.append(stance["Headline"] + " " + dataset.articles[stance["Body ID"]])
            label.append(stance["Stance"])
    count_vect = CountVectorizer(stop_words=stoplist)
    vec_result = count_vect.fit_transform(related).toarray()
    
    title = "Learning Curves (MLP)"
    
# train the neutral network model
    clf = MLPClassifier(solver='lbfgs')
# train the logistical regression
#   clf = linear_model.LogisticRegression()
    plt = plot_learning_curve(clf, title, vec_result, label, n_jobs=4)
    plt.show()
    clf.fit(vec_result, label)
# save the trained model
    #joblib.dump(clf, "model_solver_lbfgs.m")
# load the trained model
    #clf = joblib.load("model/model_solver_lbfgs.m")
# return two models: the MLP and word countVector
    return clf,count_vect

def test(test_data, similar_matrix, clf, count_vect):
# Now we will combine the training result: threshold and neutral network to predict test data
    appro = []
# Extract the head and body, and then find their similarity in previous matrix
    for stance in test_data:
        test_head = stance["Headline"]
        test_body = dataset.articles[stance["Body ID"]]
        index_row = headline.index(test_head)
        index_column = articles.index(test_body)
        simil = similar_matrix[index_row, index_column]
# if the value is less than threshold, predict it as "unrelated"
        if simil < thre:
            appro.append("unrelated")
# if it is more than threshold, use mlp to make further predicrion
        elif simil >= thre:
            whole = [test_head + " " + test_body]
            vec = count_vect.transform(whole).toarray()
            appro.append(clf.predict(vec))
# return the predicted list
    return appro
    
if __name__ == '__main__':
    stoplist = stop_list("stop_list.txt")
    
    similar_matrix, headline, articles = similarity(training_data, stoplist)
    
    thre = train_threshold(training_data, similar_matrix, headline, articles)
    print("similarity threshold:" , thre)
    
    clf, count_vect = train_related(training_data, stoplist)

    appro = test(test_data, similar_matrix, clf, count_vect)
# get the actual value of test data            
    actual = [stance['Stance'] for stance in test_data]
    count = 0      
    for i in range(len(actual)):
        if actual[i] == appro[i]:
            count += 1
    print("accuracy:", count/len(actual))
    
    report_score(actual, appro)


        
        
    


    
    
    
