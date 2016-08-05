import pandas as pd
from sklearn.decomposition import NMF
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import time
import datetime


def log(logfile, message):
    with open(logfile, "a") as f:
        f.write(message)
        f.write('\n')


def load_data(filename):
    """
    Load the data from a csv file.
    File must have columns in format:
    user_id, timestamp, name_of_food

    :param filename: name of the csv file
    :return: A list of documents for each user where the tags are the name of the foods
    """
    print('[LOG] - Loading file')
    column_names = ['user_id', 'date', 'name_of_food']
    df = pd.read_csv(filename, names=column_names,
                     converters={'name_of_food': lambda x: x.strip().replace(' ', '_') + ' '}, header=0)
    print('[LOG] - Creating documents')
    documents = df.groupby(['user_id'])['name_of_food'].sum().apply(lambda s: s.strip()).values
    print('[LOG] - DONE')
    return documents


def get_data_matrix(filename, vectorizer='tfidf'):
    """
    Given a csv file, will return a matrix of tfidf values for each
    tag in each documents. The case here being each food for each user where
    the users are the document ids and the food names are the tags.

    :param filename: csv file of food logs
    :param vectorizer: tfidf (default) or 'count' for raw counts
    :return:
    """
    documents = load_data(filename)
    vectorizer = TfidfVectorizer(sublinear_tf=True) if vectorizer == 'tfidf' else CountVectorizer()
    matrix = vectorizer.fit_transform(documents).toarray()

    print('[LOG] - DONE')
    return matrix


def split_test_train(original_matrix, train_perc):
    """
    :param original_matrix: numpy matrix of user-food interactions
    :param train_perc: how much of the data should be used for training
    :return: training matrix, testing matrix
    """
    print('[LOG] - Creating test-train split')
    test_mask = np.zeros(original_matrix.shape)
    train_mask = np.zeros(original_matrix.shape)

    for row_num in range(original_matrix.shape[0]):
        row = original_matrix[row_num].tolist()
        nonzero = [index for index, elem in enumerate(row) if elem != 0]
        np.random.shuffle(nonzero)
        split = int(len(nonzero) * train_perc)
        train_i, test_i = nonzero[:split], nonzero[split:]

        for index in train_i:
            train_mask[row_num, index] = row[index]

        for index in test_i:
            test_mask[row_num, index] = row[index]
    print('[LOG] - DONE')
    return train_mask, test_mask


def get_model(latent_factors, train):
    """
    Creates a model using the training matrix of user-food interactions
    :param latent_factors:
    :param train:
    :return: training model matrix
    """
    print('[LOG] - Factorizing train matrix')
    nmf = NMF(n_components=latent_factors, beta=0.001, eta=0.0001, init='nndsvd', max_iter=200000, nls_max_iter=2000000,
              random_state=0, sparseness=None, tol=0.001)
    W = nmf.fit_transform(train)
    H = nmf.components_
    train_model = np.dot(W, H)
    print('[LOG] - DONE')
    return train_model


def calculate_mpr(train_model, test_mask):
    """
    Compares the training model matrix against the test matrix to generate
    a mean percentile ranking for the recommender system.

    :param train_model:
    :param test_mask:
    :return: [mpr, mpr numerator, mpr denominator]
    """
    print('[LOG] - Calculating MPR')
    top_sum = 0.0
    bot_sum = 0.0
    for row_num in range(train_model.shape[0]):
        '''
        We will zero out all items in this row except for the ones will
        be testing on
        '''
        scores_from_trained_model = train_model[row_num].tolist()
        scores_from_test_data = test_mask[row_num].tolist()
        '''
        Convert these rows to lil_matrix so we can append them so the
        resulting structure looks like:

         [[score_from_trained_model_0, score_from_original_data_0, 0],
          [score_from_trained_model_1, score_from_original_data_1, 1],
          [score_from_trained_model_2, score_from_original_data_2, 2], ...]
        '''
        combined_scores = []
        for col_num in range(len(scores_from_test_data)):

            if scores_from_test_data[col_num] > 0:
                combined_score = [scores_from_trained_model[col_num], scores_from_test_data[col_num]]
                combined_scores.append(combined_score)
        '''
        Sort the combined scores by the scores from the trained model
        '''
        combined_scores.sort(key=lambda x: x[0])
        combined_scores.reverse()  # reverse this so that the highest recommended items come first

        # calculate percentiles
        for i, combined_score in enumerate(combined_scores):
            if i == 0:
                combined_score[0] = 0.0
            else:
                combined_score[0] = float(i) * (1. / (len(combined_scores) - 1.0))

            top_sum = top_sum + (combined_score[1] * combined_score[0])
            bot_sum = bot_sum + combined_score[1]

    print('[LOG] - DONE')
    return [top_sum / bot_sum if bot_sum != 0 else -1, top_sum, bot_sum]


def run(files = ["PEACH_foodDiary_1Q2015.csv"],
        train_percs=[0.9, 0.8, 0.7, 0.6],
        latent=[1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
        logfile=datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S') + '.csv'):
    """
    Runner for this recommender system. Given the parameters, will run all permutations
    of them and log the results.



    :param files: an array of the relative paths to csv files with the data
    :param train_percs: array of what percentage of the dataset should be used for training
    :param latent: an array of the number of latent factors that should be used in the matrix factorization
    :param logfile: file to logging
    :return:
    """
    log(logfile, "FILE, TRAIN_PERCENTAGE, LATENT, MPR, MPR_NUM, MPR_DNUM")
    for i, f in enumerate(files):
        tfidf_matrix = get_data_matrix(f, vectorizer='count')
        for perc in train_percs:
            train_mat, test_mat = split_test_train(tfidf_matrix, perc)
            for l in latent:
                model_mat = get_model(l, train_mat)
                metrics = calculate_mpr(model_mat, test_mat)
                print("{}, {}, {}, {}, {}, {}".format(f, perc, l, metrics[0], metrics[1], metrics[2]))
                log(logfile, "{}, {}, {}, {}, {}, {}".format(f, perc, l, metrics[0], metrics[1], metrics[2]))
