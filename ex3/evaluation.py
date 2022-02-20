#Daniel Haber
#322230020
# Import Pandas
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt


def precision_10(test_set, cf, is_user_based = False):
    "*** YOUR CODE HERE ***"
    hits = 0
    test_set = test_set.sort_values(by='userId')
    test_set = test_set.loc[test_set['rating'] >= 4]
    unique_user_ids = test_set['userId'].unique()
    top_10_predictions = []
    if is_user_based:
        for user_id in unique_user_ids:
            top_10_predictions.append((user_id, cf.user_predict_movie_ids(user_id, 10)))
    else:
        for user_id in unique_user_ids:
            top_10_predictions.append((user_id, cf.item_predict_movie_ids(user_id, 10)))
    for prediction in top_10_predictions:
        ratings_of_user = test_set.loc[test_set['userId'] == prediction[0]]
        hits += (ratings_of_user['movieId'].isin(prediction[1]).sum()/10)
    val = hits / unique_user_ids.size
    print("Precision_k: " + str(val))


def ARHA(test_set, cf, is_user_based = False):
    "*** YOUR CODE HERE ***"
    test_ratings = test_set.pivot_table(index='userId', columns='movieId', values='rating')
    unique_user_ids = test_set['userId'].unique()
    top_10_predictions = []
    if is_user_based:
        for user_id in unique_user_ids:
            top_10_predictions.append((user_id, cf.user_predict_movie_ids(user_id, 10)))
    else:
        for user_id in unique_user_ids:
            top_10_predictions.append((user_id, cf.item_predict_movie_ids(user_id, 10)))
    hits = 0.0
    for p in top_10_predictions:
        k = 1
        for j in p[1]:
            if test_ratings.at[p[0], j] >= 4:
                hits += (1/k)
            k += 1
    val = hits / unique_user_ids.size
    print("ARHR: " + str(val))


def RSME(test_set, cf, is_user_based = False):
    "*** YOUR CODE HERE ***"
    test_ratings = test_set.pivot_table(index='userId', columns='movieId', values='rating')
    test_ratings_numpy = test_ratings.to_numpy()
    if is_user_based:
        pred_ratings = cf.user_based_matrix
    else:
        pred_ratings = cf.item_pred_for_evaluation
    unique_user_ids = test_set['userId'].unique()
    predicted_i = []
    actual_i = []
    for user in unique_user_ids:
        index_of_user = cf.users.index(user)
        test_user_ratings = test_ratings_numpy[index_of_user]
        indices_not_nan = np.argwhere(np.logical_not(np.isnan(test_user_ratings)))
        predicted_i.extend(pred_ratings[index_of_user][indices_not_nan].tolist())
        actual_i.extend(test_user_ratings[indices_not_nan].tolist())
    val = sqrt(mean_squared_error(predicted_i, actual_i))

    print("RMSE: " + str(val))


