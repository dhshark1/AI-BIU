#Daniel Haber
#322230020
import sys
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import pairwise_distances


class collaborative_filtering:
    def __init__(self):
        self.user_based_matrix = []
        self.item_based_metrix = []
        self.users = None
        self.movies = None
        self.ratings = None
        self.movie_ids = None
        self.user_rated_dict = {}
        self.item_pred_for_evaluation = None

    def create_fake_user(self, rating):
        "*** YOUR CODE HERE ***"
        df = pd.DataFrame({'userId': [283238, 283238, 283238, 283238, 283238, 283238, 283238, 283238, 283238],
                           'movieId': [594, 364, 362, 455, 586, 673, 1186, 593, 968],
                           'rating': [5, 5, 5, 5, 5, 5, 1, 1, 1]})
        rating = pd.concat([rating, df], ignore_index=True, axis=0)
        return rating

    def create_user_based_matrix(self, data):
        self.ratings = data[0]
        self.movies = data[1]

        # for adding fake user
        # self.ratings = self.create_fake_user(self.ratings)

        user_based_matrix = self.ratings.pivot_table(index='userId', columns='movieId', values='rating')
        user_rated_ids = list(zip(sp.coo_matrix(user_based_matrix.notnull()).nonzero()))
        self.users = sorted(list(dict.fromkeys(self.ratings['userId'])))
        self.movie_ids = sorted(list(dict.fromkeys(self.ratings['movieId'])))
        realUsers = list(user_rated_ids[0][0])
        realMovies = list(user_rated_ids[1][0])
        for x in range(len(realUsers)):
            index_of_user = realUsers[x]
            real_user = self.users[index_of_user]
            index_of_movie = realMovies[x]
            real_movie = self.movie_ids[index_of_movie]
            if real_user in self.user_rated_dict:
                self.user_rated_dict[real_user].append(real_movie)
            else:
                self.user_rated_dict[real_user] = [real_movie]
        mean_user_rating = user_based_matrix.mean(axis=1).to_numpy().reshape(-1, 1)
        ratings_diff = (user_based_matrix - mean_user_rating)
        ratings_diff[np.isnan(ratings_diff)] = 0
        user_similarity = 1 - pairwise_distances(ratings_diff, metric='cosine')
        pred = mean_user_rating + user_similarity.dot(ratings_diff) / np.array([np.abs(user_similarity).sum(axis=1)]).T
        pred = np.nan_to_num(pred)
        self.user_based_matrix = pred

    def create_item_based_matrix(self, data):
        "*** YOUR CODE HERE ***"
        self.ratings = data[0]
        self.movies = data[1]

        item_based_matrix = self.ratings.pivot_table(index='userId', columns='movieId', values='rating')
        user_rated_ids = list(zip(sp.coo_matrix(item_based_matrix.notnull()).nonzero()))
        self.users = sorted(list(dict.fromkeys(self.ratings['userId'])))
        self.movie_ids = sorted(list(dict.fromkeys(self.ratings['movieId'])))
        realUsers = list(user_rated_ids[0][0])
        realMovies = list(user_rated_ids[1][0])
        for x in range(len(realUsers)):
            index_of_user = realUsers[x]
            real_user = self.users[index_of_user]
            index_of_movie = realMovies[x]
            real_movie = self.movie_ids[index_of_movie]
            if real_user in self.user_rated_dict:
                self.user_rated_dict[real_user].append(real_movie)
            else:
                self.user_rated_dict[real_user] = [real_movie]
        mean_user_rating = item_based_matrix.mean(axis=1).to_numpy().reshape(-1, 1)
        ratings_diff = (item_based_matrix - mean_user_rating)
        ratings_diff[np.isnan(ratings_diff)] = 0
        item_similarity = 1 - pairwise_distances(ratings_diff.T, metric='cosine')
        pred = mean_user_rating + ratings_diff.dot(item_similarity) / np.array([np.abs(item_similarity).sum(axis=1)])
        self.item_pred_for_evaluation = np.nan_to_num(pred)
        self.item_based_metrix = pred

    def predict_movies(self, user_id, k, is_user_based=False):
        "*** YOUR CODE HERE ***"
        if is_user_based:
            index_of_user = self.users.index(int(user_id))
            ratings_with_movie_id = sorted(zip(self.user_based_matrix[index_of_user], self.movie_ids), key=lambda x: x[0],
                                           reverse=True)
            to_remove = []
            for i in ratings_with_movie_id:
                if i[1] in self.user_rated_dict[int(user_id)]:
                    to_remove.append(i)
            ratings_with_movie_id = [x for x in ratings_with_movie_id if x not in to_remove]
            for j in range(k):
                index_of_movie = self.movie_ids.index(ratings_with_movie_id[j][1])
                print(self.movies['title'][index_of_movie])
        else:

            ratings_with_movie_id = sorted(zip(pd.DataFrame(self.item_based_metrix).loc[int(user_id)], self.movie_ids), key=lambda x: x[0],
                                           reverse=True)
            to_remove = []
            for i in ratings_with_movie_id:
                if i[1] in self.user_rated_dict[int(user_id)]:
                    to_remove.append(i)
            ratings_with_movie_id = [x for x in ratings_with_movie_id if x not in to_remove]
            for j in range(k):
                index_of_movie = self.movie_ids.index(ratings_with_movie_id[j][1])
                print(self.movies['title'][index_of_movie])

    def user_predict_movie_ids(self, user_id, k):
        movie_id = []
        index_of_user = self.users.index(int(user_id))
        ratings_with_movie_id = sorted(zip(self.user_based_matrix[index_of_user], self.movie_ids), key=lambda x: x[0],
                                       reverse=True)
        to_remove = []
        for i in ratings_with_movie_id:
            if i[1] in self.user_rated_dict[int(user_id)]:
                to_remove.append(i)
        new_ratings_with_movie_id = [x for x in ratings_with_movie_id if x not in to_remove]
        for j in range(k):
            movie_id.append(new_ratings_with_movie_id[j][1])
        return movie_id

    def item_predict_movie_ids(self, user_id, k):
        movie_id = []
        ratings_with_movie_id = sorted(zip(self.item_based_metrix.loc[int(user_id)], self.movie_ids), key=lambda x: x[0],
                                       reverse=True)
        to_remove = []
        for i in ratings_with_movie_id:
            if i[1] in self.user_rated_dict[int(user_id)]:
                to_remove.append(i)
        new_ratings_with_movie_id = [x for x in ratings_with_movie_id if x not in to_remove]
        for j in range(k):
            movie_id.append(new_ratings_with_movie_id[j][1])
        return movie_id