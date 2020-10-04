import numpy as np
import scipy
import matplotlib.pyplot as plt 
import seaborn as sns
import random
from tqdm import tqdm
import time
import pandas as pd

def recommend(uid, data, model, top_n = 100):
    """
    Returns the mean and covariance matrix of the demeaned dataset X (e.g. for PCA)
    
    Parameters
    ----------
    uid : int
        user id
    data : surprise object with data
        The entire system, ratings of users (Constructed with reader from surprise)
    model : susrprise object
        Trained algorithm
    top_n : int
        The number of movies to recommend

    Returns
    -------
    pd.DataFrame
        recommended movies
    pd.DataFram
        predicted ratings for the recommended movies
    data_update
        predicted movies and ratings in the movielens format (uid, iid, rating)
    
    """
    all_movie_ids = data.df['iid'].unique()
    uid_rated = data.df[data.df['uid'] == uid]['iid']
    movies_to_recommend = np.setdiff1d(all_movie_ids, uid_rated)
    if len(movies_to_recommend) == 0:
        print('NO MOVIES TO RECOMMEND!')
    prediction_set = [[uid, iid, 0] for iid in movies_to_recommend] #here 0 is arbitrary, ratings don't matter
    predictions = model.test(prediction_set)
    pred_ratings = np.array([pred.est for pred in predictions])
    top = pred_ratings.argsort()[::-1][:top_n]
    data_update = pd.DataFrame([[uid, movies_to_recommend[top][i], pred_ratings[top][i]] for i in range(top_n)], columns = ['uid', 'iid', 'rating'])
    return movies_to_recommend[top], pred_ratings[top], data_update#len(movies_to_recommend), len(all_movie_ids) 

from tqdm import tqdm
import time


def simulate(alg, data, sample_users, n_epochs = 10, top_n = 15, lower_rating = 0.5, upper_rating = 5.0):
    #Note that data changes inplace!
    evolution_of_movies = []
    evolution_of_ratings = []
    #First prediction
    for epoch in tqdm(range(n_epochs)):
        print('Epoch #{}'.format(epoch))
        user_top_matrix_per_epoch = []
        user_top_rating_matrix_per_epoch = []
        start_fit = time.time()
        print('Shape of the data:', data.df.shape)
        model = alg.fit(data.build_full_trainset())
        end_fit = time.time()
        print('    Fit took {} seconds'.format(end_fit-start_fit))
        #Recommend for everyone
        data_new = data.df.copy()
        start_predict = time.time()
        for uid in sample_users:
            iid_recommended, ratings_recommended, data_update = recommend(uid, data, model, top_n = top_n)
            user_top_matrix_per_epoch.append(iid_recommended)
            user_top_rating_matrix_per_epoch.append(ratings_recommended)
#             print(data_new.columns, data_update.columns)
            data_new = data_new.append([data_update]).reset_index(drop = True)
#             print(data_new.shape)
        end_predict = time.time()
        print('    Predict took {} seconds'.format(end_predict-start_predict))
        user_top_df_per_epoch = pd.DataFrame(user_top_matrix_per_epoch, index = ['uid_{}'.format(i) for i in sample_users], 
                     columns = ['top_'+str(i+1) for i in range(top_n)])
        user_top_rating_df_per_epoch = pd.DataFrame(user_top_rating_matrix_per_epoch, index = ['uid_{}'.format(i) for i in sample_users], 
                     columns = ['top_'+str(i+1) for i in range(top_n)])
        evolution_of_movies.append(user_top_df_per_epoch)
        evolution_of_ratings.append(user_top_rating_df_per_epoch)
        print('Evolution of movies and ratings appended')
        start_update_data = time.time()
        reader = surprise.Reader(rating_scale = (lower_rating, upper_rating))
        data = surprise.Dataset.load_from_df(data_new, reader)
        end_update_data = time.time()
        print('    Update_data took {} seconds'.format(end_update_data-start_update_data))
        print('Shape:', data.df.shape)
    return evolution_of_movies, evolution_of_ratings

if __name__ == "__main__":
    pass
