#Daniel Haber
#322230020
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def watch_data_info(data):
    for d in data:
        # This function returns the first 5 rows for the object based on position.
        # It is useful for quickly testing if your object has the right type of data in it.
        print(d.head())

        # This method prints information about a DataFrame including the index dtype and column dtypes, non-null values and memory usage.
        print(d.info())

        # Descriptive statistics include those that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values.
        print(d.describe(include='all').transpose())


def print_data(data):
    "*** YOUR CODE HERE ***"
    print("the number of ratings in the file is", len(data[0]))
    uniqueUserIDs = data[0]['userId'].value_counts(ascending=True)
    uniqueMovieIDs = data[0]['movieId'].value_counts(ascending=True)
    print("the number of unique userID's in ratings is ", len(uniqueUserIDs))
    print("the number of unique movies in ratings is ", len(uniqueMovieIDs))
    print("the movie with the most ratings is", uniqueMovieIDs.iloc[-1])
    print("the movie with the least ratings is", uniqueMovieIDs.iloc[0])
    print("the user with the most ratings is", uniqueUserIDs.iloc[-1])
    print("the user with the least ratings is", uniqueUserIDs.iloc[0])

    # sys.exit(1)


def plot_data(data, plot=True):
    "*** YOUR CODE HERE ***"
    plt.figure(figsize=(16, 6))
    data[0]['rating'].value_counts(normalize=True).sort_index(ascending=True).plot(kind='line')
    plt.title('Distribution of Movie Ratings by ID', weight='bold')
    plt.xlabel('Rating', weight='bold')
    plt.ylabel('Probability', weight='bold')

    plt.savefig('movie_distribution.png')
    if plot:
        plt.show()
    # sys.exit(1)
