# movie-recommendation-pca
Use Simultaneous Power-Iteration on MovieLens

# Simultaneous Power-Iteration
PCA is found by computing the eigenvalues / eigenvectors of the covariance matrix. Power-iteration is a computational method of finding an eigenvector by continuously multiplying by A (in this case, your covariance matrix) until convergence. Simultaneous power-iteration uses QR-factorization to compute all basis at once.

If there are duplicate eigenvalues, this method will not converge. This is not a production-ready method of performing PCA.

As such, when applying to MovieLens, we will begin with only one movie in the user-movie matrix and grow the number of movies. This corresponds to the size of our covariance matrix (num_movies x num_movies).

Architecturally, we will:

  write a filtered csv file -> compute the dataset matrix -> compute the covariance matrix -> run simultaneous power iteration on this matrix

  O(num_ratings)            ->  O(num_users * num_movies) -> O(num_users * num_movies) -> O(num_movies^3)               

since simultaneous power-iteration uses QR-factorization, which is O(n^3).

## On imputing values
When creating the dataset, we want to write the user-movie matrix as "would I recommend this movie to this user". If the movie received a good rating, we recommend. If the movie received a bad rating, we do not recommend. The question is what if the movie did not receive a rating?

In this case, we use the mean of the movie. "I might watch the movie". If we recommend the movie to more people, we should also recommend the movie to more viewers. Computationally, we do not want this value to contribute to the covariance matrix, which occurs at the mean.

## Power-iteration convergence

| k | Number of iterations | Time |
| - | -------------------- | ---- |
| 2 | 18  | ? |
| 3 | 52  | ? |
| 5 | 101 | ? |
| 10 | 179 | ? |
| 50 | 777 | ? |
| 100 | 2352 | ? |
| 500 | 3720 | 4:55 |
| 1000 | 5011 | 12:15 |

The number of iterations seems to scale linearly with the number of dimensions. It works!

# Resources

  * ratings.csv can be found in the MovieLens website ([link](https://grouplens.org/datasets/movielens/)) under ml-32m.zip.
