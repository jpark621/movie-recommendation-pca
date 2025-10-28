# # Find most popular movies
# from collections import defaultdict

# d = defaultdict(int)
# with open("ratings.csv", 'r') as f:
# 	f.readline()
# 	while f:
# 		line = f.readline()
# 		if line == "":
# 			break
# 		movie = line.split(",")[1]
# 		d[movie] += 1

# movies = list(d.items())
# movies.sort(key=lambda items: items[1], reverse=True)

# with open("top_movies.csv", 'w') as f:
# 	for k, v in movies:
# 		f.write(str(k) + "," + str(v) + "\n")

# Filter by the top k movies
k = 5000
top_k_movies = set()
with open("top_movies.csv", 'r') as f:
	for _ in range(k):
		movie = f.readline().split(",")[0]
		top_k_movies.add(movie)

with open("ratings_" + str(k) + "movies.csv", 'w') as fw:
	with open("ratings.csv", 'r') as fr:
		while fr:
			line = fr.readline()
			if line == "":
				break
			if line.split(",")[1] in top_k_movies:
				fw.write(line)

# Create user-movie recommendation matrix
users = set()
movies = set()
with open("ratings_" + str(k) + "movies.csv", 'r') as f:
	while f:
		line = f.readline()
		if line == "":
			break
		user, movie, _, _ = line.split(",")
		users.add(user)
		movies.add(movie)
N = len(users)
M = len(movies)

import numpy as np
A = np.full((N, M), None)

movie_encoder = {}
with open("top_movies.csv", 'r') as f:
	for i in range(k):
		movie, _ = f.readline().split(",")
		movie_encoder[movie] = i

user_encoder = {}
index = 0
with open("ratings_" + str(k) + "movies.csv", 'r') as f:
	while f:
		line = f.readline()
		if line == "":
			break

		user = line.split(",")[0]
		if user not in user_encoder:
			user_encoder[user] = index
			index += 1

with open("ratings_" + str(k) + "movies.csv", 'r') as f:
	while f:
		line = f.readline()
		if line == "":
			break
		user, movie, rating, _ = line.split(",")

		if float(rating) >= 2.5:
			A[user_encoder[user] - 1,movie_encoder[movie]] = 1
		else:
			A[user_encoder[user] - 1,movie_encoder[movie]] = 0

## Impute with the mean
for j in range(M):
	total = 0
	for i in range(N):
		if A[i, j] == 1:
			total += 1
	mean = total / N

	for i in range(N):
		if A[i, j] is None:
			A[i, j] = mean
A = A.astype(float)
np.save("user_movie_dataset_" + str(k) + "movies.npy", A)

# Create covariance matrix
import numpy as np
A = np.load("user_movie_dataset_" + str(k) + "movies.npy")
Ac = A - A.mean(axis=0)
Sigma = 1 / A.shape[0] * Ac.T @ Ac

# Simultaneous power-iteration
def simultaneous_power_iteration(A, tol=0.0001, max_iter=100000):
	Q, R = np.linalg.qr(A)
	prev = np.zeros(A.shape)
	for i in range(max_iter):
		prev = Q[:]
		X = A @ Q
		Q, R = np.linalg.qr(X)
		if np.allclose(Q, prev, atol=tol):
			break
	return Q, i

eigenvectors, num_iterations = simultaneous_power_iteration(Sigma)

