import pandas as pd
from surprise import Dataset, Reader, SVD, BaselineOnly
from surprise.model_selection import cross_validate, train_test_split
from surprise import accuracy

# Load the MovieLens dataset
ratings = pd.read_csv('data/ratings.csv')

# Create a Surprise dataset from the ratings DataFrame
reader = Reader(rating_scale = (0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Split the dataset into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Baseline model using the BaselineOnly algorithm
baseline_model = BaselineOnly()
baseline_model.fit(trainset)

baseline_predictions = baseline_model.test(testset)
rmse_baseline = accuracy.rmse(baseline_predictions, verbose=False)

baseline_rmse = accuracy.rmse(baseline_predictions)

# Train a collaborative filtering model using SVD
model = SVD(n_factors=100, random_state=42, n_epochs = 50)
model.fit(trainset)

svd_predictions = model.test(testset)

rmse_svd = accuracy.rmse(svd_predictions, verbose=False)


print(f'Baseline RMSE: {rmse_baseline:.4f}')
print(f'SVD      RMSE: {rmse_svd:.4f}')
print(f'Improvement : {(rmse_baseline - rmse_svd) / rmse_baseline * 100:.1f}%')