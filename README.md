# Movie Recommender

A movie recommendation system built with Python — content-based filtering **and**
collaborative filtering (matrix factorisation). Trained and evaluated on the MovieLens dataset.

---

## Features

- **Content-based recommendations** — "movies similar to this one", using TF-IDF on genres + cosine similarity
- **Collaborative filtering** — predicts how a user would rate a movie, using SVD matrix factorisation
- **Evaluated properly** — train/test split with RMSE, compared against a baseline

---

## Example

```python
>>> recommend('Toy Story (1995)')
['Jumanji (1995)', 'The Goonies (1985)', 'Aladdin (1992)', 'Hook (1991)', 'Babe (1995)']
```

---

## Dataset

[MovieLens](https://grouplens.org/datasets/movielens/) `ml-latest-small`:

- 100,836 ratings
- 9,742 movies
- 610 users

Download the zip and place the CSVs in a `data/` folder at the project root.

---

## Project structure

```
movie-recommender/
├── data/                  # MovieLens CSVs (not committed - see .gitignore)
├── recommender.py         # Stage 1: content-based recommender
├── collab_filter.py       # Stage 2: collaborative filtering (SVD)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Usage

**Stage 1 — similar movies:**
```bash
python recommender.py
```

**Stage 2 — rating prediction + evaluation:**
```bash
python collab_filter.py
```

---

## How it works

### Stage 1 — Content-based filtering
Each movie's genres are turned into a TF-IDF vector, and cosine similarity is computed
between every pair of movies. Recommendations are the highest-similarity movies to the input.

*This is a similarity-based approach, not a trained model — no parameters are learned.*

### Stage 2 — Collaborative filtering (SVD)
Matrix factorisation learns latent factor vectors for users and movies by minimising the
error between predicted and actual ratings on the training set. This **is** a trained model.

The ratings are split 80/20 into train and test. The model is fit on the training set only,
then evaluated on the held-out test set with RMSE. It is compared against a baseline that
always predicts the mean rating — the baseline is what tells us whether the model actually
learned anything.

---

## Results

| Model | RMSE (test set) |
|-------|-----------------|
| Baseline (mean rating) | 0.8785 |
| SVD (collaborative filtering) | 0.8900 |

*Lower RMSE is better. SVD should beat the baseline — if it doesn't, the model isn't learning.*

---

## Limitations

- Content-based recommendations rely on genres only, which is coarse: all comedies look
  alike, so suggestions can be off in mood or tone.
- The collaborative model predicts ratings; turning those predictions into a ranked
  recommendation list (top-N) is a further step.
- No handling of the cold-start problem: a new user or movie with no ratings gets no
  meaningful prediction.

---

## Roadmap

- [x] Stage 1: content-based recommender
- [x] Stage 2: collaborative filtering (SVD) with RMSE vs baseline
- [ ] Top-N recommendations from predicted ratings
- [ ] Streamlit app + live deployment

---

## Tech

Python  pandas  scikit-learn  scikit-surprise
