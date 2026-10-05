## Project Objective
This project builds a personalized content-based movie recommender system for User ID 37. By analyzing historical rating patterns, the recommender identifies key genre preferences and uses TF-IDF vectorization with cosine similarity to suggest five highly relevant movies the user has not yet seen.

## Datasets
* `movies.csv`: Contains movie metadata including `movie_id`, `title`, and pipe-separated `genres`.
* `ratings.csv`: Contains historical user ratings with `user_id`, `movie_id`, and numerical `rating` (0.5 to 5.0).

## Method / Approach
1. **User Profiling:** Filtered `ratings.csv` for `user_id == 37`, calculated average genre scores, and extracted movies rated $\ge 4.0$.
2. **Feature Extraction:** Converted movie genre tags into sparse TF-IDF vectors using scikit-learn's `TfidfVectorizer`.
3. **Similarity Calculation:** Computed pairwise Cosine Similarity across all movie genre vectors.
4. **Filtering & Ranking:** Averaged similarity scores against the user's liked movies, filtered out already-rated titles, and returned the top 5 highest-scoring recommendations.

## How to Run the Notebook
1. Prerequisites: Python 3.8+, `pandas`, `numpy`, `scikit-learn`, `jupyter`.
2. Ensure `movies.csv` and `ratings.csv` are in the same directory as the notebook.
3. Open `DSA4060_Recommender_<StudentID>.ipynb` in Jupyter Notebook / VS Code.
4. Run all cells sequentially (`Run All`).

## Recommendation Results
Generated recommendations match User 37's top-rated genre profiles based on high similarity scores without including any previously rated titles.

## Limitation and Suggested Improvement
* **Limitation:** Pure content-based filtering creates an echo chamber (lack of serendipity) and depends strictly on genre tags without considering overall movie quality or user ratings.
* **Improvement:** Implement a hybrid model combining genre-based similarity with Collaborative Filtering (SVD/Matrix Factorization) to leverage collective user ratings.
