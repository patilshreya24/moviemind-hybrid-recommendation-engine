from pathlib import Path
from datetime import datetime
import shutil
import joblib
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "ml" / "data" / "raw"
MODEL_DIR = BASE_DIR / "ml" / "models"

RATINGS_FILE = RAW_DIR / "ratings_small.csv"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

N_COMPONENTS = 50
MIN_MOVIE_RATINGS = 5
N_NEIGHBORS = 150
RANDOM_STATE = 42


# ============================================================
# MODEL FILES
# ============================================================

MODEL_FILES = [
    MODEL_DIR / "user_movie_matrix.pkl",
    MODEL_DIR / "collab_svd.pkl",
    MODEL_DIR / "movie_latent.pkl",
    MODEL_DIR / "collab_knn.pkl",
]


# ============================================================
# BACKUP EXISTING MODELS
# ============================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_dir = MODEL_DIR / f"backup_before_rebuild_{timestamp}"

backup_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("MOVIEMIND COLLABORATIVE MODEL REBUILD")
print("=" * 60)

print(f"Ratings file: {RATINGS_FILE}")
print(f"Model directory: {MODEL_DIR}")
print()


for file_path in MODEL_FILES:
    if file_path.exists():
        destination = backup_dir / file_path.name
        shutil.copy2(file_path, destination)
        print(f"Backed up: {file_path.name}")


print()
print(f"Backup directory: {backup_dir}")
print()


# ============================================================
# LOAD RATINGS
# ============================================================

if not RATINGS_FILE.exists():
    raise FileNotFoundError(
        f"Could not find ratings file:\n{RATINGS_FILE}"
    )


ratings = pd.read_csv(RATINGS_FILE)

required_columns = {"userId", "movieId", "rating"}

missing = required_columns - set(ratings.columns)

if missing:
    raise ValueError(
        f"ratings_small.csv is missing columns: {missing}"
    )


print("Raw ratings:")
print(f"  Rows: {len(ratings):,}")
print(f"  Users: {ratings['userId'].nunique():,}")
print(f"  Movies: {ratings['movieId'].nunique():,}")
print()


# ============================================================
# CLEAN DATA
# ============================================================

ratings = ratings[
    ["userId", "movieId", "rating"]
].copy()

ratings["userId"] = pd.to_numeric(
    ratings["userId"],
    errors="coerce"
)

ratings["movieId"] = pd.to_numeric(
    ratings["movieId"],
    errors="coerce"
)

ratings["rating"] = pd.to_numeric(
    ratings["rating"],
    errors="coerce"
)

ratings = ratings.dropna()

ratings["userId"] = ratings["userId"].astype(int)
ratings["movieId"] = ratings["movieId"].astype(int)

# In case the raw file contains duplicate user/movie rows,
# aggregate them rather than silently overwriting values.
ratings = (
    ratings
    .groupby(["userId", "movieId"], as_index=False)["rating"]
    .mean()
)


# ============================================================
# MOVIE FILTER
# ============================================================

movie_rating_counts = (
    ratings
    .groupby("movieId")
    .size()
    .sort_values(ascending=False)
)

print("Movie rating statistics BEFORE filtering:")
print(movie_rating_counts.describe())
print()

# Keep movies with enough ratings to produce a meaningful
# collaborative representation.
valid_movies = movie_rating_counts[
    movie_rating_counts >= MIN_MOVIE_RATINGS
].index

ratings = ratings[
    ratings["movieId"].isin(valid_movies)
].copy()


# ============================================================
# OPTIONAL USER FILTER
# ============================================================

# Remove extremely sparse users as well.
user_rating_counts = (
    ratings
    .groupby("userId")
    .size()
)

valid_users = user_rating_counts[
    user_rating_counts >= 5
].index

ratings = ratings[
    ratings["userId"].isin(valid_users)
].copy()


# ============================================================
# FINAL DATASET SUMMARY
# ============================================================

movie_ids = np.sort(
    ratings["movieId"].unique()
)

user_ids = np.sort(
    ratings["userId"].unique()
)

print("AFTER FILTERING")
print("-" * 60)
print(f"Users: {len(user_ids):,}")
print(f"Movies: {len(movie_ids):,}")
print(f"Ratings: {len(ratings):,}")
print()


# ============================================================
# CREATE USER × MOVIE MATRIX
# ============================================================

print("Creating user × movie matrix...")

user_index = {
    user_id: index
    for index, user_id in enumerate(user_ids)
}

movie_index = {
    movie_id: index
    for index, movie_id in enumerate(movie_ids)
}


rows = ratings["userId"].map(user_index).to_numpy()
cols = ratings["movieId"].map(movie_index).to_numpy()
values = ratings["rating"].astype(float).to_numpy()


sparse_matrix = csr_matrix(
    (
        values,
        (rows, cols)
    ),
    shape=(
        len(user_ids),
        len(movie_ids)
    ),
    dtype=np.float32
)


# Dense DataFrame is small enough for MovieLens-small.
user_movie_matrix = pd.DataFrame(
    sparse_matrix.toarray(),
    index=user_ids,
    columns=movie_ids
)

user_movie_matrix.index.name = "userId"
user_movie_matrix.columns.name = "movieId"


print(
    "Matrix shape:",
    user_movie_matrix.shape
)

print(
    "Non-zero ratings:",
    int(np.count_nonzero(user_movie_matrix.values))
)

print()


# ============================================================
# IMPORTANT VALIDATION
# ============================================================

print("Checking important movies...")

for movie_id in [2, 7974, 14793, 29568]:
    if movie_id in user_movie_matrix.columns:

        count = int(
            np.count_nonzero(
                user_movie_matrix[movie_id].to_numpy()
            )
        )

        print(
            f"MovieLens ID {movie_id}: "
            f"{count} ratings"
        )

    else:
        print(
            f"MovieLens ID {movie_id}: "
            f"NOT IN FILTERED MATRIX"
        )

print()


# ============================================================
# SAVE USER-MOVIE MATRIX
# ============================================================

matrix_path = MODEL_DIR / "user_movie_matrix.pkl"

joblib.dump(
    user_movie_matrix,
    matrix_path
)

print(
    f"Saved: {matrix_path}"
)


# ============================================================
# TRAIN TRUNCATED SVD
# ============================================================

print()
print("=" * 60)
print("Training Truncated SVD...")
print("=" * 60)

n_components = min(
    N_COMPONENTS,
    min(sparse_matrix.shape) - 1
)

svd = TruncatedSVD(
    n_components=n_components,
    n_iter=10,
    random_state=RANDOM_STATE,
    algorithm="randomized"
)

svd.fit(sparse_matrix)


print(
    f"SVD components: {n_components}"
)

print(
    f"Explained variance ratio: "
    f"{svd.explained_variance_ratio_.sum():.4f}"
)

print()


# ============================================================
# CREATE MOVIE LATENT VECTORS
# ============================================================

# X ≈ U S Vt
#
# Movie embeddings are V × S.
#
# sklearn stores Vt in svd.components_
# and S in svd.singular_values_.

movie_latent = (
    svd.components_.T
    * svd.singular_values_
)


movie_latent = np.asarray(
    movie_latent,
    dtype=np.float32
)


print(
    "Movie latent shape:",
    movie_latent.shape
)


# ============================================================
# SAVE SVD
# ============================================================

svd_path = MODEL_DIR / "collab_svd.pkl"

joblib.dump(
    svd,
    svd_path
)

print(
    f"Saved: {svd_path}"
)


# ============================================================
# SAVE MOVIE LATENT
# ============================================================

latent_path = MODEL_DIR / "movie_latent.pkl"

joblib.dump(
    movie_latent,
    latent_path
)

print(
    f"Saved: {latent_path}"
)


# ============================================================
# TRAIN COSINE KNN
# ============================================================

print()
print("=" * 60)
print("Training collaborative KNN...")
print("=" * 60)

actual_neighbors = min(
    N_NEIGHBORS,
    len(movie_ids)
)

collab_knn = NearestNeighbors(
    n_neighbors=actual_neighbors,
    metric="cosine",
    algorithm="brute",
    n_jobs=-1
)

collab_knn.fit(movie_latent)


print(
    f"KNN neighbours: {actual_neighbors}"
)

print(
    "KNN metric: cosine"
)


# ============================================================
# SAVE KNN
# ============================================================

knn_path = MODEL_DIR / "collab_knn.pkl"

joblib.dump(
    collab_knn,
    knn_path
)

print(
    f"Saved: {knn_path}"
)


# ============================================================
# VALIDATE JUMANJI
# ============================================================

print()
print("=" * 60)
print("VALIDATING JUMANJI")
print("=" * 60)

if 2 in movie_index:

    jumanji_index = movie_index[2]

    jumanji_vector = movie_latent[
        jumanji_index
    ]

    jumanji_norm = np.linalg.norm(
        jumanji_vector
    )

    print(
        f"Jumanji latent index: {jumanji_index}"
    )

    print(
        f"Jumanji latent norm: {jumanji_norm:.10f}"
    )

    jumanji_rating_count = int(
        np.count_nonzero(
            user_movie_matrix[2].to_numpy()
        )
    )

    print(
        f"Jumanji rating count: "
        f"{jumanji_rating_count}"
    )

    distances, indices = collab_knn.kneighbors(
        jumanji_vector.reshape(1, -1),
        n_neighbors=min(11, len(movie_ids))
    )

    print()
    print("Jumanji nearest neighbours:")

    for distance, latent_index in zip(
        distances[0][1:],
        indices[0][1:]
    ):

        movie_id = int(
            movie_ids[latent_index]
        )

        similarity = 1.0 - float(distance)

        print(
            f"  MovieLens {movie_id}: "
            f"{similarity:.4f}"
        )

else:

    print(
        "WARNING: Jumanji (MovieLens ID 2) "
        "was removed by filtering."
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 60)
print("COLLABORATIVE REBUILD COMPLETE")
print("=" * 60)

print(
    f"Final matrix: {user_movie_matrix.shape}"
)

print(
    f"Movie latent: {movie_latent.shape}"
)

print(
    f"Models saved in: {MODEL_DIR}"
)

print(
    f"Old models backed up in: {backup_dir}"
)

print()
print("Next step:")
print(
    "Restart the FastAPI backend and test Jumanji."
)