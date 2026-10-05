from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
import numpy as np
import joblib
import os
import re
import requests

from functools import lru_cache
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# ENVIRONMENT
# ============================================================

BACKEND_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

load_dotenv(
    os.path.join(
        BACKEND_DIR,
        ".env"
    )
)

TMDB_API_KEY = os.getenv(
    "TMDB_API_KEY"
)

TMDB_READ_ACCESS_TOKEN = os.getenv(
    "TMDB_READ_ACCESS_TOKEN"
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="MovieMind Hybrid Recommendation Engine",
    description="AI-powered hybrid movie recommendation system",
    version="3.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "ml",
    "models"
)


# ============================================================
# LOAD CONTENT MODELS
# ============================================================

tfidf = joblib.load(
    os.path.join(
        MODEL_DIR,
        "content_tfidf.pkl"
    )
)

content_model = joblib.load(
    os.path.join(
        MODEL_DIR,
        "content_knn.pkl"
    )
)

content_data = pd.read_pickle(
    os.path.join(
        MODEL_DIR,
        "content_data.pkl"
    )
)


# ============================================================
# LOAD COLLABORATIVE MODELS
# ============================================================

collab_knn = joblib.load(
    os.path.join(
        MODEL_DIR,
        "collab_knn.pkl"
    )
)

collab_svd = joblib.load(
    os.path.join(
        MODEL_DIR,
        "collab_svd.pkl"
    )
)

movie_latent = joblib.load(
    os.path.join(
        MODEL_DIR,
        "movie_latent.pkl"
    )
)

# IMPORTANT:
# user_movie_matrix.pkl was saved using joblib.
user_movie_matrix = joblib.load(
    os.path.join(
        MODEL_DIR,
        "user_movie_matrix.pkl"
    )
)


# ============================================================
# CLEAN TITLES
# ============================================================

content_data["title_clean"] = (
    content_data["title"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
)


def normalize_title(title):
    """
    Creates a consistent title representation.

    Used for duplicate detection and input exclusion.
    """

    if title is None:
        return ""

    return (
        str(title)
        .strip()
        .lower()
    )


# ============================================================
# MOVIE ID NORMALIZATION
# ============================================================

def normalize_lookup_title(title):
    """
    Normalize titles for matching content/TMDB titles
    to MovieLens titles.
    """

    if title is None:
        return ""

    text = str(title).strip().lower()

    # Remove year at the end, e.g. Titanic (1997)
    text = re.sub(
        r"\s*\(\d{4}\)\s*$",
        "",
        text
    )

    # Remove punctuation
    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text
    )

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


def normalize_movie_id(movie_id):

    if movie_id is None:
        return None

    try:
        if pd.isna(movie_id):
            return None
    except Exception:
        pass

    try:
        return int(movie_id)
    except Exception:
        return str(movie_id)


# ============================================================
# COLLABORATIVE MOVIE MAPPING
# ============================================================

collab_movie_ids = [
    normalize_movie_id(movie_id)
    for movie_id in user_movie_matrix.columns
]

movie_id_to_latent_index = {
    movie_id: index
    for index, movie_id in enumerate(
        collab_movie_ids
    )
}


# ============================================================
# CONTENT + COLLABORATIVE MOVIE MAPPING
# ============================================================

title_to_content_index = {}

movie_id_to_content_index = {}

title_to_movie_id = {}

# Map normalized titles to MovieLens IDs
# that actually exist in the collaborative model.
collab_title_to_movie_id = {}

# Robust TMDB <-> MovieLens bridge.
collab_tmdb_to_movie_id = {}
collab_movie_id_to_tmdb = {}


# ------------------------------------------------------------
# Build title lookup from content_data
# ------------------------------------------------------------

for index, row in content_data.iterrows():

    title = str(
        row["title"]
    )

    clean_title = normalize_lookup_title(
        title
    )

    movie_id = normalize_movie_id(
        row.get("id")
    )

    # Keep first content record for a title
    if clean_title not in title_to_content_index:

        title_to_content_index[
            clean_title
        ] = index

    # Keep content movie ID -> content index
    if movie_id is not None:

        if movie_id not in movie_id_to_content_index:

            movie_id_to_content_index[
                movie_id
            ] = index

        if clean_title not in title_to_movie_id:

            title_to_movie_id[
                clean_title
            ] = movie_id


# ------------------------------------------------------------
# IMPORTANT:
#
# Build MovieLens title -> movieId mapping from the files
# actually present in ml/data/raw.
#
# links_small.csv connects MovieLens movieId to TMDB tmdbId.
# movies_metadata.csv provides the TMDB movie title.
# ------------------------------------------------------------

raw_data_dir = os.path.join(
    BASE_DIR,
    "ml",
    "data",
    "raw"
)

links_file = os.path.join(
    raw_data_dir,
    "links_small.csv"
)

metadata_file = os.path.join(
    raw_data_dir,
    "movies_metadata.csv"
)


if (
    os.path.exists(links_file)
    and os.path.exists(metadata_file)
):

    try:

        links_df = pd.read_csv(
            links_file
        )

        metadata_df = pd.read_csv(
            metadata_file,
            low_memory=False
        )

        metadata_df["tmdb_id_num"] = pd.to_numeric(
            metadata_df["id"],
            errors="coerce"
        )

        links_df["tmdb_id_num"] = pd.to_numeric(
            links_df["tmdbId"],
            errors="coerce"
        )

        metadata_lookup = (
            metadata_df[
                [
                    "tmdb_id_num",
                    "title"
                ]
            ]
            .dropna(
                subset=[
                    "tmdb_id_num",
                    "title"
                ]
            )
        )

        links_lookup = (
            links_df[
                [
                    "movieId",
                    "tmdb_id_num"
                ]
            ]
            .dropna(
                subset=[
                    "movieId",
                    "tmdb_id_num"
                ]
            )
        )

        movie_mapping_df = links_lookup.merge(
            metadata_lookup,
            on="tmdb_id_num",
            how="inner"
        )

        for _, row in movie_mapping_df.iterrows():

            movie_id = normalize_movie_id(
                row["movieId"]
            )

            tmdb_id = normalize_movie_id(
                row["tmdb_id_num"]
            )

            title = normalize_lookup_title(
                row["title"]
            )

            if (
                movie_id is not None
                and tmdb_id is not None
                and movie_id in movie_id_to_latent_index
            ):

                collab_tmdb_to_movie_id[
                    tmdb_id
                ] = movie_id

                collab_movie_id_to_tmdb[
                    movie_id
                ] = tmdb_id

                if (
                    title
                    and title not in collab_title_to_movie_id
                ):

                    collab_title_to_movie_id[
                        title
                    ] = movie_id

        print(
            "Collaborative mapping files:",
            links_file,
            "and",
            metadata_file
        )

    except Exception as error:

        print(
            "Could not build collaborative title mapping:",
            error
        )

else:

    print(
        "Collaborative mapping files not found:",
        links_file,
        metadata_file
    )


print(
    "Collaborative title mappings:",
    len(collab_title_to_movie_id)
)

print(
    "TMDB -> MovieLens mappings:",
    len(collab_tmdb_to_movie_id)
)


# Reverse lookup:
# MovieLens movieId -> normalized title.
collab_movie_id_to_title = {
    movie_id: title
    for title, movie_id
    in collab_title_to_movie_id.items()
}


# ============================================================
# STARTUP INFORMATION
# ============================================================

print("================================================")
print("MovieMind Hybrid Engine v3")
print("================================================")

print(
    "Content movies:",
    len(content_data)
)

print(
    "Collaborative movies:",
    len(collab_movie_ids)
)

print(
    "Movie latent shape:",
    movie_latent.shape
)

print(
    "Collaborative KNN features:",
    collab_knn.n_features_in_
)

print(
    "Collaborative KNN metric:",
    collab_knn.metric
)

print(
    "Models loaded successfully!"
)

print("================================================")


# ============================================================
# LOCAL POSTER DETECTION
# ============================================================

poster_column = None

possible_columns = [
    "poster_path",
    "poster_url",
    "poster",
    "image_url",
    "image",
    "posterPath"
]


for column in possible_columns:

    if column in content_data.columns:

        poster_column = column

        break


print(
    "Local poster column:",
    poster_column
)


# ============================================================
# LOCAL POSTER
# ============================================================

def get_local_poster(row):

    if poster_column is None:
        return None

    value = row.get(
        poster_column
    )

    if pd.isna(value):
        return None

    value = str(
        value
    ).strip()

    if (
        not value
        or value.lower() == "nan"
    ):
        return None

    if (
        value.startswith("http://")
        or value.startswith("https://")
    ):
        return value

    if value.startswith("/"):

        return (
            "https://image.tmdb.org/t/p/w500"
            + value
        )

    return None


# ============================================================
# TMDB SESSION
# ============================================================

tmdb_session = requests.Session()

tmdb_session.headers.update(
    {
        "accept": "application/json"
    }
)


# ============================================================
# EXTRACT YEAR
# ============================================================

def extract_title_and_year(title):

    title = str(
        title
    ).strip()

    match = re.search(
        r"\s*\((\d{4})\)\s*$",
        title
    )

    if match:

        year = int(
            match.group(1)
        )

        clean_title = re.sub(
            r"\s*\(\d{4}\)\s*$",
            "",
            title
        ).strip()

        return clean_title, year

    return title, None


# ============================================================
# TMDB LOOKUP
# ============================================================

@lru_cache(maxsize=10000)
def get_tmdb_movie(
    title,
    preferred_tmdb_id=None
):

    empty_result = {
        "tmdb_id": None,
        "poster": None,
        "overview": None,
        "release_date": None,
        "vote_average": None
    }

    if (
        not TMDB_API_KEY
        and not TMDB_READ_ACCESS_TOKEN
    ):
        return empty_result

    clean_title, year = extract_title_and_year(
        title
    )

    headers = {
        "accept": "application/json"
    }

    params = {
        "language": "en-US"
    }

    if TMDB_READ_ACCESS_TOKEN:

        headers["Authorization"] = (
            f"Bearer {TMDB_READ_ACCESS_TOKEN}"
        )

    else:

        params["api_key"] = TMDB_API_KEY


    try:

        # --------------------------------------------------------
        # METHOD 1:
        # If we already know the TMDB ID, use it directly.
        #
        # This is the most reliable method because it avoids
        # ambiguous title searches.
        # --------------------------------------------------------

        if preferred_tmdb_id is not None:

            try:

                tmdb_id = int(
                    preferred_tmdb_id
                )

                url = (
                    f"https://api.themoviedb.org/3/movie/"
                    f"{tmdb_id}"
                )

                response = tmdb_session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=5
                )

                if response.ok:

                    result = response.json()

                    poster_path = result.get(
                        "poster_path"
                    )

                    poster_url = None

                    if poster_path:

                        poster_url = (
                            "https://image.tmdb.org/t/p/w500"
                            + poster_path
                        )

                    return {
                        "tmdb_id": result.get("id"),
                        "poster": poster_url,
                        "overview": result.get(
                            "overview"
                        ),
                        "release_date": result.get(
                            "release_date"
                        ),
                        "vote_average": result.get(
                            "vote_average"
                        )
                    }

            except Exception as error:

                print(
                    f"Direct TMDB ID lookup failed for "
                    f"{title}: {error}"
                )


        # --------------------------------------------------------
        # METHOD 2:
        # Fall back to title search.
        # --------------------------------------------------------

        url = (
            "https://api.themoviedb.org/3/search/movie"
        )

        search_params = params.copy()

        search_params["query"] = clean_title
        search_params["include_adult"] = "false"

        if year:

            search_params["year"] = year


        response = tmdb_session.get(
            url,
            params=search_params,
            headers=headers,
            timeout=5
        )

        response.raise_for_status()

        results = response.json().get(
            "results",
            []
        )

        if not results:

            return empty_result


        # --------------------------------------------------------
        # Choose the best title/year match.
        # --------------------------------------------------------

        normalized_title = normalize_lookup_title(
            clean_title
        )

        best_result = None

        for result in results:

            result_title = (
                result.get("title")
                or result.get("original_title")
                or ""
            )

            normalized_result_title = (
                normalize_lookup_title(
                    result_title
                )
            )

            if (
                normalized_result_title
                != normalized_title
            ):
                continue


            # If we know the year,
            # prefer the matching year.

            if year:

                release_date = (
                    result.get(
                        "release_date"
                    )
                    or ""
                )

                result_year = None

                if len(release_date) >= 4:

                    try:

                        result_year = int(
                            release_date[:4]
                        )

                    except Exception:

                        result_year = None


                if result_year == year:

                    best_result = result
                    break

            else:

                # Exact title match when no year is available.
                best_result = result
                break


        # If no exact match was found,
        # use TMDB's first result.

        if best_result is None:

            best_result = results[0]


        poster_path = best_result.get(
            "poster_path"
        )

        poster_url = None

        if poster_path:

            poster_url = (
                "https://image.tmdb.org/t/p/w500"
                + poster_path
            )


        return {
            "tmdb_id": best_result.get(
                "id"
            ),

            "poster": poster_url,

            "overview": best_result.get(
                "overview"
            ),

            "release_date": best_result.get(
                "release_date"
            ),

            "vote_average": best_result.get(
                "vote_average"
            )
        }


    except Exception as error:

        print(
            f"TMDB lookup failed for {title}: {error}"
        )

        return empty_result


# ============================================================
# MOVIE METADATA
# ============================================================

def get_movie_metadata(row):

    title = str(
        row["title"]
    )

    local_poster = get_local_poster(
        row
    )

    # The ID stored in content_data is the TMDB/content-side ID.
    preferred_tmdb_id = normalize_movie_id(
        row.get("id")
    )

    tmdb = get_tmdb_movie(
        title,
        preferred_tmdb_id=preferred_tmdb_id
    )

    return {

        "poster": (
            local_poster
            if local_poster
            else tmdb["poster"]
        ),

        "tmdb_id": (
            tmdb["tmdb_id"]
        ),

        "overview": (
            tmdb["overview"]
        ),

        "release_date": (
            tmdb["release_date"]
        ),

        "vote_average": (
            tmdb["vote_average"]
        )
    }


# ============================================================
# CONTENT CANDIDATES
# ============================================================

def get_content_candidates(
    movie_title,
    candidate_count=150
):

    clean_title = normalize_lookup_title(
        movie_title
    )

    if (
        clean_title
        not in title_to_content_index
    ):

        return None, []


    input_index = (
        title_to_content_index[
            clean_title
        ]
    )


    movie_features = content_data.loc[
        input_index,
        "content_features"
    ]


    movie_vector = tfidf.transform(
        [movie_features]
    )


    distances, indices = (
        content_model.kneighbors(
            movie_vector,
            n_neighbors=min(
                candidate_count + 1,
                len(content_data)
            )
        )
    )


    candidates = []


    for distance, index in zip(
        distances[0],
        indices[0]
    ):

        row = content_data.iloc[
            index
        ]

        title = str(
            row["title"]
        )


        # Skip the input movie
        # and duplicate title records.

        if (
            normalize_lookup_title(title)
            == normalize_lookup_title(
                movie_title
            )
        ):
            continue


        movie_id = normalize_movie_id(
            row.get("id")
        )


        content_similarity = max(
            0.0,
            1.0 - float(distance)
        )


        candidates.append(
            {
                "title": title,
                "movie_id": movie_id,
                "content_index": index,
                "content_similarity":
                    content_similarity
            }
        )


    return input_index, candidates


# ============================================================
# COLLABORATIVE ENGINE
# ============================================================

def get_collaborative_scores(
    input_movie_id,
    candidate_count=150
):

    input_movie_id = normalize_movie_id(
        input_movie_id
    )


    if (
        input_movie_id is None
        or input_movie_id
        not in movie_id_to_latent_index
    ):

        return {}


    latent_index = (
        movie_id_to_latent_index[
            input_movie_id
        ]
    )


    movie_vector = movie_latent[
        latent_index
    ].reshape(
        1,
        -1
    )


    distances, indices = (
        collab_knn.kneighbors(
            movie_vector,
            n_neighbors=min(
                candidate_count + 1,
                len(movie_latent)
            )
        )
    )


    scores = {}


    for distance, index in zip(
        distances[0],
        indices[0]
    ):

        candidate_movie_id = normalize_movie_id(
            collab_movie_ids[index]
        )


        # Never include the input movie.

        if (
            candidate_movie_id
            == input_movie_id
        ):
            continue


        collaborative_similarity = max(
            0.0,
            1.0 - float(distance)
        )


        scores[
            candidate_movie_id
        ] = collaborative_similarity


    return scores


# ============================================================
# HYBRID RECOMMENDER
# ============================================================

def recommend_movies(
    movie_title,
    n=10,
    content_weight=0.70,
    collaborative_weight=0.30
):

    clean_input_title = normalize_lookup_title(
        movie_title
    )

    if clean_input_title not in title_to_content_index:
        return []

    input_index = title_to_content_index[
        clean_input_title
    ]

    input_row = content_data.loc[
        input_index
    ]

    # ========================================================
    # RESOLVE MOVIELENS ID
    # ========================================================

    input_content_id = normalize_movie_id(
        input_row.get("id")
    )

    input_movie_id = collab_tmdb_to_movie_id.get(
        input_content_id
    )

    if input_movie_id is None:
        input_movie_id = collab_title_to_movie_id.get(
            clean_input_title
        )

    if (
        input_movie_id is None
        and input_content_id in movie_id_to_latent_index
    ):
        input_movie_id = input_content_id

    # ========================================================
    # CONTENT FALLBACK
    # ========================================================

    if input_movie_id is None:

        _, content_candidates = get_content_candidates(
            movie_title,
            candidate_count=100
        )

        results = []

        for candidate in content_candidates[:n]:

            content_index = candidate.get(
                "content_index"
            )

            if content_index is None:
                continue

            row = content_data.iloc[
                content_index
            ]

            metadata = get_movie_metadata(
                row
            )

            score = candidate[
                "content_similarity"
            ]

            results.append(
                {
                    "rank": len(results) + 1,
                    "title": candidate["title"],
                    "similarity": round(
                        score * 100,
                        1
                    ),
                    "content_score": round(
                        score * 100,
                        1
                    ),
                    "collaborative_score": 0.0,
                    "poster": metadata["poster"],
                    "tmdb_id": metadata["tmdb_id"],
                    "overview": metadata["overview"],
                    "release_date": metadata["release_date"],
                    "vote_average": metadata["vote_average"]
                }
            )

        return results

    # ========================================================
    # GET COLLABORATIVE CANDIDATES
    # ========================================================

    collaborative_scores = get_collaborative_scores(
        input_movie_id,
        candidate_count=100
    )

    if not collaborative_scores:
        return []

    # ========================================================
    # INPUT CONTENT VECTOR
    # ========================================================

    input_content_vector = tfidf.transform(
        [input_row["content_features"]]
    )

    candidates = []

    # ========================================================
    # COLLABORATIVE CANDIDATES
    #
    # Collaborative filtering creates the candidate pool.
    # Content similarity then contributes to the final score.
    # ========================================================

    for movie_id, collaborative_score in (
        collaborative_scores.items()
    ):

        movie_id = normalize_movie_id(
            movie_id
        )

        tmdb_id = collab_movie_id_to_tmdb.get(
            movie_id
        )

        if tmdb_id is None:
            continue

        content_index = (
            movie_id_to_content_index.get(
                tmdb_id
            )
        )

        if content_index is None:

            collab_title = (
                collab_movie_id_to_title.get(
                    movie_id,
                    ""
                )
            )

            content_index = (
                title_to_content_index.get(
                    normalize_lookup_title(
                        collab_title
                    )
                )
            )

        if content_index is None:
            continue

        row = content_data.iloc[
            content_index
        ]

        title = str(
            row["title"]
        )

        title_key = normalize_lookup_title(
            title
        )

        if (
            not title_key
            or title_key == clean_input_title
        ):
            continue

        # ----------------------------------------------------
        # REAL CONTENT SIMILARITY
        # ----------------------------------------------------

        candidate_content_vector = tfidf.transform(
            [row["content_features"]]
        )

        content_similarity = float(
            cosine_similarity(
                input_content_vector,
                candidate_content_vector
            )[0][0]
        )

        content_similarity = max(
            0.0,
            min(
                1.0,
                content_similarity
            )
        )

        collaborative_score = max(
            0.0,
            min(
                1.0,
                float(collaborative_score)
            )
        )

        # ----------------------------------------------------
        # HYBRID SCORE
        #
        # IMPORTANT:
        # We keep the original similarity scales.
        # No min-max normalization.
        #
        # 70% content
        # 30% collaborative
        # ----------------------------------------------------

        hybrid_score = (
            content_weight
            * content_similarity
            +
            collaborative_weight
            * collaborative_score
        )

        candidates.append(
            {
                "title": title,
                "movie_id": tmdb_id,
                "content_index": content_index,
                "content_similarity": content_similarity,
                "collaborative_similarity":
                    collaborative_score,
                "hybrid_score": hybrid_score
            }
        )

    # ========================================================
    # SORT
    # ========================================================

    candidates.sort(
        key=lambda x: x["hybrid_score"],
        reverse=True
    )

    # ========================================================
    # REMOVE DUPLICATE TITLES
    # ========================================================

    unique_candidates = []
    seen_titles = set()

    for candidate in candidates:

        title_key = normalize_lookup_title(
            candidate["title"]
        )

        if title_key in seen_titles:
            continue

        seen_titles.add(
            title_key
        )

        unique_candidates.append(
            candidate
        )

        if len(unique_candidates) >= n:
            break

    # ========================================================
    # BUILD RESPONSE
    # ========================================================

    recommendations = []

    for rank, item in enumerate(
        unique_candidates,
        start=1
    ):

        content_index = item[
            "content_index"
        ]

        row = content_data.iloc[
            content_index
        ]

        metadata = get_movie_metadata(
            row
        )

        recommendations.append(
            {
                "rank": rank,

                "title": item["title"],

                "similarity": round(
                    item["hybrid_score"] * 100,
                    1
                ),

                "content_score": round(
                    item["content_similarity"] * 100,
                    1
                ),

                "collaborative_score": round(
                    item["collaborative_similarity"] * 100,
                    1
                ),

                "poster": metadata["poster"],

                "tmdb_id": metadata["tmdb_id"],

                "overview": metadata["overview"],

                "release_date":
                    metadata["release_date"],

                "vote_average":
                    metadata["vote_average"]
            }
        )

    return recommendations


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {

        "message":
            "MovieMind Hybrid API is running",

        "status":
            "success",

        "engine":
            "Hybrid Content + Collaborative",

        "content_weight":
            0.70,

        "collaborative_weight":
            0.30
    }


# ============================================================
# MOVIES
# ============================================================

@app.get("/movies")
def get_movies():

    movies = (
        content_data["title"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    return {
        "movies":
            movies
    }


# ============================================================
# RECOMMEND
# ============================================================

@app.get("/recommend")
def get_recommendations(
    movie: str,
    n: int = 10
):

    n = max(
        1,
        min(n, 50)
    )


    results = recommend_movies(
        movie,
        n
    )


    if not results:

        raise HTTPException(
            status_code=404,
            detail="Movie not found"
        )


    clean_title = normalize_lookup_title(
        movie
    )


    input_index = (
        title_to_content_index.get(
            clean_title
        )
    )


    input_movie_id = None


    if input_index is not None:

        input_row = content_data.loc[
            input_index
        ]


        input_movie_id = (
            collab_tmdb_to_movie_id.get(
                normalize_movie_id(
                    input_row.get("id")
                )
            )
        )


        if input_movie_id is None:

            input_movie_id = (
                collab_title_to_movie_id.get(
                    clean_title
                )
            )


    is_hybrid = (
        input_movie_id is not None
        and input_movie_id
        in movie_id_to_latent_index
    )


    return {

        "input_movie":
            movie,

        "recommendation_type":
            (
                "hybrid"
                if is_hybrid
                else "content_fallback"
            ),

        "weights": {

            "content":
                0.70
                if is_hybrid
                else 1.0,

            "collaborative":
                0.30
                if is_hybrid
                else 0.0,
        },

        "recommendations":
            results,
    }