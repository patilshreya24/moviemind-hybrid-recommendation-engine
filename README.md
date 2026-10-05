```markdown

\# 🎬 MovieMind — Hybrid Movie Recommendation Engine



> \*\*AI-powered movie recommendation system combining content-based filtering and collaborative filtering to discover movies tailored to your taste.\*\*



MovieMind is a full-stack AI/ML movie recommendation platform that recommends movies based on the movie a user already likes.



The system combines:



\- 🎯 \*\*Content-Based Filtering\*\* using TF-IDF and cosine similarity

\- 👥 \*\*Collaborative Filtering\*\* using user-rating patterns

\- 🔗 \*\*Hybrid Recommendation\*\* combining both approaches

\- 🖼️ \*\*TMDB API\*\* for movie posters and metadata

\- ⚡ \*\*FastAPI\*\* backend

\- ⚛️ \*\*React + Vite\*\* frontend

\- 🎨 Modern dark/glassmorphism interface



\---



\## ✨ Features



\### 🎯 Hybrid Recommendations



MovieMind combines two recommendation approaches:



\*\*70% Content Similarity + 30% Collaborative Similarity\*\*



This allows the system to consider both:



\- What the movie is about

\- How users have rated similar movies



\### 🧠 Content-Based Filtering



Movie metadata such as:



\- Genres

\- Keywords

\- Cast

\- Crew

\- Overview



is transformed into a TF-IDF representation.



Movies are then compared using cosine similarity to identify movies with similar content.



\### 👥 Collaborative Filtering



Collaborative filtering learns from user-movie rating patterns.



The system uses:



\- User-movie rating matrix

\- Truncated SVD

\- Latent movie representations

\- K-Nearest Neighbors



This allows MovieMind to discover relationships between movies based on how users rate them.



\### 🔀 Hybrid Ranking



The final recommendation score combines both models:



```text

Hybrid Score =

0.70 × Content Similarity

\+

0.30 × Collaborative Similarity

```



The recommendations are then ranked using the combined score.



\### 🖼️ Movie Metadata



Movie information is enriched using the TMDB API, including:



\- Movie poster

\- Overview

\- Release date

\- Rating

\- TMDB ID



\### 🔎 Movie Search



Users can search for movies using the interactive search interface.



MovieMind provides:



\- Movie suggestions

\- Enter-to-search support

\- Discover button

\- Loading states

\- Error handling



\### 🎬 Movie Details



Clicking a recommendation opens a movie details panel containing:



\- Movie poster

\- Release year

\- TMDB rating

\- AI match percentage

\- Movie summary

\- Content similarity score

\- Collaborative similarity score

\- Hybrid match score



\---



\# 🏗️ System Architecture



```text

&#x20;                        ┌──────────────────────┐

&#x20;                        │       User           │

&#x20;                        │  Selects a Movie     │

&#x20;                        └──────────┬───────────┘

&#x20;                                   │

&#x20;                                   ▼

&#x20;                        ┌──────────────────────┐

&#x20;                        │   React Frontend     │

&#x20;                        │     + Vite           │

&#x20;                        └──────────┬───────────┘

&#x20;                                   │

&#x20;                                   ▼

&#x20;                        ┌──────────────────────┐

&#x20;                        │     FastAPI API      │

&#x20;                        └──────────┬───────────┘

&#x20;                                   │

&#x20;                 ┌─────────────────┴─────────────────┐

&#x20;                 │                                   │

&#x20;                 ▼                                   ▼

&#x20;       ┌──────────────────┐                ┌──────────────────┐

&#x20;       │ Content-Based    │                │ Collaborative    │

&#x20;       │ Recommendation   │                │ Filtering        │

&#x20;       │                  │                │                  │

&#x20;       │ TF-IDF           │                │ Rating Matrix    │

&#x20;       │ Cosine Similarity│                │ Truncated SVD    │

&#x20;       │ KNN              │                │ KNN              │

&#x20;       └────────┬─────────┘                └────────┬─────────┘

&#x20;                │                                   │

&#x20;                └────────────────┬──────────────────┘

&#x20;                                 │

&#x20;                                 ▼

&#x20;                      ┌──────────────────────┐

&#x20;                      │   Hybrid Ranking     │

&#x20;                      │                      │

&#x20;                      │ 70% Content          │

&#x20;                      │ 30% Collaborative    │

&#x20;                      └──────────┬───────────┘

&#x20;                                 │

&#x20;                                 ▼

&#x20;                      ┌──────────────────────┐

&#x20;                      │     TMDB API         │

&#x20;                      │ Posters + Metadata   │

&#x20;                      └──────────┬───────────┘

&#x20;                                 │

&#x20;                                 ▼

&#x20;                      ┌──────────────────────┐

&#x20;                      │ Recommendation Cards│

&#x20;                      │ + Movie Details      │

&#x20;                      └──────────────────────┘

```



\---



\# 🧠 Machine Learning Pipeline



\## 1. Data Preparation



Movie metadata and user ratings are processed before training the recommendation models.



The collaborative filtering data is filtered to remove extremely sparse movies and users.



The final collaborative dataset contains:



\- \*\*671 users\*\*

\- \*\*3,496 movies\*\*

\- \*\*90,072 ratings\*\*



\---



\## 2. Content Representation



Movie information is transformed into a combined textual representation.



TF-IDF converts the movie information into numerical feature vectors.



These vectors allow the system to compare movies based on their content.



\---



\## 3. Content Similarity



Movie similarity is calculated using cosine similarity.



```text

Cosine Similarity =

(A · B) / (||A|| × ||B||)

```



A higher similarity indicates that two movies have more similar content representations.



\---



\## 4. Collaborative Filtering



A user-movie rating matrix is created from MovieLens ratings.



The matrix is reduced using Truncated SVD.



```text

User-Movie Matrix

&#x20;       ↓

&#x20;  Truncated SVD

&#x20;       ↓

Latent Movie Features

&#x20;       ↓

&#x20;      KNN

&#x20;       ↓

Similar Movies

```



The collaborative model captures relationships between movies based on user rating behavior.



\---



\## 5. Hybrid Recommendation



The content and collaborative scores are combined.



```text

Final Score =

0.70(Content Score)

\+

0.30(Collaborative Score)

```



The movies with the highest final scores are returned as recommendations.



\---



\# 🛠️ Technology Stack



\## Frontend



\- React

\- Vite

\- JavaScript

\- CSS

\- Responsive UI

\- Glassmorphism design



\## Backend



\- Python

\- FastAPI

\- Uvicorn

\- Pandas

\- NumPy

\- Scikit-learn

\- Joblib



\## Machine Learning



\- TF-IDF

\- Cosine Similarity

\- K-Nearest Neighbors

\- Truncated SVD

\- Content-Based Filtering

\- Collaborative Filtering

\- Hybrid Recommendation



\## External API



\- TMDB API



\## Development Tools



\- VS Code

\- Git

\- GitHub

\- Jupyter Notebook



\---



\# 📁 Project Structure



```text

moviemind-hybrid-recommendation-engine/

│

├── backend/

│   ├── main.py

│   └── rebuild\_collaborative.py

│

├── frontend/

│   ├── public/

│   ├── src/

│   │   ├── assets/

│   │   ├── App.jsx

│   │   ├── App.css

│   │   ├── index.css

│   │   └── main.jsx

│   │

│   ├── package.json

│   ├── package-lock.json

│   └── vite.config.js

│

├── ml/

│   └── notebooks/

│       └── 01\_data\_exploration.ipynb

│

├── requirements.txt

├── .gitignore

└── README.md

```



\---



\# 🚀 Getting Started



\## 1. Clone the Repository



```bash

git clone https://github.com/patilshreya24/moviemind-hybrid-recommendation-engine.git

cd moviemind-hybrid-recommendation-engine

```



\# 🐍 Backend Setup



\## 2. Create a Virtual Environment



```bash

python -m venv .venv

```



Activate it on Windows:



```powershell

.venv\\Scripts\\Activate.ps1

```



\## 3. Install Python Dependencies



```bash

pip install -r requirements.txt

```



\## 4. Configure TMDB API



Create:



```text

backend/.env

```



Add your TMDB credentials:



```env

TMDB\_API\_KEY=your\_tmdb\_api\_key

TMDB\_READ\_ACCESS\_TOKEN=your\_tmdb\_read\_access\_token

```



Do \*\*not\*\* commit your `.env` file to GitHub.



\---



\# ▶️ Run the Backend



Open a terminal inside the `backend` directory:



```powershell

cd backend

python -m uvicorn main:app --reload

```



Backend:



```text

http://127.0.0.1:8000

```



FastAPI documentation:



```text

http://127.0.0.1:8000/docs

```



\---



\# ⚛️ Frontend Setup



Open another terminal:



```powershell

cd frontend

npm install

npm run dev

```



Frontend:



```text

http://localhost:5173

```



\---



\# 🔌 API Endpoints



\## Health Check



```http

GET /

```



Returns the current status of the MovieMind recommendation engine.



\## Get Movies



```http

GET /movies

```



Returns the available movie titles used by the recommendation system.



\## Get Recommendations



```http

GET /recommend?movie=Jumanji\&n=10

```



Example response:



```json

{

&#x20; "input\_movie": "Jumanji",

&#x20; "recommendation\_type": "hybrid",

&#x20; "weights": {

&#x20;   "content": 0.7,

&#x20;   "collaborative": 0.3

&#x20; },

&#x20; "recommendations": \[

&#x20;   {

&#x20;     "rank": 1,

&#x20;     "title": "Aladdin",

&#x20;     "similarity": 27.9,

&#x20;     "content\_score": 5.0,

&#x20;     "collaborative\_score": 81.4,

&#x20;     "poster": "...",

&#x20;     "tmdb\_id": 812,

&#x20;     "overview": "...",

&#x20;     "release\_date": "1992-11-25",

&#x20;     "vote\_average": 7.657

&#x20;   }

&#x20; ]

}

```



\---



\# 🎯 Example Recommendation



For an input such as:



```text

Jumanji

```



MovieMind analyzes the movie and combines content and collaborative signals to generate recommendations.



Example:



```text

Because you liked Jumanji



1\. Aladdin

2\. Mrs. Doubtfire

3\. Beauty and the Beast

...

```



Each recommendation displays its AI match score and movie poster.



\---



\# 📊 Recommendation Signals



\### Content Score



Measures how similar the movie's content representation is to the input movie.



\### Collaborative Score



Measures how strongly the movie is related to the input movie based on user-rating patterns.



\### Hybrid Score



Combines both signals:



```text

70% Content

\+

30% Collaborative

```



This provides a more balanced recommendation than relying on a single technique.



\---



\# 🧪 Machine Learning Models



The project uses the following trained components:



```text

content\_tfidf.pkl

content\_knn.pkl

content\_data.pkl



collab\_svd.pkl

collab\_knn.pkl

movie\_latent.pkl

user\_movie\_matrix.pkl

```



\### Content Model



```text

TF-IDF

&#x20;  ↓

Content Feature Vectors

&#x20;  ↓

KNN

&#x20;  ↓

Similar Movies

```



\### Collaborative Model



```text

User-Movie Rating Matrix

&#x20;  ↓

Truncated SVD

&#x20;  ↓

Movie Latent Representations

&#x20;  ↓

KNN

&#x20;  ↓

Collaborative Similarity

```



\### Hybrid Model



```text

Content Similarity ──────┐

&#x20;                        ├──→ Hybrid Score

Collaborative Similarity ┘

```



\---



\# 🔄 Movie ID Mapping



The project uses both MovieLens and TMDB movie identifiers.



To connect the two systems, MovieMind uses:



```text

links\_small.csv

```



to map:



```text

MovieLens movieId

&#x20;       ↕

TMDB movieId

```



This allows the collaborative filtering model to work together with the content and TMDB metadata systems.



\---



\# 🎨 User Interface



The frontend was designed as a modern AI-powered movie discovery experience.



Key interface elements include:



\- Dark cinematic theme

\- Purple accent color

\- Glassmorphism panels

\- Movie recommendation cards

\- Ranking indicators

\- AI match percentages

\- Search suggestions

\- Loading states

\- Error states

\- Movie details modal

\- Recommendation signal breakdown



\---



\# 🎬 Movie Details



Selecting a movie recommendation opens a detailed discovery panel containing:



\- Movie poster

\- Movie title

\- Release year

\- TMDB rating

\- AI match

\- Movie summary

\- Content similarity

\- Collaborative similarity

\- Hybrid match score



This gives users more context about why a movie was recommended.



\---



\# 🔐 Security



Sensitive credentials are excluded from version control.



The following files should never be committed:



```text

.env

\_\_pycache\_\_/

\*.pyc

```



Large datasets and generated model files are also excluded from the Git repository where appropriate.



\---



\# 📚 Dataset



The project was developed using movie metadata and rating information from MovieLens and TMDB-related datasets.



The datasets contain information such as:



\- Movie titles

\- Genres

\- Keywords

\- Cast

\- Crew

\- Movie descriptions

\- User ratings

\- Movie identifiers



The datasets are used for machine learning experimentation and recommendation model development.



\---



\# 📈 Future Improvements



Potential future improvements include:



\- Personalized recommendations using user profiles

\- Rating-based user preference learning

\- Genre preference controls

\- Mood-based recommendations

\- Multi-movie preference input

\- Recommendation explanations

\- Advanced ranking models

\- Recommendation history

\- User accounts

\- Cloud deployment

\- Model serving optimization

\- Larger-scale collaborative filtering



\---



\# 👩‍💻 Author



\*\*Shreya Patil\*\*



AI \& ML



GitHub: https://github.com/patilshreya24



\---



\# 📜 License



This project is intended for educational, academic, and portfolio purposes.



\---



\# 🎥 TMDB Attribution



This product uses the TMDB API but is not endorsed or certified by TMDB.



Movie posters and movie metadata displayed through the application are provided by TMDB.

```

