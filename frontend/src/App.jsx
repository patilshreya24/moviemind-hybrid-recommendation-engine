import { useEffect, useMemo, useState } from "react";
import "./App.css";

const API = "http://127.0.0.1:8000";

function App() {
  const [movies, setMovies] = useState([]);
  const [movie, setMovie] = useState("");
  const [recommendations, setRecommendations] = useState([]);
  const [searchFocused, setSearchFocused] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Selected movie for details modal
  const [selectedMovie, setSelectedMovie] = useState(null);

  useEffect(() => {
    fetch(`${API}/movies`)
      .then((res) => res.json())
      .then((data) => setMovies(data.movies || []))
      .catch(() => {
        setError("Unable to connect to recommendation engine.");
      });
  }, []);

  // Close modal with Escape key
  useEffect(() => {
    const handleEscape = (event) => {
      if (event.key === "Escape") {
        setSelectedMovie(null);
      }
    };

    document.addEventListener("keydown", handleEscape);

    return () => {
      document.removeEventListener("keydown", handleEscape);
    };
  }, []);

  const suggestions = useMemo(() => {
    if (!movie.trim()) return [];

    return movies
      .filter((title) =>
        title.toLowerCase().includes(movie.toLowerCase())
      )
      .slice(0, 7);
  }, [movie, movies]);

  const getRecommendations = async (selectedMovieTitle = movie) => {
    if (!selectedMovieTitle.trim()) return;

    setMovie(selectedMovieTitle);
    setLoading(true);
    setError("");
    setRecommendations([]);
    setSelectedMovie(null);

    try {
      const response = await fetch(
        `${API}/recommend?movie=${encodeURIComponent(
          selectedMovieTitle
        )}&n=10`
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Movie not found");
      }

      setRecommendations(data.recommendations || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      {/* BACKGROUND */}
      <div className="ambient ambient-one"></div>
      <div className="ambient ambient-two"></div>
      <div className="noise"></div>

      {/* NAVBAR */}
      <nav className="navbar">
        <div className="brand">
          <div className="brand-mark">M</div>

          <div>
            <div className="brand-name">
              Movie<span>Mind</span>
            </div>

            <div className="brand-subtitle">
              RECOMMENDATION ENGINE
            </div>
          </div>
        </div>

        <div className="nav-status">
          <span className="status-dot"></span>
          AI ENGINE ONLINE
        </div>
      </nav>

      {/* HERO */}
      <main>
        <section className="hero">
          <div className="hero-badge">
            <span>✦</span>
            AI-POWERED DISCOVERY
          </div>

          <h1>
            Your next
            <span> obsession</span>
            <br />
            starts here.
          </h1>

          <p className="hero-description">
            Discover movies that match your taste using intelligent
            hybrid recommendation technology.
          </p>

          {/* SEARCH */}
          <div
            className={`search-wrapper ${
              searchFocused ? "focused" : ""
            }`}
          >
            <div className="search-icon">⌕</div>

            <input
              value={movie}
              onChange={(e) => setMovie(e.target.value)}
              onFocus={() => setSearchFocused(true)}
              onBlur={() =>
                setTimeout(() => setSearchFocused(false), 150)
              }
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  getRecommendations();
                }
              }}
              placeholder="Search a movie you love..."
            />

            {movie && (
              <button
                className="clear-btn"
                onClick={() => {
                  setMovie("");
                  setRecommendations([]);
                  setSelectedMovie(null);
                }}
              >
                ×
              </button>
            )}

            <button
              className="discover-btn"
              onClick={() => getRecommendations()}
              disabled={loading}
            >
              {loading ? (
                <span className="spinner"></span>
              ) : (
                <>
                  Discover
                  <span>→</span>
                </>
              )}
            </button>

            {/* SEARCH SUGGESTIONS */}
            {searchFocused && suggestions.length > 0 && (
              <div className="suggestions">
                <div className="suggestions-label">
                  MOVIES
                </div>

                {suggestions.map((title) => (
                  <button
                    key={title}
                    className="suggestion"
                    onMouseDown={() => {
                      setMovie(title);
                      getRecommendations(title);
                    }}
                  >
                    <div className="suggestion-icon">
                      ▶
                    </div>

                    <span>{title}</span>

                    <span className="suggestion-arrow">
                      →
                    </span>
                  </button>
                ))}
              </div>
            )}
          </div>

          <div className="hero-hint">
            <span>↵</span>
            Press Enter to discover
          </div>
        </section>

        {/* ERROR */}
        {error && (
          <div className="error-box">
            <span>!</span>
            {error}
          </div>
        )}

        {/* RESULTS */}
        {recommendations.length > 0 && (
          <section className="results-section">
            {/* RESULT HEADER */}
            <div className="result-heading">
              <div>
                <div className="eyebrow">
                  YOUR DISCOVERY
                </div>

                <h2>
                  Because you liked{" "}
                  <span>{movie}</span>
                </h2>
              </div>

              <div className="result-count">
                <strong>
                  {recommendations.length}
                </strong>

                <span>matches found</span>
              </div>
            </div>

            {/* ENGINE INFO */}
            <div className="engine-panel">
              <div className="engine-icon">
                ✦
              </div>

              <div className="engine-copy">
                <strong>
                  Hybrid Recommendation Intelligence
                </strong>

                <span>
                  Your results combine content similarity with
                  collaborative filtering to rank movies that
                  match your taste.
                </span>
              </div>

              <div className="engine-tags">
                <span>70% CONTENT</span>
                <span>30% COLLABORATIVE</span>
                <span>HYBRID AI</span>
              </div>
            </div>

            {/* MOVIE GRID */}
            <div className="movie-grid">
              {recommendations.map((item, index) => (
                <article
                  className={`movie-card ${
                    index === 0 ? "featured" : ""
                  }`}
                  key={`${item.title}-${index}`}
                  onClick={() => setSelectedMovie(item)}
                  tabIndex="0"
                  role="button"
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      setSelectedMovie(item);
                    }
                  }}
                >
                  <div className="card-top">
                    <div className="rank">
                      {String(index + 1).padStart(2, "0")}
                    </div>

                    {index === 0 && (
                      <div className="top-match">
                        TOP MATCH
                      </div>
                    )}
                  </div>

                  <div className="poster">
                    {item.poster ? (
                      <img
                        src={item.poster}
                        alt={item.title}
                        onError={(e) => {
                          e.currentTarget.style.display = "none";
                        }}
                      />
                    ) : (
                      <div className="poster-fallback">
                        {item.title.charAt(0).toUpperCase()}
                      </div>
                    )}

                    <div className="poster-overlay"></div>

                    <div className="poster-label">
                      MOVIEMIND
                    </div>

                    {/* CLICK INDICATOR */}
                    <div className="view-details">
                      VIEW DETAILS
                    </div>
                  </div>

                  <div className="movie-info">
                    <h3>{item.title}</h3>

                    <div className="match-row">
                      <div className="match-label">
                        <span className="match-dot"></span>
                        AI MATCH
                      </div>

                      <strong>
                        {item.similarity}%
                      </strong>
                    </div>

                    <div className="match-bar">
                      <div
                        style={{
                          width: `${Math.min(
                            item.similarity,
                            100
                          )}%`,
                        }}
                      ></div>
                    </div>
                  </div>
                </article>
              ))}
            </div>
          </section>
        )}

        {/* EMPTY STATE */}
        {!loading &&
          recommendations.length === 0 &&
          !error && (
            <section className="explore-section">
              <div className="explore-line"></div>

              <div className="explore-content">
                <div className="explore-number">
                  01
                </div>

                <div>
                  <div className="eyebrow">
                    HOW IT WORKS
                  </div>

                  <h2>
                    One movie.
                    <br />
                    <span>Endless possibilities.</span>
                  </h2>
                </div>

                <div className="explore-description">
                  <p>
                    Enter a movie you already love.
                    MovieMind combines content similarity
                    and collaborative filtering to rank
                    movies that match your taste.
                  </p>
                </div>
              </div>

              <div className="feature-strip">
                <div>
                  <span>01</span>
                  <strong>Understand</strong>
                  <p>
                    Movie content is transformed
                    into meaningful features.
                  </p>
                </div>

                <div>
                  <span>02</span>
                  <strong>Compare</strong>
                  <p>
                    Collaborative and content models
                    identify relevant movies.
                  </p>
                </div>

                <div>
                  <span>03</span>
                  <strong>Recommend</strong>
                  <p>
                    Hybrid scoring ranks the best
                    matches for you.
                  </p>
                </div>
              </div>
            </section>
          )}
      </main>

      {/* FOOTER */}
      <footer>
        <div className="footer-brand">
          MOVIEMIND
        </div>

        <div>
          Intelligent movie discovery
        </div>

        <div>
          CSE / AI & ML PROJECT
        </div>
      </footer>

      {/* =====================================================
          MOVIE DETAILS MODAL
          ===================================================== */}

      {selectedMovie && (
        <div
          className="movie-modal-backdrop"
          onClick={() => setSelectedMovie(null)}
        >
          <div
            className="movie-modal"
            onClick={(e) => e.stopPropagation()}
          >
            {/* CLOSE */}
            <button
              className="modal-close"
              onClick={() => setSelectedMovie(null)}
              aria-label="Close"
            >
              ×
            </button>

            {/* MODAL POSTER */}
            <div className="modal-poster">
              {selectedMovie.poster ? (
                <img
                  src={selectedMovie.poster}
                  alt={selectedMovie.title}
                />
              ) : (
                <div className="modal-poster-fallback">
                  {selectedMovie.title
                    .charAt(0)
                    .toUpperCase()}
                </div>
              )}
            </div>

            {/* MODAL CONTENT */}
            <div className="modal-content">
              <div className="modal-eyebrow">
                MOVIEMIND DISCOVERY
              </div>

              <h2>{selectedMovie.title}</h2>

              {/* META */}
              <div className="modal-meta">
                {selectedMovie.release_date && (
                  <span>
                    {selectedMovie.release_date.substring(
                      0,
                      4
                    )}
                  </span>
                )}

                {selectedMovie.vote_average !== null &&
                  selectedMovie.vote_average !==
                    undefined && (
                    <span>
                      ★{" "}
                      {Number(
                        selectedMovie.vote_average
                      ).toFixed(1)}
                    </span>
                  )}

                <span className="modal-match">
                  {selectedMovie.similarity}% AI MATCH
                </span>
              </div>

              {/* SUMMARY */}
              <div className="modal-section">
                <div className="modal-section-title">
                  SUMMARY
                </div>

                <p className="modal-overview">
                  {selectedMovie.overview ||
                    "No summary is available for this movie."}
                </p>
              </div>

              {/* RECOMMENDATION SIGNALS */}
              <div className="modal-section">
                <div className="modal-section-title">
                  RECOMMENDATION SIGNALS
                </div>

                <div className="modal-scores">
                  <div>
                    <span>CONTENT</span>
                    <strong>
                      {selectedMovie.content_score}%
                    </strong>
                  </div>

                  <div>
                    <span>COLLABORATIVE</span>
                    <strong>
                      {selectedMovie.collaborative_score}%
                    </strong>
                  </div>

                  <div>
                    <span>HYBRID MATCH</span>
                    <strong>
                      {selectedMovie.similarity}%
                    </strong>
                  </div>
                </div>
              </div>

              <div className="modal-footer-hint">
                Click outside or press ESC to close
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;