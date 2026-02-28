import pandas as pd
import numpy as np

df = pd.read_csv("data/processed/movies_final.csv")

# Build director stats
director_stats = df.groupby('Director')['vote_average'].agg(['mean', 'count']).reset_index()
director_stats.columns = ['director', 'success_rate', 'movie_count']
director_stats = director_stats[director_stats['movie_count'] >= 1]
director_stats.to_csv("data/processed/director_stats.csv", index=False)
print(f"✅ Director stats saved: {len(director_stats)} directors")

# Build actor stats
actor_stats = df.groupby('Star1')['vote_average'].agg(['mean', 'count']).reset_index()
actor_stats.columns = ['actor', 'success_rate', 'movie_count']
actor_stats = actor_stats[actor_stats['movie_count'] >= 1]
actor_stats.to_csv("data/processed/actor_stats.csv", index=False)
print(f"✅ Actor stats saved: {len(actor_stats)} actors")

# Build genre stats
genre_stats = df.groupby('Genre')['vote_average'].agg(['mean', 'count']).reset_index()
genre_stats.columns = ['genre', 'success_rate', 'movie_count']
genre_stats.to_csv("data/processed/genre_stats.csv", index=False)
print(f"✅ Genre stats saved: {len(genre_stats)} genres")

# Check Christopher Nolan specifically
nolan = director_stats[director_stats['director'] == 'Christopher Nolan']
print(f"\n🎬 Christopher Nolan stats: {nolan}")