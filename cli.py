"""
Command-line interface: rate some movies, get your top 5.

    python cli.py              # interactive
    python cli.py --demo       # runs a built-in example profile (no typing needed)
"""

import argparse

from src.recommender import (
    MIN_USER_RATINGS,
    InvalidRatingError,
    MovieRecommender,
)

# A small example profile (Matrix, Star Wars IV, Shawshank, Pulp Fiction, Toy Story, Taxi Driver)
DEMO_RATINGS = {2571: 5.0, 260: 4.5, 318: 5.0, 296: 4.0, 1: 4.5, 111: 3.5}


def ask_rating(prompt: str):
    """Return a float rating, or None for skip. Re-asks on bad input."""
    while True:
        raw = input(prompt).strip().lower()
        if raw in ("", "s", "skip"):
            return None
        try:
            value = float(raw)
        except ValueError:
            print("  Enter a number from 0.5 to 5 (half-star steps), or press Enter to skip.")
            continue
        if 0.5 <= value <= 5.0 and (value * 2) % 1 == 0:
            return value
        print("  Rating must be 0.5 to 5.0 in half-star steps.")


def print_recs(recs) -> None:
    print("\nYour top 5 recommendations")
    print("-" * 60)
    for rank, row in recs.iterrows():
        print(f"{rank}. {row.title}  [predicted {row.predicted_rating:.2f}/5]")
        print(f"   {row.genres}")


def interactive(rec: MovieRecommender) -> None:
    ratings = {}
    print("Rate movies you have seen. Pick one of:")
    print("  [1] rate from a list of popular movies")
    print("  [2] search for a movie by title")
    print("  [d] done (get recommendations)   [q] quit")
    while True:
        print(f"\nYou have rated {len(ratings)} movie(s) (need at least {MIN_USER_RATINGS}; 15+ works better).")
        choice = input("Choice (1 / 2 / d / q): ").strip().lower()
        if choice == "q":
            return
        if choice == "d":
            try:
                print_recs(rec.recommend(ratings))
            except InvalidRatingError as err:
                print(f"  {err}")
                continue
            return
        if choice == "1":
            page = rec.popular_movies(60)
            page = page[~page.movieId.isin(ratings)].head(10)
            for _, row in page.iterrows():
                r = ask_rating(f"  {row.title}  ({row.genres})  rating 0.5-5 / Enter=skip: ")
                if r is not None:
                    ratings[int(row.movieId)] = r
        elif choice == "2":
            hits = rec.search_movies(input("  Title contains: "), limit=8)
            if hits.empty:
                print("  No matches.")
                continue
            for i, row in hits.iterrows():
                print(f"  {i + 1}. {row.title}  ({row.genres})")
            pick = input("  Which number (Enter to cancel): ").strip()
            if pick.isdigit() and 1 <= int(pick) <= len(hits):
                row = hits.iloc[int(pick) - 1]
                r = ask_rating(f"  Your rating for {row.title} (0.5-5): ")
                if r is not None:
                    ratings[int(row.movieId)] = r


def main() -> None:
    parser = argparse.ArgumentParser(description="Movie recommender (command line)")
    parser.add_argument("--demo", action="store_true", help="use a built-in example profile")
    args = parser.parse_args()

    rec = MovieRecommender()
    if args.demo:
        print("Demo profile:")
        for mid, r in DEMO_RATINGS.items():
            print(f"  {rec.movies.loc[mid, 'title']}: {r}")
        print_recs(rec.recommend(DEMO_RATINGS))
    else:
        interactive(rec)


if __name__ == "__main__":
    main()
