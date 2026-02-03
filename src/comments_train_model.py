import datetime
import glob
import os
import pickle
import random

import numpy as np
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser


TIME_PERIODS = ["before_2016", "2017_2020", "2021_2024"]
SEED = 23


def get_date_from_comment(comment):
    """Extract date from a comment dictionary."""
    try:
        return datetime.datetime.strptime(comment["date"], "%Y-%m-%d").date()
    except (KeyError, ValueError):
        try:
            return datetime.datetime.fromtimestamp(int(comment["timestamp"])).date()
        except (KeyError, ValueError):
            return None


def get_period(date):
    """Determine which time period a date belongs to."""
    if date is None:
        return None
    year = date.year
    if year <= 2016:
        return "before_2016"
    if 2017 <= year <= 2020:
        return "2017_2020"
    if 2021 <= year <= 2024:
        return "2021_2024"
    return None


def build_bigram_model(comments):
    """Build a bigram model for the given comments."""
    sentences = [c["processed_text"] for c in comments if "processed_text" in c]
    phrases = Phrases(sentences, min_count=10, threshold=10)
    return Phraser(phrases)


def apply_bigrams(comments, bigram_model):
    """Apply bigram model to comments."""
    return [
        bigram_model[c["processed_text"]]
        for c in comments
        if "processed_text" in c
    ]


def create_or_update_model(
    period,
    comments,
    vector_size,
    window,
    min_count,
    workers,
    sg,
    epochs,
    existing_model=None,
):
    """Create a new model or update an existing one."""
    if existing_model is None:
        model = Word2Vec(
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            sg=sg,
            seed=SEED,
        )
        model.build_vocab(comments)
        print(f"{period} vocabulary size: {len(model.wv.index_to_key)}")
    else:
        model = existing_model
        model.build_vocab(comments, update=True)
        print(f"{period} vocabulary size: {len(model.wv.index_to_key)}")

    model.train(comments, total_examples=len(comments), epochs=epochs)
    return model


def save_model(model, subreddit, period, model_dir, is_interim=False):
    """Save model to disk."""
    if is_interim:
        path = f"{model_dir}/interim/{subreddit}_{period}_interim.model"
    else:
        path = f"{model_dir}/{subreddit}_{period}.model"
    model.save(path)


def load_global_bigram_model(global_bigram_path):
    """Load global bigram model if available, with identical output."""
    if os.path.exists(global_bigram_path):
        print(f"Loading global bigram model from {global_bigram_path}")
        return Phraser.load(global_bigram_path)
    print(f"Global bigram model not found at {global_bigram_path}, will train on each chunk.")
    return None


def get_pickle_files(base_data_dir, subreddit):
    pattern = f"{base_data_dir}/{subreddit}/{subreddit}_batch*.pkl"
    return sorted(glob.glob(pattern))


def update_period_buckets(comments, comments_by_period):
    for comment in comments:
        date = get_date_from_comment(comment)
        period = get_period(date)
        if period:
            comments_by_period[period].append(comment)


def process_chunk(
    period,
    chunk,
    global_bigram_model,
    models,
    vector_size,
    window,
    min_count,
    workers,
    sg,
    epochs,
    min_comments_to_train,
    subreddit,
    model_dir,
):
    if global_bigram_model is not None:
        bigram_model = global_bigram_model
    else:
        bigram_model = build_bigram_model(chunk)

    processed_chunk = apply_bigrams(chunk, bigram_model)

    if len(processed_chunk) > min_comments_to_train:
        model = create_or_update_model(
            period,
            processed_chunk,
            vector_size,
            window,
            min_count,
            workers,
            sg,
            epochs,
            models[period],
        )
        models[period] = model
        save_model(model, subreddit, period, model_dir, is_interim=True)


def finalize_period(
    period,
    remaining_comments,
    global_bigram_model,
    models,
    vector_size,
    window,
    min_count,
    workers,
    sg,
    epochs,
    min_comments_to_train,
    subreddit,
    model_dir,
):
    if len(remaining_comments) > min_comments_to_train:
        print(f"Processing final {len(remaining_comments)} comments for {period}")
        if global_bigram_model is not None:
            bigram_model = global_bigram_model
        else:
            bigram_model = build_bigram_model(remaining_comments)

        processed_chunk = apply_bigrams(remaining_comments, bigram_model)
        model = create_or_update_model(
            period,
            processed_chunk,
            vector_size,
            window,
            min_count,
            workers,
            sg,
            epochs,
            models[period],
        )
        models[period] = model
        save_model(model, subreddit, period, model_dir, is_interim=False)
    else:
        print(
            f"Skipping final {len(remaining_comments)} comments for {period} "
            f"(less than minimum required)"
        )


def build_models_for_subreddit(
    subreddit,
    base_data_dir,
    model_dir,
    vector_size=300,
    window=5,
    min_count=5,
    epochs=5,
    workers=16,
    sg=0,
    min_comments_to_train=10_000,
    chunk_size=1_000_000,
    global_bigram_path=None,
):
    models = {period: None for period in TIME_PERIODS}

    global_bigram_model = load_global_bigram_model(global_bigram_path)
    if global_bigram_model is None:
        return

    pickle_files = get_pickle_files(base_data_dir, subreddit)
    if not pickle_files:
        print(f"No pickle files found for {subreddit} in {base_data_dir}/{subreddit}/")
        return

    comments_by_period = {period: [] for period in TIME_PERIODS}

    for file_path in pickle_files:
        try:
            with open(file_path, "rb") as f:
                comments = pickle.load(f)
            print(f"Loaded {len(comments)} comments from {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

        update_period_buckets(comments, comments_by_period)

        for period in TIME_PERIODS:
            period_comments = comments_by_period[period]
            while len(period_comments) >= chunk_size:
                print(f"Processing chunk of {chunk_size} comments for {period}")
                chunk = period_comments[:chunk_size]
                period_comments = period_comments[chunk_size:]

                process_chunk(
                    period,
                    chunk,
                    global_bigram_model,
                    models,
                    vector_size,
                    window,
                    min_count,
                    workers,
                    sg,
                    epochs,
                    min_comments_to_train,
                    subreddit,
                    model_dir,
                )

            comments_by_period[period] = period_comments

    for period, remaining_comments in comments_by_period.items():
        finalize_period(
            period,
            remaining_comments,
            global_bigram_model,
            models,
            vector_size,
            window,
            min_count,
            workers,
            sg,
            epochs,
            min_comments_to_train,
            subreddit,
            model_dir,
        )

    for period, model in models.items():
        if model is not None:
            save_model(model, subreddit, period, model_dir, is_interim=False)

    print(f"Model saved to {model_dir}")
    print(f"Completed building models for {subreddit}")


def main():
    model_dir = "data/models"
    global_bigram_path = "data/models/bigram/bigram.phr"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(f"{model_dir}/interim", exist_ok=True)

    random.seed(SEED)
    np.random.seed(SEED)

    subreddits = ["democrats", "republican", "liberal"]
    for subreddit in subreddits:
        build_models_for_subreddit(
            subreddit,
            base_data_dir="data/processed_comments",
            model_dir=model_dir,
            vector_size=300,
            window=5,
            min_count=10,
            epochs=5,
            workers=16,
            sg=0,
            min_comments_to_train=10_000,
            chunk_size=1_000_000,
            global_bigram_path=global_bigram_path,
        )


if __name__ == "__main__":
    main()