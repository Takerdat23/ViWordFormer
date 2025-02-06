import os
import json
import pandas as pd
import numpy as np

def get_column_name(schema, raw_tk):
    """
    Map (schema, raw tokenizer folder) -> one of the 7 desired column labels.
    schema='s1' -> starred version, schema='s2' -> unstarred.
    raw_tk in ["vipher", "bpe", "unigram", "wordpiece"].
    
    We produce exactly one of:
      "ViPher*", "BPE*", "Unigram*", "WordPiece*", "BPE", "Unigram", "WordPiece"
    """
    # For convenience, define a sub-mapping for how to "capitalize" each tk name:
    capital_map = {
        "vipher": "ViPher",
        "bpe": "BPE",
        "unigram": "Unigram",
        "wordpiece": "WordPiece"
    }
    if schema == "s1":
        # Star version
        return f"{capital_map[raw_tk]}*"
    else:
        # No star
        return capital_map[raw_tk]

def collect_scores(root_dir="sentiment"):
    """
    Collects model scores for both traditional classification and Aspect-Based Sentiment Analysis (ABSA).
    
    - Handles both single sentiment scores and aspect-based sentiment analysis scores.
    - Supports multiple schemas, models, and tokenizers.

    Returns:
        scores_dict[model][col_label] = {
            'precision': x, 'recall': y, 'f1': z
        }  # For normal classification

        OR

        scores_dict[model][col_label] = {
            'aspect': { 'precision': x, 'recall': y, 'f1': z },
            'sentiment': { 'precision': x, 'recall': y, 'f1': z }
        }  # For ABSA
    """
    possible_schemas = ["s1", "s2"]
    models = ["BiGRU", "BiLSTM", "LSTM", "GRU"]
    tokenizers = ["vipher", "bpe", "unigram", "wordpiece"]

    scores_dict = {}

    for schema in possible_schemas:
        schema_path = Path(root_dir) / schema
        if not schema_path.is_dir():
            continue  

        for model in models:
            model_path = schema_path / model
            if not model_path.is_dir():
                continue

            for tk in tokenizers:
                tk_path = model_path / tk
                if not tk_path.is_dir():
                    continue

                for sub_dir in tk_path.iterdir():
                    score_file = sub_dir / "scores.json"
                    if not score_file.is_file():
                        continue

                    try:
                        with open(score_file, "r", encoding="utf-8") as f:
                            data = json.load(f)

                            # Normalize JSON structure
                            if isinstance(data, list) and data and "test_scores" in data[0]:
                                scores = data[0]["test_scores"]
                            else:
                                scores = data  # Direct structure without wrapping

                            col_label = get_column_name(schema, tk)
                            if model not in scores_dict:
                                scores_dict[model] = {}

                            # Check if it follows the ABSA format (aspect & sentiment keys exist)
                            # This is temporary and will be removed once all models are updated.
                            if "aspect" in scores and "sentiment" in scores:     
                                for key, value in scores.items():
                                    if isinstance(value, dict):  # Task-specific scores
                                            formatted_scores[key] = {
                                                'precision': value.get("precision", 0.0),
                                                'recall': value.get("recall", 0.0),
                                                'f1': value.get("f1", 0.0),
                                            }
                                        
                            else:  # Traditional classification format
                                scores_dict[model][col_label] = {
                                    'precision': scores.get("precision", 0.0),
                                    'recall': scores.get("recall", 0.0),
                                    'f1': scores.get("f1", 0.0),
                                }

                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"Skipping {score_file} due to error: {e}")
                        continue

    return scores_dict

def build_dataframe(scores_dict):
    """
    Build a DataFrame with:
      - Rows = [precision, recall, f1]
      - Columns = MultiIndex of (Model, one_of_7_tokenizer_columns).

    The 7 tokenizer columns are in a fixed order:
       ["ViPher*", "BPE*", "Unigram*", "WordPiece*", "BPE", "Unigram", "WordPiece"]
    """
    # Known columns in fixed order
    all_tokenizer_cols = ["ViPher*", "BPE*", "Unigram*", "WordPiece*",
                          "BPE", "Unigram", "WordPiece"]

    # Models come from the dictionary (sorted for consistency)
    all_models = sorted(scores_dict.keys())

    # Build a MultiIndex:
    #    top level = model, sub-level = each of the 7 tokenizer cols
    arrays = []
    for model in all_models:
        for tk_col in all_tokenizer_cols:
            arrays.append((model, tk_col))

    columns = pd.MultiIndex.from_tuples(arrays, names=["Model", "Tokenizer"])
    index = ["precision", "recall", "f1"]

    # Construct an empty DataFrame
    df = pd.DataFrame(index=index, columns=columns, dtype=float)

    # Fill the DataFrame
    for model in all_models:
        sub_dict = scores_dict[model]  # { "ViPher*": [prec, rec, f1], ... }
        for tk_col in all_tokenizer_cols:
            if tk_col not in sub_dict:
                continue  # remains NaN
            prec, rec, f1 = sub_dict[tk_col]
            df.loc["precision",  (model, tk_col)] = round(prec*100, 2)
            df.loc["recall",     (model, tk_col)] = round(rec*100, 2)
            df.loc["f1",         (model, tk_col)] = round(f1*100, 2)

    return df

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--path", type=str, required=True)
args = parser.parse_args()

if __name__ == "__main__":
    scores_dict = collect_scores(
        root_dir=args.path
    )

    df = build_dataframe(scores_dict)

    print(df)
    # get final folder name from the path
    final_folder = args.path.split("/")[-1]
    save_path = args.path + f"/{final_folder}_scores.xlsx"
    df.to_excel(save_path)