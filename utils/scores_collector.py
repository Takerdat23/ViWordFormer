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
    Crawl the directory 'root_dir', find schema subdirs (e.g. s1, s2),
    then model subdirs (BiGRU, GRU, LSTM...), then tokenizer subdirs
    (bpe, unigram, wordpiece, vipher), and finally the 'scores.json'.

    Returns a dict of structure:
        scores_dict[model][col_label] = [precision, recall, f1]

    where col_label is one of the 7 columns we want to see in the final DF:
       ViPher*, BPE*, Unigram*, WordPiece*, BPE, Unigram, WordPiece
    """
    possible_schemas = ["s1", "s2"]  
    models = ["BiGRU", "BiLSTM", "LSTM", "GRU"]  
    tokenizers = ["vipher", "bpe", "unigram", "wordpiece"]  

    scores_dict = {}  # { model: { col_label: [prec, rec, f1] } }

    for schema in possible_schemas:
        schema_path = os.path.join(root_dir, schema)
        if not os.path.isdir(schema_path):
            continue  # skip if s1 or s2 folder doesn't exist

        for model in models:
            model_path = os.path.join(schema_path, model)
            if not os.path.isdir(model_path):
                continue

            for tk in tokenizers:
                tk_path = os.path.join(model_path, tk)
                if not os.path.isdir(tk_path):
                    continue

                # Within this tokenizer folder, find subdirs with scores.json
                for sub_name in os.listdir(tk_path):
                    sub_path = os.path.join(tk_path, sub_name)
                    if not os.path.isdir(sub_path):
                        continue

                    score_file = os.path.join(sub_path, "scores.json")
                    if os.path.isfile(score_file):
                        # Read the JSON: 
                        with open(score_file, "r", encoding="utf-8") as f:
                            # Adjust to the actual structure of your JSON.
                            # In your code, you had data = json.load(f)[0]["test_scores"]
                            # or it could just be data = json.load(f)
                            data = json.load(f)[0]["test_scores"]  
                            prec = data.get("precision", 0.0)
                            rec  = data.get("recall", 0.0)
                            f1   = data.get("f1", 0.0)

                        col_label = get_column_name(schema, tk)  
                        if model not in scores_dict:
                            scores_dict[model] = {}
                        scores_dict[model][col_label] = [prec, rec, f1]

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