import argparse
import json
import time
import pickle
from multiprocessing import Pool
from transformers import AutoTokenizer
from rank_bm25 import BM25Okapi
from tqdm import tqdm

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data

def run_query(query):
    tokenized_query = tokenizer.tokenize(query["prediction"])
    results = bm25.get_top_n(tokenized_query, corpus, n=5)
    return {"data_id": query["data_id"], "prediction": query["prediction"], "bm25": results}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="oven_predictions/debug.jsonl")
    parser.add_argument("--output_file", type=str, default="oven_predictions/debug_bm25.jsonl")
    args = parser.parse_args()

    corpus = load_json("wikipedia_6m.jsonl")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    corpus = [
        [d["content"], d["id"]] for d in corpus
    ]

    print("Loading bm25 pickle...")
    with open("wikipedia6m_index.pkl", "rb") as f:
        bm25 = pickle.load(f)

    input_query = load_json(args.input_file)

    print("Running bm25 query...")
    with Pool() as p:
        max_ = len(input_query)
        with tqdm(total=max_) as pbar:
            for i, _ in tqdm(enumerate(p.imap_unordered(run_query, input_query))):
                pbar.update()
        output = list(p.imap_unordered(run_query, input_query))

    with open(args.output_file, "w", encoding="utf-8") as f:
        for d in output:
            f.write(json.dumps(d) + "\n")
