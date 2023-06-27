"""Run BM25 Index to Wikipedia 6M."""
import json
from transformers import AutoTokenizer
from rank_bm25 import BM25Okapi
import pickle


def load_json(path):
    # load jsonl file
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data

corpus = load_json("wikipedia_6m.jsonl")
"""
wikipedia_6m.jsonl format:

{"id": "Q16221564", "content": "Leaf Huang"}
{"id": "Q4933113", "content": "Bob Lanois"}
"""
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

corpus = [
    d["content"] for d in corpus
]

tokenized_corpus = [tokenizer.tokenize(doc) for doc in corpus]

bm25 = BM25Okapi(tokenized_corpus)
# <rank_bm25.BM25Okapi at 0x1047881d0>
# save bm25 in pickle
with open("wikipedia6m_index.pkl", "wb") as f:
    pickle.dump(bm25, f)

for query in ["Astacia", "wheelchair curling", "B\u00e1nh Department"]:
    tokenized_query = tokenizer.tokenize(query)
    print(bm25.get_top_n(tokenized_query, corpus, n=3))