from typing import Dict, List
from annoy import AnnoyIndex
import sentence_transformers
import tqdm
import random

def retrieve_from_kb(
    data_kb: List[Dict],
    data_query: List[Dict],
    k=5,
    prevent_hardmatch=False,
) -> List[List[Dict]]:
    """
    Each line in data_kb and data_query should have a "src" key by which they are retrieved.

    TODO: test this
    """
    R_FALLBACK = random.Random(0)

    src_all = list({line["src"] for line in data_kb + data_query})

    model = sentence_transformers.SentenceTransformer("all-MiniLM-L12-v2")
    src_to_embd = {
        txt: txt_e
        for txt, txt_e in zip(src_all, model.encode(src_all, show_progress_bar=True, batch_size=256))
    }
    for line in data_kb + data_query:
        line["embd"] = src_to_embd[line["src"]]

    data_kb_index = AnnoyIndex(len(data_kb[0]["embd"]), 'dot')
    for i, line in enumerate(data_kb):
        data_kb_index.add_item(i, line["embd"])
    data_kb_index.build(50)


    data_out = []
    for line in tqdm.tqdm(data_query):
        # select increasingly large neighbours because we filter later on
        for n in [50, 100, 500, 1000, 5000]:
            idx, dists = data_kb_index.get_nns_by_vector(line["embd"], n=n, include_distances=True)
            tgts = [
                data_kb[i]
                for i, d in zip(idx, dists)
            ]
            if prevent_hardmatch:
                tgts = [
                    x for x in tgts
                    if x["src"] != line["src"] and ("mt" not in x or x["mt"] != line["mt"])
                ]
            tgts = tgts[:k]
            if len(tgts) == k:
                break
        else:
            # just take random examples at this point
            tgts += R_FALLBACK.sample(data_kb, k-len(tgts))
        
        data_out.append(tgts)

    # remove the embeddings again
    for line in data_out:
        for line in line:
            if "embd" in line:
                del line["embd"]

    return data_out


"""
import comet_poly.retrieval
import json

print(json.dumps(comet_poly.retrieval.retrieve_from_kb(
    data_kb=[{"src": "dog"}, {"src": "cat"}, {"src": "salad"}],
    data_query=[{"src": "dog"}, {"src": "carrot"}],
    k=1,
    prevent_hardmatch=False,
), indent=2))

print(json.dumps(comet_poly.retrieval.retrieve_from_kb(
    data_kb=[{"src": "dog"}, {"src": "cat"}, {"src": "salad"}],
    data_query=[{"src": "dog"}, {"src": "carrot"}],
    k=1,
    prevent_hardmatch=True,
), indent=2))


print(json.dumps(comet_poly.retrieval.retrieve_from_kb(
    data_kb=[{"src": "b"}, {"src": "c"}, {"src": "a"}, {"src": "x"}],
    data_query=[{"src": "a"}],
    k=3,
    prevent_hardmatch=False,
), indent=2))
"""