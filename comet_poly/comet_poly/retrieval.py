from typing import Dict, List
from annoy import AnnoyIndex
import sentence_transformers
import tqdm
import random

def retrieve_from_kb(
    data: List[Dict],
    data_kb: List[Dict],
    k=5,
    prevent_hardmatch=False,
    key="src",
) -> List[List[Dict]]:
    """
    Each line in data_kb and data_query should have a "src" key (default) by which they are retrieved.
    """
    R_FALLBACK = random.Random(0)

    key_all = list({line[key] for line in data_kb + data})

    model = sentence_transformers.SentenceTransformer("all-MiniLM-L12-v2")
    key_to_embd = {
        txt: txt_e
        for txt, txt_e in zip(key_all, model.encode(key_all, show_progress_bar=True, batch_size=256))
    }
    for line in data_kb + data:
        line["embd"] = key_to_embd[line[key]]

    data_kb_index = AnnoyIndex(len(data_kb[0]["embd"]), 'dot')
    for i, line in enumerate(data_kb):
        data_kb_index.add_item(i, line["embd"])
    data_kb_index.build(50)


    data_out = []
    for line in tqdm.tqdm(data):
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
                    if x[key] != line[key]
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
    data=[{"src": "dog"}, {"src": "carrot"}],
    data_kb=[{"src": "dog"}, {"src": "cat"}, {"src": "salad"}],
    k=1,
    prevent_hardmatch=False,
), indent=2))

print(json.dumps(comet_poly.retrieval.retrieve_from_kb(
    data=[{"src": "dog"}, {"src": "carrot"}],
    data_kb=[{"src": "dog"}, {"src": "cat"}, {"src": "salad"}],
    k=1,
    prevent_hardmatch=True,
), indent=2))


print(json.dumps(comet_poly.retrieval.retrieve_from_kb(
    data=[{"src": "a"}],
    data_kb=[{"src": "b"}, {"src": "c"}, {"src": "a"}, {"src": "x"}],
    k=3,
    prevent_hardmatch=False,
), indent=2))

# or retrieval on our whole data
import datasets
data_kb = list(datasets.load_dataset("zouharvi/wmt-human-all", split="train"))
data_retrieved = comet_poly.retrieval.retrieve_from_kb(
    data=data,
    data_kb=data_kb,
    k=1,
    prevent_hardmatch=False,
)
# add the retrieved data
for line, lines_retrieved in zip(data, data_retrieved):
    for i in range(len(lines_retrieved)):
        line[f"src{i+2}"] = lines_retrieved[i]["src"]
        line[f"mt{i+2}"] = lines_retrieved[i]["mt"]
        line[f"score{i+2}"] = lines_retrieved[i]["score"]

"""