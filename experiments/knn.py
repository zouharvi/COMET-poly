import argparse
import pandas as pd
import sentence_transformers
import numpy as np
import scipy
import re

""" Example
python knn.py --setup srcCmt --m 05 --data_path 'data/'
"""

def softmax_kernel(D: np.ndarray, gamma: float | np.ndarray) -> np.ndarray:
    # 1) compute raw scores
    scores = -D / gamma

    # 2) subtract row‐wise max for stability
    max_scores = np.max(scores, axis=1, keepdims=True)
    scores -= max_scores

    # 3) exponentiate and renormalize
    exp_scores = np.exp(scores)
    W = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    return W

def extract_numbers(strings):
    number_pattern = r'[-+]?\d*\.\d+|[-+]?\d+'
    numbers = set()
    for s in strings:
        found = re.findall(number_pattern, s)
        numbers.update(map(float, found))
    return np.sort([int(n) for n in numbers]).tolist()

if __name__ == "__main__":
    # parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate retrieval with different setups")
    parser.add_argument(
        '--setup', '-s',
        type=str,
        choices=['mt', 'src', 'srcAmt', 'srcCmt'],
        required=True,
        help="which embeddings to use: 'mt', 'src', 'srcAmt', or 'srcCmt'"
    )
    parser.add_argument(
        '--m',
        type=str,
        choices=['05', '07', '11'],
        required=True,
        help="'05', '07', or '11'"
    )
    parser.add_argument(
        '--data_path', '-d',
        type=str,
        default='',
        help="path prefix for your CSV files"
    )
    args = parser.parse_args()

    setup = args.setup
    m = args.m
    data_path = args.data_path

    encoder = sentence_transformers.SentenceTransformer("all-MiniLM-L12-v2")
    gammas = np.logspace(-3, 3, 100)

    ### Data
    data_test = pd.read_csv(f"{data_path}test_retrieval_minilm_{m}_{setup}.csv")
    data_dev  = pd.read_csv(f"{data_path}dev_retrieval_minilm_{m}_{setup}.csv")
    columns   = [''] + [str(n) for n in extract_numbers(data_test.keys())]

    print("Encoding texts...")
    if 'mt' in setup:
        mt_e_test = [
            encoder.encode(data_test.loc[:, 'mt' + c], show_progress_bar=False, batch_size=256)
            for c in columns
        ]
        mt_e_dev = [
            encoder.encode(data_dev.loc[:, 'mt' + c],  show_progress_bar=False, batch_size=256)
            for c in columns
        ]
    if 'src' in setup:
        src_e_test = [
            encoder.encode(data_test.loc[:, 'src' + c], show_progress_bar=False, batch_size=256)
            for c in columns
        ]
        src_e_dev = [
            encoder.encode(data_dev.loc[:, 'src' + c],  show_progress_bar=False, batch_size=256)
            for c in columns
        ]

    # Embeddings
    if setup == "mt":
        E_test, E_dev = mt_e_test, mt_e_dev
    elif setup == "src":
        E_test, E_dev = src_e_test, src_e_dev
    elif setup == "srcAmt":
        E_test = [e1 + e2 for e1, e2 in zip(mt_e_test, src_e_test)]
        E_dev  = [e1 + e2 for e1, e2 in zip(mt_e_dev,  src_e_dev)]
    elif setup == "srcCmt":
        E_test = [np.hstack((e1, e2)) for e1, e2 in zip(mt_e_test, src_e_test)]
        E_dev  = [np.hstack((e1, e2)) for e1, e2 in zip(mt_e_dev,  src_e_dev)]
    else:
        raise ValueError("Unknown embedding key")

    E_test = [e / np.linalg.norm(e, axis=-1)[:, None] for e in E_test]
    E_dev  = [e / np.linalg.norm(e, axis=-1)[:, None] for e in E_dev]

    # Dissimilarities
    D_test = 1 - np.vstack([(E_test[0] * e).sum(1) for e in E_test[1:]]).T
    D_dev  = 1 - np.vstack([(E_dev[0]  * e).sum(1) for e in E_dev[1:]]).T

    # Scores
    S_test = np.vstack([data_test.loc[:, 'score' + c].to_numpy() for c in columns[1:]]).T
    S_dev  = np.vstack([data_dev.loc[:,  'score' + c].to_numpy() for c in columns[1:]]).T

    # Labels
    Y_test = data_test.loc[:, 'score' + columns[0]].to_numpy().squeeze()
    Y_dev  = data_dev.loc[:,  'score' + columns[0]].to_numpy().squeeze()

    ### Making predictions
    Y_hat = S_test.mean(-1)

    # find best gamma on dev
    corrs = []
    for gamma in gammas:
        W_dev = softmax_kernel(D_dev, gamma)
        corrs.append(scipy.stats.pearsonr((W_dev * S_dev).sum(-1), Y_dev)[0])
    gamma_opt = gammas[np.argmax(corrs)]

    W_test = softmax_kernel(D_test, gamma_opt)
    Y_hat_weighted = (W_test * S_test).sum(-1)

    ### Output
    output = {
        'pearson': {
            'simple_avg': abs(scipy.stats.pearsonr(Y_hat, Y_test)[0]),
            'weighted_avg': abs(scipy.stats.pearsonr(Y_hat_weighted, Y_test)[0])
        },
        'kendall': {
            'simple_avg': abs(scipy.stats.kendalltau(Y_hat, Y_test, variant="b")[0]),
            'weighted_avg': abs(scipy.stats.kendalltau(Y_hat_weighted, Y_test, variant="b")[0])
        },
        'meanerr': {
            'simple_avg': np.mean(np.abs(Y_hat - Y_test)),
            'weighted_avg': np.mean(np.abs(Y_hat_weighted - Y_test))
        }
    }

    print(output)
