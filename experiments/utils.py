# %%

def get_data():
    """Returns (train, test)"""
    import subset2evaluate.utils

    data = subset2evaluate.utils.load_data_wmt_all(min_items=10, normalize=False, zero_bad=False, include_ref=True)
    for data_name, data_v in data.items():
        for line in data_v:
            line["langs"] = "/".join(data_name)
    data_train = [line for data_name, data_v in data.items() for line in data_v if data_name[0] != "wmt24"]
    data_test = [line for data_name, data_v in data.items() for line in data_v if data_name[0] == "wmt24"]

    return data_train, data_test


def eval_da_per_lang(y_pred, data):
    import collections
    import scipy.stats
    import numpy as np

    score = collections.defaultdict(list)
    for line, y_pred in zip(data, y_pred):
        score[line["langs"]].append((float(line["score"]), y_pred))

    out_pearson = []
    out_kendall = []
    for score_v in score.values():
        score_v = np.array(score_v)
        out_pearson.append(scipy.stats.pearsonr(score_v[:, 0], score_v[:, 1])[0])
        out_kendall.append(scipy.stats.kendalltau(score_v[:, 0], score_v[:, 1], variant="b")[0])

    return np.average(out_pearson), np.average(out_kendall)