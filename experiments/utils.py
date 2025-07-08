# %%

def get_data(data_name="wmt"):
    """Returns (train, test)"""
    import subset2evaluate.utils
    import numpy as np

    if data_name == "wmt":
        data = subset2evaluate.utils.load_data_wmt_all(min_items=10, normalize=False, zero_bad=False, include_ref=True)
        for data_name, data_v in data.items():
            scores = [tgt["human"] for line in data_v for tgt in line["scores"].values()]
            # if the average is below zero then it has to be MQM
            is_mqm = np.average(scores) < 0
            for line in data_v:
                line["langs"] = "/".join(data_name)
                # normalize the score
                if is_mqm:
                    for sys in line["scores"].keys():
                        line["scores"][sys]["human"] = 100 + line["scores"][sys]["human"]
        data_train = [line for data_name, data_v in data.items() for line in data_v if data_name[0] != "wmt24"]
        data_test = [line for data_name, data_v in data.items() for line in data_v if data_name[0] == "wmt24"]
    elif data_name == "bio":
        data_train = subset2evaluate.utils.load_data_biomqm(split='dev', normalize=False)
        data_test = subset2evaluate.utils.load_data_biomqm(split='test', normalize=False)

        for data_name, data_v in data_train.items():
            for line in data_v:
                line["langs"] = "/".join(data_name)
        data_train = [line for data_v in data_train.values() for line in data_v]

        for data_name, data_v in data_test.items():
            for line in data_v:
                line["langs"] = "/".join(data_name)
        data_test = [line for data_v in data_test.values() for line in data_v]

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
    out_meanerr = []
    for score_v in score.values():
        score_v = np.array(score_v)
        # take abs because sometimes the direction can be flipped 
        out_pearson.append(abs(scipy.stats.pearsonr(score_v[:, 0], score_v[:, 1])[0]))
        out_kendall.append(abs(scipy.stats.kendalltau(score_v[:, 0], score_v[:, 1], variant="b")[0]))
        out_meanerr.append(np.mean(np.abs(score_v[:, 0] - score_v[:, 1])))

    return np.average(out_pearson), np.average(out_kendall), np.average(out_meanerr)
