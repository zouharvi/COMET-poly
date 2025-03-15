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