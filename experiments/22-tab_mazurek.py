# %%
import json
import collections

data_raw = [json.loads(x) for x in open("../computed/mazurek.out", "r")]
data_raw.sort(key=lambda x: x["t"])

data = collections.defaultdict(list)
for line in data_raw:
    data[line["model"]].append(line)

def get_model_name(model):
    if "0t00" in model:
        return "$f(s, t){\\rightarrow}\\hat{y}$"
    elif "1t00" in model:
        return "$f(s, t, t_2){\\rightarrow}\\hat{y}$"
    elif "1t01" in model:
        return "$f(s, t, t_2){\\rightarrow}\\hat{y}, \\hat{y_{t_2}}$"
    elif "1t10" in model:
        return "$f(s, t, t_2, y_{t_2}){\\rightarrow}\\hat{y}$"

def format_cell(o, key):
    if key in {"pearson", "kendall"}:
        return f"{o[key]:.3f}"
    elif key == "meanerr":
        return f"{o[key]:.1f}"

for model, model_v in data.items():
    if "1t01" in model:
        continue

    print(
        get_model_name(model),
    )
    # make sure it's increasing
    assert sorted([x["t"] for x in model_v]) == [x["t"] for x in model_v]

    if "0t00" in model:
        # duplicate baseline
        model_v = model_v * 5
    for key in ["pearson", "kendall", "meanerr"]:
        print(
            "",
            *[format_cell(x, key) for x in model_v],
            sep=" & ",
        )
    if "0t00" in model:
        print("\\\\\\\\[-0.1em]")
    else:
        print("\\\\")
        # if "0t00" in model: