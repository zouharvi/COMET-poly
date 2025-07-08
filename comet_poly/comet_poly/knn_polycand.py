from typing import Dict, List, Any
import comet_poly
import comet_poly.retrieval
import numpy as np


def knn_polycand(
    data: List[Dict],
    model: str | Any = "zouharvi/comet-poly-base-wmt25",
    K : int | None = None,
    weights: List | None = None,
    predict_kwargs: Dict = {},
):
    """
    Data is a list of items that have "mt2", "mt3", ... "mtK+1" keys
    """
    
    if K is None:
        K = max([k for k in range(2, 12) if f"mt{k}" in data[0]])-1
    print(K)
    assert all([f"mt{k}" in line for line in data for k in range(2, K+2)]), "Data must have the same mt2, mt3, ... keys"

    if weights is None:
        # one half is on the actual prediction
        weights = [1]+[1/K]*K

    srcmt_uniq = {
        (line["src"], line[f"mt{k}"])
        for line in data
        for k in range(2, K+2)
    } | {
        (line["src"], line["mt"])
        for line in data
    }
    srcmt_uniq_data = [
        {"src": src, "mt": mt}
        for src, mt in srcmt_uniq
    ]

    if type(model) is str:
        try:
            model = comet_poly.load_from_checkpoint(model)
        except Exception:
            model = comet_poly.load_from_checkpoint(
                comet_poly.download_model(model))
    else:
        # assume that the passed arg is a COMET model
        pass

    model: comet_poly.models.PolyCandMetric

    srcmt2score = {
        srcmt: score
        for srcmt, score in zip(
            srcmt_uniq,
            model.predict(srcmt_uniq_data, **predict_kwargs).scores
        )
    }

    # just return an average
    return [
        float(np.average(
            [
                srcmt2score[(line["src"], line[f"mt"])]
            ] + [
                srcmt2score[(line["src"], line[f"mt{k}"])]
                for k in range(2, K+2)
            ],
            weights=weights
        ))
        for line in data
    ]


"""
import comet_poly.knn_polycand
data = [
    {
        "src": "Iceberg lettuce got its name in the 1920s when it was shipped packed in ice to stay fresh.",
        "mt": "Eisbergsalat erhielt seinen Namen in den 1920er-Jahren, als er in Eis verpackt verschickt wurde, um frisch zu bleiben.",
        "mt2": "Eisbergsalat bekam seinen Namen, weil man ihn mit Eis schickte, damit er frisch bleibt.",
    },
    {
        "src": "Goats have rectangular pupils, which give them a wide field of vision—up to 320 degrees!",
        "mt": "Kozy mají obdélníkové zornice, což jim umožňuje vidět skoro všude kolem sebe, aniž by musely otáčet hlavou.",
        "mt2": "Kozy mají obdélníkové zornice, které jim umožňují mít zorné pole až 320 stupňů.",
    },
    {
        "src": "This helps them spot predators from almost all directions without moving their heads.",
        "mt": "Điều này giúp chúng phát hiện kẻ săn mồi từ gần như mọi hướng mà không cần quay đầu.",
        "mt2": "Nhờ vậy, chúng có thể thấy kẻ săn mồi từ hầu hết mọi phía mà không phải xoay đầu.",
    }
]
print(comet_poly.knn_polycand.knn_polycand(data, "zouharvi/comet-poly-base-wmt25"))
"""