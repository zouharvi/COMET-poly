from typing import Any, Dict, List
import comet_poly
import comet_poly.retrieval
import numpy as np


def knn_polyic(
    data: List[Dict],
    data_kb: List[Dict],
    model: str | Any = "zouharvi/comet-poly-base-wmt25",
    K=5,
    weights = None,
    prevent_hardmatch=False,
    predict_kwargs: Dict = {},
    key="mt",
):
    """
    TODO documentation
    """
    data_retrieved = comet_poly.retrieval.retrieve_from_kb(
        data=data,
        data_kb=data_kb,
        k=K,
        prevent_hardmatch=prevent_hardmatch,
        key=key,
    )

    if weights is None:
        # one half is on the actual prediction
        weights = [1]+[1/K]*K
    
    srcmt_uniq = {
        (line["src"], line[f"mt"])
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
                l["score"]
                for l in lines
            ],
            weights=weights
        ))
        for line, lines in zip(data, data_retrieved)
    ]


"""
import comet_poly.knn_polyic
data = [
    {
        "src": "Iceberg lettuce got its name in the 1920s when it was shipped packed in ice to stay fresh.",
        "mt": "Eisbergsalat erhielt seinen Namen in den 1920er-Jahren, als er in Eis verpackt verschickt wurde, um frisch zu bleiben.",
    },
    {
        "src": "The banana plant is actually an herb, not a tree.",
        "mt": "Die Bananenpflanze ist tatsächlich ein Kraut und kein Baum.",
    },
]
data_kb = [
    {
        "src": "Tomatoes are technically berries because they develop from the ovary of a flower.",
        "mt": "Tomaten sind technisch gesehen Beeren, da sie sich aus dem Fruchtknoten einer Blüte entwickeln.",
        "score": 94,
    },
    {
        "src": "Apples float in water because they are made up of 25% air.",
        "mt": "Äpfel schwimmen im Wasser, weil sie zu 25 % aus Luft bestehen.",
        "score": 97,
    },
    {
        "src": "Carrots were originally purple before the orange variety became popular.",
        "mt": "Karotten waren ursprünglich lila, bevor die orangefarbene Sorte populär wurde.",
        "score": 95,
    },
    {
        "src": "Rice is a staple food for more than half of the world’s population.",
        "mt": "Reis ist ein Grundnahrungsmittel für mehr als die Hälfte der Weltbevölkerung.",
        "score": 96,
    },
    {
        "src": "The banana plant is actually an herb, not a tree.",
        "mt": "Die Bananenpflanze ist tatsächlich ein Kraut und kein Baum.",
        "score": 93,
    },
    {
        "src": "The Earth rotates once every 24 hours, causing day and night.",
        "mt": "Die Erde dreht sich alle 24 Stunden einmal, was Tag und Nacht verursacht.",
        "score": 97,
    },
    {
        "src": "Sharks do not have bones; their skeletons are made of cartilage.",
        "mt": "Haie haben keine Knochen; ihr Skelett besteht aus Knorpel.",
        "score": 95,
    },
    {
        "src": "The Eiffel Tower can grow by up to 15 cm in summer due to metal expansion.",
        "mt": "Der Eiffelturm kann sich im Sommer aufgrund der Ausdehnung des Metalls um bis zu 15 cm vergrößern.",
        "score": 94,
    },
    {
        "src": "Lightning is hotter than the surface of the sun.",
        "mt": "Blitze sind heißer als die Oberfläche der Sonne.",
        "score": 96,
    },
]

print(comet_poly.knn_polyic.knn_polyic(data=data, data_kb=data_kb, model="zouharvi/comet-poly-base-wmt25"))
"""