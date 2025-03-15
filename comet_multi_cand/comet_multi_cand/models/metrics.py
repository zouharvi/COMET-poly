# -*- coding: utf-8 -*-
# Copyright (C) 2020 Unbabel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""
Metrics
=======
    Regression and Ranking metrics to be used during training to measure
    correlations with human judgements
"""
from itertools import combinations
from typing import Any, Callable, List, Optional

import pandas as pd
import scipy.stats as stats
import torch
from torchmetrics import Metric
from torchmetrics.classification import MulticlassMatthewsCorrCoef


def system_accuracy(y_hat: List[float], y: List[float], system: List[str]) -> float:
    """Implementation of system-level accuracy proposed in
        [To Ship not to Ship](https://aclanthology.org/2021.wmt-1.57/)

    Args:
        y_hat (List[int]): List of metric scores
        y (List[int]): List of ground truth scores
        system (List[str]): List with the systems that produced a given translation.

    Return:
        Float: System-level accuracy.
    """
    try:
        data = pd.DataFrame({"y_hat": y_hat, "y": y, "system": system})
    except ValueError:
        raise Exception(
            "The program will be interrupted, followed by a series of errors."
            " This probably happens because you're using ddp strategy in the"
            " trainer config. System accuracy computation does not currently"
            " work with ddp. Please make sure your VALIDATION data DOES NOT"
            " include a 'system' column, and try again."
        )

    data = data.groupby("system").mean()
    pairs = list(combinations(data.index.tolist(), 2))

    tp = 0
    for system_a, system_b in pairs:
        human_delta = data.loc[system_a]["y"] - data.loc[system_b]["y"]
        model_delta = data.loc[system_a]["y_hat"] - data.loc[system_b]["y_hat"]
        if (human_delta >= 0) ^ (model_delta < 0):
            tp += 1

    accuracy = tp / len(pairs) if len(pairs) != 0 else 0
    return float(accuracy)


class MCCMetric(MulticlassMatthewsCorrCoef):
    def __init__(self, prefix: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self.prefix = prefix

    def compute(self) -> torch.Tensor:
        """Computes matthews correlation coefficient."""
        mcc = super(MCCMetric, self).compute()
        return {self.prefix + "_mcc": mcc}


class RegressionMetrics(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False
    preds: List[torch.Tensor]
    target: List[torch.Tensor]

    def __init__(
        self,
        prefix: str = "",
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")
        self.add_state("systems", default=[], dist_reduce_fx=None)
        self.prefix = prefix

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        systems: Optional[List[str]] = None,
    ) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds (torch.Tensor): Predictions from model
            target (torch.Tensor): Ground truth values
        """
        self.preds.append(preds)
        self.target.append(target)

        if systems:
            self.systems += systems

    def compute(self) -> torch.Tensor:
        """Computes spearmans correlation coefficient."""
        try:
            preds = torch.cat(self.preds, dim=0)
            target = torch.cat(self.target, dim=0)
        except TypeError:
            preds = self.preds
            target = self.target
        kendall, _ = stats.kendalltau(preds.tolist(), target.tolist())
        spearman, _ = stats.spearmanr(preds.tolist(), target.tolist())
        pearson, _ = stats.pearsonr(preds.tolist(), target.tolist())
        report = {
            self.prefix + "_kendall": kendall,
            self.prefix + "_spearman": spearman,
            self.prefix + "_pearson": pearson,
        }

        if len(self.systems) > 0:
            system_acc = system_accuracy(
                preds.cpu().tolist(), target.cpu().tolist(), self.systems
            )
            report["system_acc"] = system_acc

        return report


class WMTKendall(Metric):
    full_state_update = True

    def __init__(
        self,
        prefix: str = "",
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.add_state("concordance", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("discordance", default=torch.tensor(0), dist_reduce_fx="sum")
        self.prefix = prefix

    def update(self, distance_pos: torch.Tensor, distance_neg: torch.Tensor):
        assert distance_pos.shape == distance_neg.shape
        self.concordance += torch.sum((distance_pos < distance_neg)).to(
            self.concordance.device
        )
        self.discordance += torch.sum((distance_pos >= distance_neg)).to(
            self.discordance.device
        )

    def compute(self):
        return {
            self.prefix
            + "_kendall": (self.concordance - self.discordance)
            / (self.concordance + self.discordance)
        }


class PairwiseAccuracy(Metric):
    full_state_update = True

    def __init__(
        self,
        prefix: str = "",
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.add_state("accuracy", default=torch.tensor([]), dist_reduce_fx="cat")
        self.prefix = prefix

    def update(self, prediction: torch.Tensor, target: torch.Tensor, *args, **kwargs):
        assert prediction.shape == target.shape
        self.accuracy = torch.cat([self.accuracy, ((prediction > 0.5)*1.0 == target).to(self.accuracy.device)])

    def compute(self):
        return {
            self.prefix
            + "_accuracy": torch.mean(self.accuracy)
        }


class PairwiseDifferenceMSE(Metric):
    full_state_update = True

    def __init__(
        self,
        prefix: str = "",
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.add_state("mse", default=torch.tensor([]), dist_reduce_fx="cat")
        self.prefix = prefix

    def update(self, prediction: torch.Tensor, target: torch.Tensor, *args, **kwargs):
        assert prediction.shape == target.shape
        self.mse = torch.cat([self.mse, ((prediction-target)**2).to(self.mse.device)])

    def compute(self):
        return {
            self.prefix
            + "_mse": torch.mean(self.mse)
        }


class MultitaskMetrics(Metric):
    full_state_update = True

    def __init__(
            self,
            prefix: str = "",
            dist_sync_on_step: bool = False,
            process_group: Optional[Any] = None,
            dist_sync_fn: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.prefix = prefix

        self.da_corr1 = RegressionMetrics(prefix, dist_sync_on_step, process_group, dist_sync_fn)
        self.da_corr2 = RegressionMetrics(prefix, dist_sync_on_step, process_group, dist_sync_fn)
        self.pw_acc = PairwiseAccuracy(prefix, dist_sync_on_step, process_group, dist_sync_fn)

    def update(self, predictions: torch.Tensor, targets: torch.Tensor, *args, **kwargs):
        self.da_corr1.update(predictions[0], targets[0])
        self.da_corr2.update(predictions[1], targets[1])
        self.pw_acc.update(predictions[2], targets[2], args, kwargs)

    def compute(self):
        da_corr1_result = self.da_corr1.compute()
        da_corr1_result = {f"{key}1": value for key, value in da_corr1_result.items()}
        da_corr2_result = self.da_corr2.compute()
        da_corr2_result = {f"{key}2": value for key, value in da_corr2_result.items()}
        pw_acc_result = self.pw_acc.compute()
        final_output = {**da_corr1_result, **da_corr2_result, **pw_acc_result}
        final_output[f"{self.prefix}_avg"] = sum(list(final_output.values())) / len(final_output)
        return final_output


class PairRegressionMetric(Metric):
    full_state_update = True

    def __init__(
            self,
            prefix: str = "",
            dist_sync_on_step: bool = False,
            process_group: Optional[Any] = None,
            dist_sync_fn: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.prefix = prefix

        self.regression_metrics = RegressionMetrics(prefix, dist_sync_on_step, process_group, dist_sync_fn)

    def update(self, predictions: torch.Tensor, targets: torch.Tensor, systems: Optional[List[str]] = None):
        self.regression_metrics.update(predictions.view(-1), targets.view(-1), systems)

    def compute(self):
        return self.regression_metrics.compute()


class MultiCandMetrics(RegressionMetrics):
    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        systems: Optional[List[str]] = None,
    ) -> None:  # type: ignore
        # just take the first element (prediction for main translation) and fall back with other functions
        super().update(preds[:, 0], target[:, 0], systems)