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
MultitaskRankingMetric
========================
    TODO
"""
import os
import warnings
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch

from comet.models.base import CometModel
from comet.models.utils import Prediction, Target
from comet.modules import FeedForward


from torch import nn
from transformers.optimization import Adafactor, get_constant_schedule_with_warmup

from comet.models.base import CometModel
from comet.models.metrics import MultitaskMetrics
from comet.models.utils import Prediction

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset
import pytorch_lightning as ptl
from comet.models.predict_pbar import PredictProgressBar
from comet.models.predict_writer import CustomWriter
from comet.models.utils import (
    OrderedSampler,
    Prediction,
    Target,
    flatten_metadata,
    restore_list_order,
)
import numpy as np


class MultitaskRankingMetric(CometModel):
    """MultitaskRankingMetric:

    Args:
        nr_frozen_epochs (Union[float, int]): Number of epochs (% of epoch) that the
            encoder is frozen. Defaults to 0.9.
        keep_embeddings_frozen (bool): Keeps the encoder frozen during training. Defaults
            to True.
        optimizer (str): Optimizer used during training. Defaults to 'AdamW'.
        warmup_steps (int): Warmup steps for LR scheduler.
        encoder_learning_rate (float): Learning rate used to fine-tune the encoder model.
            Defaults to 3.0e-06.
        learning_rate (float): Learning rate used to fine-tune the top layers. Defaults
            to 3.0e-05.
        layerwise_decay (float): Learning rate % decay from top-to-bottom encoder layers.
            Defaults to 0.95.
        encoder_model (str): Encoder model to be used. Defaults to 'XLM-RoBERTa'.
        pretrained_model (str): Pretrained model from Hugging Face. Defaults to
            'microsoft/infoxlm-large'.
        pool (str): Type of sentence level pooling (options: 'max', 'cls', 'avg').
            Defaults to 'avg'
        layer (Union[str, int]): Encoder layer to be used for regression ('mix'
            for pooling info from all layers). Defaults to 'mix'.
        layer_transformation (str): Transformation applied when pooling info from all
            layers (options: 'softmax', 'sparsemax'). Defaults to 'sparsemax'.
        layer_norm (bool): Apply layer normalization. Defaults to 'False'.
        loss (str): Loss function to be used. Defaults to 'mse'.
        dropout (float): Dropout used in the top-layers. Defaults to 0.1.
        batch_size (int): Batch size used during training. Defaults to 4.
        train_data (Optional[List[str]]): List of paths to training data. Each file is
            loaded consecutively for each epoch. Defaults to None.
        validation_data (Optional[List[str]]): List of paths to validation data.
            Validation results are averaged across validation set. Defaults to None.
        hidden_sizes (List[int]): Hidden sizes for the Feed Forward regression.
        activations (str): Feed Forward activation function.
        final_activation (str): Feed Forward final activation.
        local_files_only (bool): Whether or not to only look at local files.
    """

    def __init__(
        self,
        nr_frozen_epochs: Union[float, int] = 0.3,
        keep_embeddings_frozen: bool = True,
        optimizer: str = "AdamW",
        warmup_steps: int = 0,
        encoder_learning_rate: float = 1e-06,
        learning_rate: float = 1.5e-05,
        layerwise_decay: float = 0.95,
        encoder_model: str = "XLM-RoBERTa",
        pretrained_model: str = "xlm-roberta-large",
        pool: str = "avg",
        layer: Union[str, int] = "mix",
        layer_transformation: str = "softmax",
        layer_norm: bool = True,
        loss: str = None,
        dropout: float = 0.1,
        batch_size: int = 4,
        train_data: Optional[List[str]] = None,
        validation_data: Optional[List[str]] = None,
        hidden_sizes: List[int] = [2048, 1024],
        activations: str = "Tanh",
        load_pretrained_weights: bool = True,
        local_files_only: bool = False,
    ) -> None:
        super(MultitaskRankingMetric, self).__init__(
            nr_frozen_epochs=nr_frozen_epochs,
            keep_embeddings_frozen=keep_embeddings_frozen,
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            encoder_learning_rate=encoder_learning_rate,
            learning_rate=learning_rate,
            layerwise_decay=layerwise_decay,
            encoder_model=encoder_model,
            pretrained_model=pretrained_model,
            pool=pool,
            layer=layer,
            layer_transformation=layer_transformation,
            layer_norm=layer_norm,
            loss=loss,
            dropout=dropout,
            batch_size=batch_size,
            train_data=train_data,
            validation_data=validation_data,
            class_identifier="multitask_ranking_metric",
            load_pretrained_weights=load_pretrained_weights,
            local_files_only=local_files_only,
        )
        self.save_hyperparameters()
        self.da_estimator = FeedForward(
            in_dim=self.encoder.output_units * (1 + 3),
            hidden_sizes=self.hparams.hidden_sizes,
            activations=self.hparams.activations,
            dropout=self.hparams.dropout,
            final_activation=None,
            out_dim=1,
        )
        self.pw_binary_estimator = FeedForward(
            in_dim=self.encoder.output_units * (1 + 3 + 3),
            hidden_sizes=self.hparams.hidden_sizes,
            activations=self.hparams.activations,
            dropout=self.hparams.dropout,
            final_activation="Sigmoid",
            out_dim=1,
        )

    @property
    def loss(self):
        """
        Returns a list of losses corresponding to the three tasks (da1, da2, pw)
        """
        return [torch.nn.MSELoss(), torch.nn.MSELoss(), torch.nn.BCELoss()]

    def compute_loss(self, predictions: list[Prediction], targets: list[Target]) -> torch.Tensor:
        """Computes Losses values between batch Predictions and respective Targets."""
        losses = []
        for loss, prediction, target in zip(self.loss, predictions, targets):
            losses.append(loss(prediction.score, target.score))
        return sum(losses)

    def requires_references(self) -> bool:
        return False

    def enable_context(self):
        if self.pool == "avg":
            self.use_context = True

    def prepare_sample(
        self, sample: List[Dict[str, Union[str, float]]], stage: str = "train"
    ) -> Union[
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], Dict[str, torch.Tensor]
    ]:
        """This method will be called by dataloaders to prepared data to input to the
        model.

        Args:
            sample (List[dict]): Batch of train/val/test samples.
            stage (str): model stage (options: 'fit', 'validate', 'test', or
                'predict'). Defaults to 'fit'.

        Returns:
            Model inputs and depending on the 'stage' training labels/targets.
        """
        inputs = {k: [str(dic[k]) for dic in sample] for k in sample[0] if k != "score"}
        src_inputs = self.encoder.prepare_sample(inputs["src"])
        mt1_inputs = self.encoder.prepare_sample(inputs["mt1"])
        mt2_inputs = self.encoder.prepare_sample(inputs["mt2"])

        src_inputs = {"src_" + k: v for k, v in src_inputs.items()}
        mt1_inputs = {"mt1_" + k: v for k, v in mt1_inputs.items()}
        mt2_inputs = {"mt2_" + k: v for k, v in mt2_inputs.items()}
        model_inputs = {**src_inputs, **mt1_inputs, **mt2_inputs}

        if stage == "predict":
            return model_inputs

        scores_da1 = [float(s["score_da1"]) for s in sample]
        scores_da2 = [float(s["score_da2"]) for s in sample]
        scores_pw = [float(s["score_pw"]) for s in sample]
        targets = [Target(score=torch.tensor(scores_da1, dtype=torch.float)),
                   Target(score=torch.tensor(scores_da2, dtype=torch.float)),
                   Target(score=torch.tensor(scores_pw, dtype=torch.float))]

        if "system" in inputs:
            targets["system"] = inputs["system"]

        return model_inputs, targets

    def training_step(
        self,
        batch: Tuple[dict, Target],
        batch_idx: int,
    ) -> torch.Tensor:
        """Pytorch Lightning training step.

        Args:
            batch (Tuple[dict, Target]): The output of your `prepare_sample` method.
            batch_idx (int): Integer displaying which batch this is.

        Returns:
            [torch.Tensor] Loss value
        """
        batch_input, batch_targets = batch
        batch_predictions = self.forward(**batch_input)
        loss_value = self.compute_loss(batch_predictions, batch_targets)

        if (
            (
                # nr_frozen_epochs is actually number of steps
                self.nr_frozen_epochs >= 1 and
                batch_idx > self.nr_frozen_epochs
            ) or (
                self.nr_frozen_epochs < 1.0 and
                self.nr_frozen_epochs > 0.0 and
                batch_idx > self.first_epoch_total_steps * self.nr_frozen_epochs
            )
        ):
            self.unfreeze_encoder()
            self._frozen = False

        self.log(
            "train_loss",
            loss_value,
            on_step=True,
            on_epoch=True,
            batch_size=batch_targets[0].score.shape[0],
        )
        return loss_value

    def validation_step(
        self,
        batch: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
        batch_nb: int,
        dataloader_idx: int,
    ) -> None:
        """Pytorch Lightning validation step. Runs model and logs metrics.

        Args:
            batch (Tuple[dict, Target]): The output of your `prepare_sample` method.
            batch_idx (int): Integer displaying which batch this is.
        """
        batch_input, batch_targets = batch
        batch_predictions = self.forward(**batch_input)
        if dataloader_idx == 0:
            self.train_metrics.update(
                [prediction.score for prediction in batch_predictions],
                [target["score"] for target in batch_targets]
            )

        elif dataloader_idx > 0:
            self.val_metrics[dataloader_idx - 1].update(
                [prediction.score for prediction in batch_predictions],
                [target["score"] for target in batch_targets],
                batch_targets["system"] if "system" in batch_targets else None,
            )

    def predict_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Pytorch Lightning predict step.

        Args:
            batch (Tuple[dict, Target]): The output of your `prepare_sample` method.
            batch_idx (int): Integer displaying which batch this is.
            dataloader_idx (int): Integer displaying which dataloader this sample is
                coming from.

        Return:
            Prediction object

        """
        model_outputs = [Prediction(scores=x.score) for x in self(**batch)]
        if self.mc_dropout:
            mcd_outputs = torch.stack(
                [self(**batch).score for _ in range(self.mc_dropout)]
            )
            model_outputs["metadata"] = Prediction(
                mcd_scores=mcd_outputs.mean(dim=0),
                mcd_std=mcd_outputs.std(dim=0),
            )
        return model_outputs

    def forward(
        self,
        src_input_ids: torch.tensor,
        src_attention_mask: torch.tensor,
        mt1_input_ids: torch.tensor,
        mt1_attention_mask: torch.tensor,
        mt2_input_ids: torch.tensor,
        mt2_attention_mask: torch.tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """ReferencelessRegression model forward method.

        Args:
            src_input_ids [torch.tensor]: input ids from source sentences.
            src_attention_mask [torch.tensor]: Attention mask from source sentences.
            mt_input_ids [torch.tensor]: input ids from MT.
            mt_attention_mask [torch.tensor]: Attention mask from MT.

        Return:
            Prediction object with translation scores.
        """
        src_sentemb = self.get_sentence_embedding(src_input_ids, src_attention_mask)
        mt1_sentemb = self.get_sentence_embedding(mt1_input_ids, mt1_attention_mask)
        mt2_sentemb = self.get_sentence_embedding(mt2_input_ids, mt2_attention_mask)

        diff1_src = torch.abs(mt1_sentemb - src_sentemb)
        prod1_src = mt1_sentemb * src_sentemb
        diff2_src = torch.abs(mt2_sentemb - src_sentemb)
        prod2_src = mt2_sentemb * src_sentemb

        embedded_sequences_for_da1 = torch.cat(
            (
                src_sentemb,
                mt1_sentemb, prod1_src, diff1_src,
                ), dim=1
        )
        embedded_sequences_for_da2 = torch.cat(
            (
                src_sentemb,
                mt2_sentemb, prod2_src, diff2_src
            ), dim=1
        )
        embedded_sequences_for_pw_binary = torch.cat(
            (
                src_sentemb,
                mt1_sentemb, prod1_src, diff1_src,
                mt2_sentemb, prod2_src, diff2_src
            ), dim=1
        )
        return Prediction(score=self.da_estimator(embedded_sequences_for_da1).view(-1)),\
               Prediction(score=self.da_estimator(embedded_sequences_for_da2).view(-1)), \
               Prediction(score=self.pw_binary_estimator(embedded_sequences_for_pw_binary).view(-1))

    def read_training_data(self, path: str) -> List[dict]:
        """Method that reads the training data (a csv file) and returns a list of
        samples.

        Returns:
            List[dict]: List with input samples in the form of a dict
        """
        df = pd.read_csv(path)
        df = df[["src", "mt1", "mt2", "score_da1", "score_da2", "score_pw"]]
        df["src"] = df["src"].astype(str)
        df["mt1"] = df["mt1"].astype(str)
        df["mt2"] = df["mt2"].astype(str)
        df["score_da1"] = df["score_da1"].astype("float16")
        df["score_da2"] = df["score_da2"].astype("float16")
        df["score_pw"] = df["score_pw"].astype("float16")
        return df.to_dict("records")

    def read_validation_data(self, path: str) -> List[dict]:
        """Method that reads the validation data (a csv file) and returns a list of
        samples.

        Returns:
            List[dict]: List with input samples in the form of a dict
        """
        df = pd.read_csv(path)
        columns = ["src", "mt1", "mt2", "score_da1", "score_da2", "score_pw"]
        # If system in columns we will use this to calculate system-level accuracy
        if "system" in df.columns:
            columns.append("system")
            df["system"] = df["system"].astype(str)

        df = df[columns]
        df["score_da1"] = df["score_da1"].astype("float16")
        df["score_da2"] = df["score_da2"].astype("float16")
        df["score_pw"] = df["score_pw"].astype("float16")
        df["src"] = df["src"].astype(str)
        df["mt1"] = df["mt1"].astype(str)
        df["mt2"] = df["mt2"].astype(str)
        return df.to_dict("records")

    def validation_step(
        self,
        batch: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
        batch_nb: int,
        dataloader_idx: int,
    ) -> None:
        """Pytorch Lightning validation step. Runs model and logs metrics.

        Args:
            batch (Tuple[dict, Target]): The output of your `prepare_sample` method.
            batch_idx (int): Integer displaying which batch this is.
        """
        batch_input, batch_targets = batch
        batch_predictions = self.forward(**batch_input)
        if dataloader_idx == 0:
            self.train_metrics.update(
                [prediction.score for prediction in batch_predictions],
                [target.score for target in batch_targets],
            )

        elif dataloader_idx > 0:
            self.val_metrics[dataloader_idx - 1].update(
                [prediction.score for prediction in batch_predictions],
                [target.score for target in batch_targets],
                batch_targets["system"] if "system" in batch_targets else None,
            )

    def init_metrics(self):
        """Initializes train/validation metrics."""
        self.train_metrics = MultitaskMetrics(prefix="train")
        self.val_metrics = nn.ModuleList(
            [MultitaskMetrics(prefix=d) for d in self.hparams.validation_data]
        )

    def predict(
        self,
        samples: List[Dict[str, str]],
        batch_size: int = 16,
        gpus: int = 1,
        devices: Union[List[int], str, int] = None,
        mc_dropout: int = 0,
        progress_bar: bool = True,
        accelerator: str = "auto",
        num_workers: int = None,
        length_batching: bool = True,
    ) -> Prediction:
        """Method that receives a list of samples (dictionaries with translations,
        sources and/or references) and returns segment-level scores, system level score
        and any other metadata outputed by COMET models. If `mc_dropout` is set, it
        also returns for each segment score, a confidence value.

        Args:
            samples (List[Dict[str, str]]): List with dictionaries with source,
                translations and/or references.
            batch_size (int): Batch size used during inference. Defaults to 16
            devices (Optional[List[int]]): A sequence of device indices to be used.
                Default: None.
            mc_dropout (int): Number of inference steps to run using MCD. Defaults to 0
            progress_bar (bool): Flag that turns on and off the predict progress bar.
                Defaults to True
            accelarator (str): Pytorch Lightning accelerator (e.g: 'cpu', 'cuda', 'hpu'
                , 'ipu', 'mps', 'tpu'). Defaults to 'auto'
            num_workers (int): Number of workers to use when loading and preparing
                data. Defaults to None
            length_batching (bool): If set to true, reduces padding by sorting samples
                by sequence length. Defaults to True.

        Return:
            Prediction object with `scores`, `system_score` and any metadata returned
                by the model.
        """
        if mc_dropout > 0:
            self.set_mc_dropout(mc_dropout)

        if gpus > 0 and devices is not None:
            assert len(devices) == gpus, AssertionError(
                "List of devices must be same size as `gpus` or None if `gpus=0`"
            )
        elif gpus > 0:
            devices = gpus
        else: # gpu = 0
            devices = "auto"

        sampler = SequentialSampler(samples)
        if length_batching and gpus < 2:
            try:
                sort_ids = np.argsort([len(sample["src"]) for sample in samples])
            except KeyError:
                sort_ids = np.argsort([len(sample["ref"]) for sample in samples])
            sampler = OrderedSampler(sort_ids)

        # On Windows, only num_workers=0 is supported.
        is_windows = os.name == "nt"
        if num_workers is None:
            # Guideline for workers that typically works well.
            num_workers = 0 if is_windows else 2 * gpus
        elif is_windows and num_workers != 0:
            logger.warning(
                "Due to limits of multiprocessing on Windows, it is likely that setting num_workers > 0 will result"
                " in scores of 0. It is therefore recommended to set num_workers=0 or leave it to None (default)."
            )

        self.eval()
        dataloader = DataLoader(
            dataset=samples,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=self.prepare_for_inference,
            num_workers=num_workers,
            multiprocessing_context="fork" if torch.backends.mps.is_available() else None,
        )
        if gpus > 1:
            pred_writer = CustomWriter()
            callbacks = [
                pred_writer,
            ]
        else:
            callbacks = []

        if progress_bar:
            enable_progress_bar = True
            callbacks.append(PredictProgressBar())
        else:
            enable_progress_bar = False

        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=".*Consider increasing the value of the `num_workers` argument` .*",
        )
        trainer = ptl.Trainer(
            devices=devices,
            logger=False,
            callbacks=callbacks,
            accelerator=accelerator if gpus > 0 else "cpu",
            strategy="auto" if gpus < 2 else "ddp",
            enable_progress_bar=enable_progress_bar,
        )
        return_predictions = False if gpus > 1 else True
        predictions = trainer.predict(
            self, dataloaders=dataloader, return_predictions=return_predictions
        )
        if gpus > 1:
            torch.distributed.barrier()  # Waits for all processes to finish predict

        # If we are in the GLOBAL RANK we need to gather all predictions
        if gpus > 1 and trainer.is_global_zero:
            predictions = pred_writer.gather_all_predictions()
            # Delete Temp folder.
            pred_writer.cleanup()
            return predictions

        elif gpus > 1 and not trainer.is_global_zero:
            # If we are not in the GLOBAL RANK we will return None
            exit()

        outputs = []
        for i in range(3):  # 3 outputs from 3 tasks
            scores = torch.cat([pred[i]["scores"] for pred in predictions], dim=0).tolist()
            if "metadata" in predictions[0]:
                metadata = flatten_metadata([pred[i]["metadata"] for pred in predictions])
            else:
                metadata = []

            output = Prediction(scores=scores, system_score=sum(scores) / len(scores))

            # Restore order of samples!
            if length_batching and gpus < 2:
                output["scores"] = restore_list_order(scores, sort_ids)
                if metadata:
                    output["metadata"] = Prediction(
                        **{k: restore_list_order(v, sort_ids) for k, v in metadata.items()}
                    )
                outputs.append(output)
            else:
                # Add metadata to output
                if metadata:
                    output["metadata"] = metadata

                outputs.append(output)
        return outputs

    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LambdaLR]]:
        """Pytorch Lightning method to configure optimizers and schedulers."""
        layer_parameters = self.encoder.layerwise_lr(
            self.hparams.encoder_learning_rate, self.hparams.layerwise_decay
        )
        top_layers_parameters = [
            {"params": self.da_estimator.parameters(), "lr": self.hparams.learning_rate},
            {"params": self.pw_binary_estimator.parameters(), "lr": self.hparams.learning_rate}
        ]
        if self.layerwise_attention:
            layerwise_attn_params = [
                {
                    "params": self.layerwise_attention.parameters(),
                    "lr": self.hparams.learning_rate,
                }
            ]
            params = layer_parameters + top_layers_parameters + layerwise_attn_params
        else:
            params = layer_parameters + top_layers_parameters

        if self.hparams.optimizer == "Adafactor":
            optimizer = Adafactor(
                params,
                lr=self.hparams.learning_rate,
                relative_step=False,
                scale_parameter=False,
            )
        else:
            optimizer = torch.optim.AdamW(params, lr=self.hparams.learning_rate)

        # If warmup setps are not defined we don't need a scheduler.
        if self.hparams.warmup_steps < 2:
            return [optimizer], []

        scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
        )
        return [optimizer], [scheduler]
