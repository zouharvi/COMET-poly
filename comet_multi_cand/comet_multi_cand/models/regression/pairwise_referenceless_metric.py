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
PairwiseReferencelessMetric
========================
    TODO
"""
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from comet_multi_cand.models.base import CometModel
from comet_multi_cand.models.utils import Prediction, Target
from comet_multi_cand.modules import FeedForward
from torch import nn
from transformers.optimization import Adafactor, get_constant_schedule_with_warmup
from comet_multi_cand.models.base import CometModel
from comet_multi_cand.models.metrics import PairRegressionMetric
from comet_multi_cand.models.utils import Prediction
from comet_multi_cand.models.utils import (
    Prediction,
    Target,
)


class PairwiseReferencelessMetric(CometModel):
    """PairwiseReferencelessMetric:

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
        super(PairwiseReferencelessMetric, self).__init__(
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
            class_identifier="pairwise_referenceless_metric",
            load_pretrained_weights=load_pretrained_weights,
            local_files_only=local_files_only,
        )
        self.save_hyperparameters()
        self.estimator = FeedForward(
            in_dim=self.encoder.output_units * (1 + 3 + 3),
            hidden_sizes=self.hparams.hidden_sizes,
            activations=self.hparams.activations,
            dropout=self.hparams.dropout,
            final_activation=None,
            out_dim=2,
        )

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
        scores = [(da1, da2) for da1,da2 in zip(scores_da1, scores_da2)]
        targets = Target(score=torch.tensor(scores, dtype=torch.float))

        if "system" in inputs:
            targets["system"] = inputs["system"]

        return model_inputs, targets

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

        embedded_sequences = torch.cat(
            (
                src_sentemb,
                mt1_sentemb, prod1_src, diff1_src,
                mt2_sentemb, prod2_src, diff2_src
            ), dim=1
        )
        return Prediction(score=self.estimator(embedded_sequences))

    def read_training_data(self, path: str) -> List[dict]:
        """Method that reads the training data (a csv file) and returns a list of
        samples.

        Returns:
            List[dict]: List with input samples in the form of a dict
        """
        df = pd.read_csv(path)
        df = df[["src", "mt1", "mt2", "score_da1", "score_da2"]]
        df["src"] = df["src"].astype(str)
        df["mt1"] = df["mt1"].astype(str)
        df["mt2"] = df["mt2"].astype(str)
        df["score_da1"] = df["score_da1"].astype("float16")
        df["score_da2"] = df["score_da2"].astype("float16")
        return df.to_dict("records")

    def read_validation_data(self, path: str) -> List[dict]:
        """Method that reads the validation data (a csv file) and returns a list of
        samples.

        Returns:
            List[dict]: List with input samples in the form of a dict
        """
        df = pd.read_csv(path)
        columns = ["src", "mt1", "mt2", "score_da1", "score_da2"]
        # If system in columns we will use this to calculate system-level accuracy
        if "system" in df.columns:
            columns.append("system")
            df["system"] = df["system"].astype(str)

        df = df[columns]
        df["score_da1"] = df["score_da1"].astype("float16")
        df["score_da2"] = df["score_da2"].astype("float16")
        df["src"] = df["src"].astype(str)
        df["mt1"] = df["mt1"].astype(str)
        df["mt2"] = df["mt2"].astype(str)
        return df.to_dict("records")

    def init_metrics(self):
        """Initializes train/validation metrics."""
        self.train_metrics = PairRegressionMetric(prefix="train")
        self.val_metrics = nn.ModuleList(
            [PairRegressionMetric(prefix=d) for d in self.hparams.validation_data]
        )


    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LambdaLR]]:
        """Pytorch Lightning method to configure optimizers and schedulers."""
        layer_parameters = self.encoder.layerwise_lr(
            self.hparams.encoder_learning_rate, self.hparams.layerwise_decay
        )
        top_layers_parameters = [
            {"params": self.estimator.parameters(), "lr": self.hparams.learning_rate},
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
