from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch

from comet_multi_cand.models.regression.regression_metric import RegressionMetric
from comet_multi_cand.models.utils import Prediction, Target
from comet_multi_cand.modules import FeedForward

class MultiCandMetric(RegressionMetric):
    """MultiCandMetric:

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
        loss: str = "mse",
        dropout: float = 0.1,
        batch_size: int = 4,
        train_data: Optional[List[str]] = None,
        validation_data: Optional[List[str]] = None,
        hidden_sizes: List[int] = [2048, 1024],
        activations: str = "Tanh",
        final_activation: Optional[str] = None,
        load_pretrained_weights: bool = True,
        local_files_only: bool = False,
        additional_score_in: List[bool] = [False, False, False, False, False],
        additional_score_out: List[bool] = [False, False, False, False, False],
        additional_translation_in: List[bool] = [False, False, False, False, False],
    ) -> None:
        super(RegressionMetric, self).__init__(
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
            class_identifier="multicand_metric",
            load_pretrained_weights=load_pretrained_weights,
            local_files_only=local_files_only,
        )
        self.additional_score_in = additional_score_in
        self.additional_score_out = additional_score_out
        self.additional_translation_in = additional_translation_in

        self.save_hyperparameters()
        self.estimator = FeedForward(
            in_dim=self.encoder.output_units * (4 + 5 * sum(additional_translation_in)) + 1 * (sum(additional_score_in)),
            hidden_sizes=hidden_sizes,
            activations=activations,
            dropout=dropout,
            final_activation=final_activation,
            out_dim=1 + sum(additional_score_out),
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
        mt1_inputs = self.encoder.prepare_sample(inputs["mt"])

        src_inputs = {"src_" + k: v for k, v in src_inputs.items()}
        mt1_inputs = {"mt1_" + k: v for k, v in mt1_inputs.items()}

        # embed additional translations
        additional_translation_in = []
        for i, flag in enumerate(self.additional_translation_in):
            if not flag:
                continue
            obj = self.encoder.prepare_sample(inputs[f"mt{i+2}"])
            additional_translation_in.append((obj["input_ids"], obj["attention_mask"]))

        additional_score_in = torch.tensor([
            # start at i=0 which corresponds to score2
            [float(s[f"score{i+2}"]) for i, flag in enumerate(self.additional_score_in) if flag]
            for s in sample
        ], dtype=torch.float)

        model_inputs = {**src_inputs, **mt1_inputs, "additional_translation_in": additional_translation_in, "additional_score_in": additional_score_in}

        if stage == "predict":
            return model_inputs
        
        targets = Target(score=torch.tensor([
            [float(s["score"])] + [float(s[f"score{i+2}"]) for i, flag in enumerate(self.additional_score_out) if flag]
            for s in sample
        ], dtype=torch.float))

        if "system" in inputs:
            targets["system"] = inputs["system"]

        return model_inputs, targets

    def forward(
        self,
        src_input_ids: torch.tensor,
        src_attention_mask: torch.tensor,
        mt1_input_ids: torch.tensor,
        mt1_attention_mask: torch.tensor,
        additional_translation_in: List[Tuple[torch.tensor, torch.tensor]],
        additional_score_in: torch.tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """MultiCandMetric model forward method.

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
        additional_sentembd = [self.get_sentence_embedding(*inp) for inp in additional_translation_in]

        embedded_sequences = torch.cat(
            (
                mt1_sentemb, src_sentemb,
                mt1_sentemb * src_sentemb, torch.abs(mt1_sentemb - src_sentemb),
            ), dim=1
        )

        # add additional translations in
        for mtX_sentembd in additional_sentembd:
            embedded_sequences = torch.cat(
                (
                    # previous
                    embedded_sequences,
                    mtX_sentembd,
                    mtX_sentembd * src_sentemb, torch.abs(mtX_sentembd - src_sentemb),
                    mtX_sentembd * mt1_sentemb, torch.abs(mtX_sentembd - mt1_sentemb),
                ), dim=1
            )

        # add additional scores in
        embedded_sequences = torch.cat(
            (embedded_sequences, additional_score_in.view(-1, sum(self.additional_score_in))),
            dim=1
        )
        return Prediction(score=self.estimator(embedded_sequences).view(-1))

    def read_training_data(self, path: str) -> List[dict]:
        """Method that reads the training data (a csv file) and returns a list of
        samples.

        Returns:
            List[dict]: List with input samples in the form of a dict
        """
        df = pd.read_csv(path)
        df = df[["src", "mt", "score", "mt2", "score2", "mt3", "score3", "mt4", "score4", "mt5", "score5", "mt6", "score6"]]
        df["src"] = df["src"].astype(str)
        df["mt"] = df["mt"].astype(str)
        df["score"] = df["score"].astype("float16")
        df["mt2"] = df["mt2"].astype(str)
        df["score2"] = df["score2"].astype("float16")
        df["mt3"] = df["mt3"].astype(str)
        df["score3"] = df["score3"].astype("float16")
        df["mt4"] = df["mt4"].astype(str)
        df["score4"] = df["score4"].astype("float16")
        df["mt5"] = df["mt5"].astype(str)
        df["score5"] = df["score5"].astype("float16")
        df["mt6"] = df["mt6"].astype(str)
        df["score6"] = df["score6"].astype("float16")

        return df.to_dict("records")

    def read_validation_data(self, path: str) -> List[dict]:
        """Method that reads the validation data (a csv file) and returns a list of
        samples.

        Returns:
            List[dict]: List with input samples in the form of a dict
        """
        df = pd.read_csv(path)
        columns = ["src", "mt", "score", "mt2", "score2", "mt3", "score3", "mt4", "score4", "mt5", "score5", "mt6", "score6"]
        # If system in columns we will use this to calculate system-level accuracy
        if "system" in df.columns:
            columns.append("system")
            df["system"] = df["system"].astype(str)

        df = df[columns]
        df["mt"] = df["mt"].astype(str)
        df["score"] = df["score"].astype("float16")
        df["mt2"] = df["mt2"].astype(str)
        df["score2"] = df["score2"].astype("float16")
        df["mt3"] = df["mt3"].astype(str)
        df["score3"] = df["score3"].astype("float16")
        df["mt4"] = df["mt4"].astype(str)
        df["score4"] = df["score4"].astype("float16")
        df["mt5"] = df["mt5"].astype(str)
        df["score5"] = df["score5"].astype("float16")
        df["mt6"] = df["mt6"].astype(str)
        df["score6"] = df["score6"].astype("float16")
        return df.to_dict("records")
