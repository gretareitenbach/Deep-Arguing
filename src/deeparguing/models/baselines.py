from typing import Optional, override

import numpy as np
import torch
from numpy.typing import ArrayLike
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from deeparguing.feature_extractor.lstm import LSTMFeatureExtractor
from deeparguing.feature_extractor.mlp_extractor import MLPExtractor
from deeparguing.feature_extractor.resnet import Resnet32

from .model import Model

"""
Fit-Predict Baselines

"""


class FitPredictBaseline(Model):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.device = "cpu"

    @override
    def forward(self, input: ArrayLike):
        if isinstance(input, torch.Tensor):
            input = input.cpu().numpy()

        if hasattr(self.model, "predict_proba"):
            preds = self.model.predict_proba(input)
        else:
            preds = self.model.predict(input)

        return torch.tensor(preds)

    @override
    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_default: Optional[ArrayLike] = None,
        y_default: Optional[ArrayLike] = None,
        batch_size: Optional[int] = None,
    ):
        if isinstance(X_train, torch.Tensor):
            X_train = X_train.cpu().numpy()
        if isinstance(y_train, torch.Tensor):
            y_train = y_train.cpu().numpy()
            if y_train.ndim > 1 and y_train.shape[1] > 1:
                y_train = np.argmax(y_train, axis=1)

        self.model.fit(X_train, y_train)

    def to(self, device):
        self.device = device
        return self

    def train(self, mode: bool = True):
        pass

    def eval(self):
        pass


class LogisticRegressionBaseline(FitPredictBaseline):
    def __init__(self, **kwargs):
        super().__init__(model=LogisticRegression(**kwargs))


class DecisionTreeBaseline(FitPredictBaseline):
    def __init__(self, **kwargs):
        super().__init__(model=DecisionTreeClassifier(**kwargs))


class RandomForestBaseline(FitPredictBaseline):
    def __init__(self, **kwargs):
        super().__init__(model=RandomForestClassifier(**kwargs))

class KNNBaseline(FitPredictBaseline):
    def __init__(self, **kwargs):
        super().__init__(model=KNeighborsClassifier(**kwargs))


"""
Neural Network Baselines
"""


class NeuralNetworkBaseline(Model, torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        output_size: int,
        output_activation: torch.nn.Module | None = None,
        bias: bool = True,
        dropout: None | float = None,
        batch_norm: bool = False,
    ):
        super(NeuralNetworkBaseline, self).__init__()

        self.model = MLPExtractor(
            input_size,
            hidden_sizes,
            output_size,
            output_activation,
            bias,
            dropout,
            batch_norm,
        )

    @property
    def device(self):
        return next(self.parameters(), torch.tensor(0)).device

    @override
    def forward(self, input: ArrayLike):
        return self.model(input)

    @override
    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_default: Optional[ArrayLike] = None,
        y_default: Optional[ArrayLike] = None,
        batch_size: Optional[int] = None,
    ):
        pass


class ResnetBaseline(Model, torch.nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        weights_path: Optional[str] = None,
        freeze_weights: bool = False,
    ):
        super(ResnetBaseline, self).__init__()

        self.model = Resnet32(
            num_classes=num_classes,
            weights_path=weights_path,
            freeze_weights=freeze_weights
        )

    @property
    def device(self):
        return next(self.parameters(), torch.tensor(0)).device

    @override
    def forward(self, input: ArrayLike):
        return self.model(input, use_classification_head=True)

    @override
    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_default: Optional[ArrayLike] = None,
        y_default: Optional[ArrayLike] = None,
        batch_size: Optional[int] = None,
    ):
        pass


class LSTMBaseline(Model, torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_features: int = 2,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
        weights_path: Optional[str] = None,
        freeze_weights: bool = False,
    ):
        super(LSTMBaseline, self).__init__()
        self.model = LSTMFeatureExtractor(
            input_size=input_size,
            hidden_size=hidden_size,
            output_features=output_features,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            weights_path=weights_path,
            freeze_weights=freeze_weights,
        )

    @property
    def device(self):
        return next(self.parameters(), torch.tensor(0)).device

    @override
    def forward(self, input: ArrayLike):
        return self.model(input)

    @override
    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_default: Optional[ArrayLike] = None,
        y_default: Optional[ArrayLike] = None,
        batch_size: Optional[int] = None,
    ):
        pass
