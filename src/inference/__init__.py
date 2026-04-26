"""Inference module for forest fire detection system."""

from .predictor import (
    FirePredictionSystem,
    PredictionInput,
    ImagePredictionInput,
    SensorPredictionInput,
    PredictionResult
)

__all__ = [
    'FirePredictionSystem',
    'PredictionInput',
    'ImagePredictionInput',
    'SensorPredictionInput',
    'PredictionResult'
]
