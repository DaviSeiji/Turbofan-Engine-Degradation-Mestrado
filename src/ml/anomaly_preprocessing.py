"""Compatibilidade com os preprocessadores joblib salvos antes da reorganização.

Pickles antigos referenciam src.ml.anomaly_preprocessing.NominalPreprocessor.
A implementação e os novos imports ficam em src.anomaly_detection.preprocessing.
Não remover esta ponte enquanto esses artefatos forem utilizados.
"""
from src.anomaly_detection.preprocessing import NominalPreprocessor

__all__ = ['NominalPreprocessor']
