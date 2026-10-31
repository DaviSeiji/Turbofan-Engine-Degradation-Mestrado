"""Carregamento opcional dos artefatos locais existentes, somente leitura."""
import unittest
import joblib

from src.common.paths import ROOT
from src.anomaly_detection.preprocessing import NominalPreprocessor
from src.ml.anomaly_preprocessing import NominalPreprocessor as LegacyPreprocessor


class ArtifactCompatibilityTests(unittest.TestCase):
    def test_legacy_pickle_alias(self):
        self.assertIs(LegacyPreprocessor, NominalPreprocessor)

    def test_existing_preprocessors_load(self):
        paths = sorted((ROOT / 'data/processed_anomaly').glob('FD*/preprocessor.joblib'))
        if not paths:
            self.skipTest('Pré-processadores locais não disponíveis.')
        for path in paths:
            with self.subTest(dataset=path.parent.name):
                processor = joblib.load(path)
                self.assertIsInstance(processor, NominalPreprocessor)
                self.assertTrue(processor.features)


if __name__ == '__main__':
    unittest.main()
