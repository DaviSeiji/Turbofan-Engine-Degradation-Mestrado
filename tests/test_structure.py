"""Checagens estruturais sem treinamento nem execução de experimentos."""
import importlib
import json
import unittest
from unittest.mock import patch

from src.common.paths import ROOT


class StructureTests(unittest.TestCase):
    def test_all_python_sources_compile(self):
        for base in ['src', 'tests']:
            for path in (ROOT / base).rglob('*.py'):
                with self.subTest(path=path.relative_to(ROOT)):
                    compile(path.read_text(encoding='utf-8'), str(path), 'exec')

    def test_main_imports_do_not_start_experiments(self):
        import optuna
        from tensorflow import keras
        modules = [
            'src.common.preprocessing.preproc_data_nasa',
            'src.rul_prediction.models.baseline_models',
            'src.rul_prediction.models.model_builders',
            'src.rul_prediction.models.transformer_layers',
            'src.rul_prediction.experiment_utils',
            'src.rul_prediction.training.tune_cnn',
            'src.rul_prediction.training.tune_hybrid',
            'src.rul_prediction.training.tune_transformer',
            'src.rul_prediction.evaluation.testar_modelos',
            'src.anomaly_detection.preprocessing',
            'src.anomaly_detection.models.autoencoder_models',
            'src.anomaly_detection.training.tune_autoencoder',
            'src.anomaly_detection.evaluation.detect_anomalies',
            'src.anomaly_detection.evaluation.evaluate_anomaly_detection',
            'src.xai.xai_baseExplainer', 'src.xai.xai_utils',
            'src.xai.explainers.xai_shapExplainer',
            'src.xai.explainers.xai_igExplainer',
            'src.xai.explainers.xai_saliencymapsExplainer',
            'src.xai.explainers.xai_limeExplainer',
            'src.xai.explainers.xaI_gradcamExplainer',
            'src.xai.explainers.xai_windowshapExplainer',
            'src.xai.evaluation.xai_estabilidade',
            'src.xai.evaluation.xai_estabilidade_modelos',
        ]
        with patch.object(keras.Model, 'fit', side_effect=AssertionError('Treinamento proibido no teste')), \
             patch.object(keras.Model, 'save', side_effect=AssertionError('Salvar modelo proibido no teste')), \
             patch.object(optuna, 'create_study', side_effect=AssertionError('Novo estudo proibido no teste')), \
             patch.object(optuna.study.Study, 'optimize', side_effect=AssertionError('Optuna proibido no teste')):
            for module in modules:
                with self.subTest(module=module):
                    importlib.import_module(module)

    def test_timeshap_import_when_dependency_available(self):
        if importlib.util.find_spec('timeshap') is None:
            self.skipTest('Dependência opcional timeshap não instalada.')
        importlib.import_module('src.xai.explainers.xai_timeshapExplainer')

    def test_result_and_artifact_paths(self):
        from src.anomaly_detection import preprocessing
        from src.anomaly_detection.evaluation import detect_anomalies, evaluate_anomaly_detection
        from src.rul_prediction.training import tune_transformer
        from src.xai.evaluation import xai_estabilidade_modelos
        self.assertEqual(preprocessing.ROOT, ROOT)
        self.assertEqual(preprocessing.PROCESSED_DIR, ROOT / 'data/processed_anomaly')
        self.assertEqual(detect_anomalies.RESULT_DIR, ROOT / 'results/anomaly_detection/detection')
        self.assertEqual(evaluate_anomaly_detection.SOURCE_DIR, detect_anomalies.RESULT_DIR)
        self.assertEqual(evaluate_anomaly_detection.OUTPUT_DIR, ROOT / 'results/anomaly_detection/evaluation')
        self.assertEqual(tune_transformer.PASTA_MODELOS, ROOT / 'models/best_models_transformer')
        self.assertEqual(xai_estabilidade_modelos.PASTA_RESULTADOS, ROOT / 'results/xai/model_stability')

    def test_notebook_code_syntax_and_local_import_paths(self):
        from IPython.core.inputtransformer2 import TransformerManager
        transform = TransformerManager()
        for path in (ROOT / 'notebooks').rglob('*.ipynb'):
            nb = json.loads(path.read_text(encoding='utf-8'))
            for index, cell in enumerate(nb['cells']):
                if cell['cell_type'] == 'code':
                    source = ''.join(cell['source'])
                    with self.subTest(notebook=path.name, cell=index):
                        compile(transform.transform_cell(source), f'{path}:cell{index}', 'exec')
                        self.assertNotIn('from src.ml.', source)
                        self.assertNotIn("sys.path.append('../src/ml')", source)


if __name__ == '__main__':
    unittest.main()
