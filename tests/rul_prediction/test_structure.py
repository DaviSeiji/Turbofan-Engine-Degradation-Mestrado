"""Valida interfaces reutilizadas e modelos salvos, sem fit nem save."""
import unittest
import numpy as np
from src.common.paths import ROOT


class RulStructureTests(unittest.TestCase):
    def test_shared_utilities_are_same_objects(self):
        from src.rul_prediction import experiment_utils
        from src.rul_prediction.evaluation import testar_modelos
        from src.xai.evaluation import xai_estabilidade_modelos
        for name in ['MODELOS', 'carregar_best_params', 'salvar_csv_atomico']:
            self.assertIs(getattr(testar_modelos, name), getattr(experiment_utils, name))
            self.assertIs(getattr(xai_estabilidade_modelos, name), getattr(experiment_utils, name))

    def test_metrics(self):
        from src.rul_prediction.evaluation.testar_modelos import calcular_metricas
        values = calcular_metricas([1., 2., 3.], [1., 3., 5.])
        self.assertEqual(values['MAE'], 1.)
        self.assertAlmostEqual(values['RMSE'], np.sqrt(5 / 3))

    def test_existing_keras_models_load(self):
        from tensorflow import keras
        from src.rul_prediction.models.transformer_layers import PositionalEmbedding
        self.assertIs(keras.utils.get_registered_object('RUL>PositionalEmbedding'), PositionalEmbedding)
        paths = [ROOT / 'models' / folder / filename for folder, filename in [
            ('best_models_cnn_lstm', 'best_cnn_lstm_FD001.keras'),
            ('best_models_transformer', 'best_transformer_FD001.keras'),
            ('best_models_autoencoder', 'best_autoencoder_FD001.keras'),
        ]]
        found = 0
        for path in paths:
            if not path.is_file():
                continue
            found += 1
            with self.subTest(model=path.name):
                model = keras.models.load_model(path, compile=False)
                self.assertEqual(len(model.input_shape), 3)
                self.assertEqual(model.input_shape[1], 30)
                if 'autoencoder' in path.name:
                    self.assertEqual(model.output_shape, model.input_shape)
                else:
                    self.assertEqual(model.output_shape[-1], 1)
                del model
                keras.backend.clear_session()
        if not found:
            self.skipTest('Modelos locais não disponíveis.')


if __name__ == '__main__':
    unittest.main()
