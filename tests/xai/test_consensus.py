"""Consenso com explicadores sintéticos, sem calcular SHAP ou treinar modelos."""
import unittest
import numpy as np


class ConsensusTests(unittest.TestCase):
    def test_linear_top10_scores_and_top5_are_preserved(self):
        from src.xai.evaluation.xai_estabilidade import gerar_tabela_consenso

        class FakeExplainer:
            X_test = np.zeros((2, 3, 12))

            def explain_instance(self, index):
                return np.tile(np.arange(1, 13), (3, 1))

        explainer = FakeExplainer()
        features = [f's_{i}' for i in range(1, 13)]
        result = gerar_tabela_consenso(explainer, explainer, explainer, features, num_samples=2)
        self.assertEqual(result.index.tolist(), ['s_12', 's_11', 's_10', 's_9', 's_8'])
        self.assertEqual(result['Pontuação Final'].tolist(), [30, 27, 24, 21, 18])


if __name__ == '__main__':
    unittest.main()
