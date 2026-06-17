import pickle
import unittest
from pathlib import Path

from data_utils import read_conllu
from features import SparseFeatureExtractor
from oracle import StaticArcEagerOracle
from parser import GreedyDecoder, parse_sentence
from perceptron import AveragedPerceptron
from transitions import ArcEagerSystem


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "experiments" / "m0_arc_eager_static_sparse" / "model.pkl"
LABELS_PATH = PROJECT_ROOT / "experiments" / "m0_arc_eager_static_sparse" / "model_labels.pkl"
DEV_PATH = PROJECT_ROOT / "data" / "fr_gsd-ud-dev.conllu"


@unittest.skipUnless(MODEL_PATH.exists() and LABELS_PATH.exists(), "Frozen M0 artifacts are required")
class RefactoredPipelineSmokeTests(unittest.TestCase):
    def test_refactored_arc_eager_pipeline_smoke(self):
        sentences = read_conllu(str(DEV_PATH))[:3]

        model = AveragedPerceptron()
        model.load(str(MODEL_PATH))
        with LABELS_PATH.open("rb") as handle:
            labels = pickle.load(handle)

        transition_system = ArcEagerSystem()
        feature_extractor = SparseFeatureExtractor()
        decoder = GreedyDecoder()
        oracle = StaticArcEagerOracle()

        for sentence in sentences:
            self.assertIsNotNone(transition_system.initial_config(sentence))
            self.assertTrue(len(oracle.gold_transitions(transition_system.initial_config(sentence))) >= 1)

            parsed, details = parse_sentence(
                sentence,
                model,
                labels,
                return_details=True,
                transition_system=transition_system,
                feature_extractor=feature_extractor,
                decoder=decoder,
            )
            self.assertTrue(details["terminal_reached"])
            for token in parsed[1:]:
                self.assertIsNotNone(token["head"])
                self.assertNotEqual(token["deprel"], "_")


if __name__ == "__main__":
    unittest.main()
