import unittest
import joblib
from score import score
import app


class TestScoreFunction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the trained machine learning model from a file
        cls.model = joblib.load("best_model.pkl")

    def test_smoke(self):
        """Does the function execute without errors?"""
        score("test text", self.model, 0.7)

    def test_output_format(self):
        """Is the output returned in the correct format?"""
        pred, prop = score("test text", self.model, 0.7)
        self.assertIsInstance(pred, bool)
        self.assertIsInstance(prop, float)

    def test_prediction_values(self):
        """Is the prediction result either 0 or 1?"""
        pred, _ = score("test text", self.model, 0.7)
        self.assertIn(pred, [0, 1])

    def test_propensity_range(self):
        """Is the propensity score between 0 and 1?"""
        _, prop = score("test text", self.model, 0.7)
        self.assertGreaterEqual(prop, 0)
        self.assertLessEqual(prop, 1)

    def test_threshold_zero(self):
        """When threshold is 0, the prediction must be 1"""
        pred, _ = score("test text", self.model, 0)
        self.assertEqual(pred, 1)

    def test_threshold_one(self):
        """When threshold is 1, the prediction must be 0"""
        pred, _ = score("test text", self.model, 1)

        self.assertEqual(pred, 0)

    def test_obvious_spam(self):
        """Spam text should return a prediction of 1"""
        pred, prob = score("Call FREEPHONE 8005420578 now!, to win bike", self.model, 0.7)
        print(prob)
        self.assertEqual(pred, True)

    def test_obvious_non_spam(self):
        """Non-spam text should return a prediction of 0"""
        pred, _ = score("Please tell me not all of my car keys are in your purse.", self.model, 0.7)
        self.assertEqual(pred, 0)


if __name__ == '__main__':
    unittest.main()