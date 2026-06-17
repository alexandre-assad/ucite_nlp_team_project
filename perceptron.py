"""
perceptron.py — Averaged Perceptron Classifier

The perceptron is a simple linear classifier. For each possible transition
(class), it maintains a weight vector. To predict, it computes the dot
product of the weight vector with the feature vector for each class,
and picks the class with the highest score.

The "averaged" variant is crucial for good performance: instead of using
the final weights, we use the average of all weight vectors seen during
training. This acts as a regularizer and significantly improves accuracy.

Implementation trick (from Hal Daumé III / Freund & Schapire):
  Instead of storing all weight vectors, we keep a running total and
  a timestamp for each weight. The averaged weight is computed at the end.
"""

import pickle
import copy
from collections import defaultdict


class AveragedPerceptron:
    """
    An averaged perceptron classifier with sparse feature vectors.
    
    The weights are stored as nested dictionaries:
      weights[class_label][feature_name] = weight_value
    
    This is efficient because most features have zero weight for most classes.
    """

    def __init__(self):
        # weights[class][feature] = current weight
        self.weights = defaultdict(lambda: defaultdict(float))
        # _totals[class][feature] = cumulative sum of weights (for averaging)
        self._totals = defaultdict(lambda: defaultdict(float))
        # _timestamps[class][feature] = last time this weight was updated
        self._timestamps = defaultdict(lambda: defaultdict(int))
        # global step counter (incremented on each update)
        self._step = 0

    def score(self, features, class_label):
        """
        Compute the score for a given class label.
        
        The score is the dot product of the weight vector for this class
        with the feature vector. Since our features are binary (0 or 1),
        this simplifies to summing the weights of active features.
        """
        class_weights = self.weights[class_label]
        total = 0.0
        for feature, value in features.items():
            total += class_weights.get(feature, 0.0) * value
        return total

    def predict(self, features, valid_classes=None):
        """
        Predict the best class for the given features.
        
        If valid_classes is provided, only consider those classes.
        Otherwise, consider all classes that have any weights.
        
        Returns:
            (best_class, best_score)
        """
        if valid_classes is None:
            valid_classes = list(self.weights.keys())

        best_class = None
        best_score = float("-inf")

        for cls in valid_classes:
            s = self.score(features, cls)
            if s > best_score:
                best_score = s
                best_class = cls

        return best_class, best_score

    def update(self, truth, guess, features):
        """
        Update the weights based on a prediction error.
        
        If the guess is wrong (guess != truth), we:
          - Increase the weights for the correct class (truth)
          - Decrease the weights for the incorrect class (guess)
        
        This is the standard perceptron update rule.
        """
        self._step += 1

        if truth == guess:
            return  # nothing to update

        for feature in features:
            # Before updating, accumulate the current weight into totals
            # (this is the trick for computing the average efficiently)
            self._accumulate(truth, feature)
            self._accumulate(guess, feature)

            # Increase weights for the correct class
            self.weights[truth][feature] += 1.0
            # Decrease weights for the incorrect class
            self.weights[guess][feature] -= 1.0

    def _accumulate(self, class_label, feature):
        """
        Accumulate the current weight into the running total.
        
        We multiply the current weight by the number of steps since it
        was last updated, and add that to the total. This lets us compute
        the average at the end without storing all intermediate weight vectors.
        """
        elapsed = self._step - self._timestamps[class_label][feature]
        self._totals[class_label][feature] += elapsed * self.weights[class_label][feature]
        self._timestamps[class_label][feature] = self._step

    def average_weights(self):
        """
        Replace the current weights with their averaged versions.
        
        Call this ONCE after training is complete.
        After this, the model is ready for prediction.
        """
        if self._step == 0:
            return

        for cls in self.weights:
            for feat in self.weights[cls]:
                # accumulate any remaining steps
                self._accumulate(cls, feat)
                # compute the average
                self.weights[cls][feat] = self._totals[cls][feat] / self._step

    def copy(self):
        """Return a deep copy of the model for evaluation or checkpointing."""
        new_model = AveragedPerceptron()
        new_model.weights = copy.deepcopy(self.weights)
        new_model._totals = copy.deepcopy(self._totals)
        new_model._timestamps = copy.deepcopy(self._timestamps)
        new_model._step = self._step
        return new_model

    def save(self, filepath):
        """Save the model weights to a file using pickle."""
        # convert defaultdicts to regular dicts for pickling
        data = {
            "weights": {cls: dict(feats) for cls, feats in self.weights.items()},
            "step": self._step,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        """Load model weights from a pickle file."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.weights = defaultdict(lambda: defaultdict(float))
        for cls, feats in data["weights"].items():
            for feat, weight in feats.items():
                self.weights[cls][feat] = weight
        self._step = data["step"]
        print(f"Model loaded from {filepath}")
