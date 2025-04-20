import unittest
import pandas as pd
import torch
from unittest.mock import patch, MagicMock
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from v1 import label_plot, predict_plot, load_model_and_predict  # Replace 'v1' with the actual module name

class TestIndianMoviePlotClassifier(unittest.TestCase):
    def setUp(self):
        # Sample Indian movie plots for testing
        self.plot_major = "A brilliant software engineer in Bangalore develops a revolutionary app to fight corruption, but faces challenges from powerful politicians."  # Major stream (software engineer)
        self.plot_less_known = "A talented musician from a small village in Rajasthan rises to fame in Mumbai, overcoming societal barriers through his folk music."  # Less known career (musician)
        self.plot_unlabeled = "A man travels to Goa for a vacation and falls in love with a local artist."  # No keywords (unlabeled)
        
        # Mock tokenizer and model for prediction tests
        self.tokenizer = MagicMock()
        self.model = MagicMock()
        self.device = "cpu"  # Use CPU for testing to avoid GPU dependency

    def test_label_plot_major_stream(self):
        """Test label_plot for a plot with major stream keywords."""
        result = label_plot(self.plot_major)
        self.assertEqual(result, 0, "Expected label 0 for plot with 'software engineer'")

    def test_label_plot_less_known_career(self):
        """Test label_plot for a plot with less known career keywords."""
        result = label_plot(self.plot_less_known)
        self.assertEqual(result, 1, "Expected label 1 for plot with 'musician'")

    def test_label_plot_unlabeled(self):
        """Test label_plot for a plot with no relevant keywords."""
        result = label_plot(self.plot_unlabeled)
        self.assertEqual(result, -1, "Expected label -1 for plot with no relevant keywords")

    def test_label_plot_case_insensitivity(self):
        """Test label_plot for case insensitivity."""
        plot = "A SOFTWARE ENGINEER builds a startup in Chennai."
        result = label_plot(plot)
        self.assertEqual(result, 0, "Expected label 0 for plot with 'SOFTWARE ENGINEER' (case insensitive)")

    @patch("v1.AutoTokenizer")
    @patch("v1.AutoModelForSequenceClassification")
    def test_predict_plot_major_stream(self, mock_model, mock_tokenizer):
        """Test predict_plot for a major stream plot."""
        # Mock tokenizer
        mock_tokenizer.from_pretrained.return_value = self.tokenizer
        self.tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }

        # Mock model
        mock_model.from_pretrained.return_value = self.model
        self.model.to.return_value = self.model
        self.model.return_value.logits = torch.tensor([[2.0, -1.0]])  # High probability for class 0
        self.model.eval.return_value = None

        result = predict_plot(self.plot_major)
        self.assertEqual(result, "Major Stream", "Expected 'Major Stream' for plot with 'software engineer'")

    @patch("v1.AutoTokenizer")
    @patch("v1.AutoModelForSequenceClassification")
    def test_predict_plot_less_known_career(self, mock_model, mock_tokenizer):
        """Test predict_plot for a less known career plot."""
        # Mock tokenizer
        mock_tokenizer.from_pretrained.return_value = self.tokenizer
        self.tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }

        # Mock model
        mock_model.from_pretrained.return_value = self.model
        self.model.to.return_value = self.model
        self.model.return_value.logits = torch.tensor([[-1.0, 2.0]])  # High probability for class 1
        self.model.eval.return_value = None

        result = predict_plot(self.plot_less_known)
        self.assertEqual(result, "Less Known Career", "Expected 'Less Known Career' for plot with 'musician'")

    @patch("v1.AutoTokenizer")
    @patch("v1.AutoModelForSequenceClassification")
    def test_load_model_and_predict(self, mock_model, mock_tokenizer):
        """Test load_model_and_predict for a major stream plot."""
        # Mock tokenizer
        mock_tokenizer.from_pretrained.return_value = self.tokenizer
        self.tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }

        # Mock model
        mock_model.from_pretrained.return_value = self.model
        self.model.to.return_value = self.model
        self.model.return_value.logits = torch.tensor([[2.0, -1.0]])  # High probability for class 0
        self.model.eval.return_value = None

        result = load_model_and_predict(self.plot_major, model_path="./v1", device="cpu")
        self.assertEqual(result, "Major Stream", "Expected 'Major Stream' for plot with 'software engineer'")

    def test_label_plot_empty_string(self):
        """Test label_plot with an empty string."""
        result = label_plot("")
        self.assertEqual(result, -1, "Expected label -1 for empty plot string")

    def test_label_plot_none_input(self):
        """Test label_plot with None input."""
        result = label_plot(None)
        self.assertEqual(result, -1, "Expected label -1 for None plot input")

    @patch("v1.torch.cuda.is_available")
    def test_load_model_and_predict_device_selection(self, mock_cuda):
        """Test load_model_and_predict device selection."""
        mock_cuda.return_value = False
        result = load_model_and_predict(self.plot_major, model_path="./v1")
        self.assertTrue("cpu" in str(result), "Expected CPU device when CUDA is not available")

if __name__ == "__main__":
    unittest.main()