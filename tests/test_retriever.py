import unittest
from my_chatbot.fine_tuning import fine_tune_model
from transformers import AutoModelForCausalLM, AutoTokenizer

class TestFineTuning(unittest.TestCase):
    def setUp(self):
        self.model_name = "microsoft/Phi-3-mini-4k-instruct"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.mock_dataset_path = "path_to_mock_dataset.json"

    def test_fine_tuning(self):
        fine_tuned_model = fine_tune_model(self.model, self.mock_dataset_path, self.tokenizer)
        self.assertIsNotNone(fine_tuned_model)

if __name__ == "__main__":
    unittest.main()
