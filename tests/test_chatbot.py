import unittest
from my_chatbot.chatbot import Chatbot
from my_chatbot.retriever import CustomRetriever

class TestChatbot(unittest.TestCase):
    def setUp(self):
        self.retriever = CustomRetriever()
        self.retriever.add_documents(["Test document 1", "Test document 2"])
        self.chatbot = Chatbot(retriever=self.retriever)

    def test_ask(self):
        response = self.chatbot.ask("What is the topic of Test document 1?")
        self.assertIn("Test document", response)

    def test_fine_tune(self):
        # Assuming we have a mock dataset path
        mock_dataset_path = "path_to_mock_dataset.json"
        self.assertIsInstance(self.chatbot.fine_tune(mock_dataset_path), type(self.chatbot.model))

if __name__ == "__main__":
    unittest.main()
