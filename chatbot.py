# chatbot.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.chains import ConversationalRetrievalChain
from .rag_pipeline import RAGPipeline
from .retriever import CustomRetriever

class Chatbot:
    def __init__(self, model_name="microsoft/Phi-3-mini-4k-instruct", retriever=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.retriever = retriever or CustomRetriever()
        self.rag_pipeline = RAGPipeline(self.model, self.tokenizer, self.retriever)

    def ask(self, query):
        return self.rag_pipeline.generate_response(query)

    def fine_tune(self, dataset_path):
        from .fine_tuning import fine_tune_model
        self.model = fine_tune_model(self.model, dataset_path, self.tokenizer)
