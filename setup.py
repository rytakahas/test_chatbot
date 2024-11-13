from setuptools import setup, find_packages

setup(
    name="my_chatbot",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "langchain",
        "sentence-transformers",
        "faiss-cpu",
        "datasets",
    ],
)
