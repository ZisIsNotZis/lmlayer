from setuptools import setup, find_packages

setup(
    name="lmlayer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=open('requirements.txt').read().split('\n\n')[0].split(),
    author="ZisIsNotZis",
    author_email="ZisIsNotZis@Gmail.com",
    description="OpenAI-compatible proxy that adds safety, sessions, RAG, tools, quotas, and fair scheduling to any LLM server",
    url="https://github.com/ZisIsNotZis/lmlayer",
)
