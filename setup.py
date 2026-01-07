from setuptools import setup, find_packages

setup(
    name="lmlayer",
    version="0.1.0",
    packages=find_packages(),  # Automatically find all packages (e.g., my_package)
    install_requires=open('requirements.txt').read().split('\n\n')[0].split(),
    author="ZisIsNotZis",
    author_email="ZisIsNotZis@Gmail.com",
    description="OpenAI-Compatible LLM Enhancement Layer",
    url="https://github.com/ZisIsNotZis/lmlayer",
)
