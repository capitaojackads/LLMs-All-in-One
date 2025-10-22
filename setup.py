"""
Setup configuration for LLMs-All-in-One package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="llm-interface",
    version="1.0.0",
    author="LLMs All-in-One Contributors",
    description="A unified interface for all major LLM providers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/capitaojackads/LLMs-All-in-One",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "python-dotenv>=1.0.0",
        "openai>=1.0.0",
        "anthropic>=0.34.0",
        "google-generativeai>=0.3.0",
        "cohere>=5.0.0",
        "mistralai>=1.0.0",
        "huggingface-hub>=0.20.0",
        "requests>=2.31.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
            "flake8>=6.0.0",
        ],
    },
)
