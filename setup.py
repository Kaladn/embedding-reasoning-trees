"""
Setup script for Cascade Tokenizer
Production-ready AI with embedded reasoning trees
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="cascade-tokenizer",
    version="1.0.0",
    author="Cascade AI Research",
    author_email="research@cascade-ai.com",
    description="Production-ready AI tokenizer with embedded 6-1-6 reasoning trees for guided generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cascade-ai/cascade-tokenizer",
    packages=find_packages(),
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "gpu": [
            "torch[cuda]>=1.12.0",
        ],
        "visualization": [
            "plotly>=5.9.0",
            "dash>=2.5.0",
            "networkx>=2.8.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "cascade-demo=cascade_demo:main",
        ],
    },
    include_package_data=True,
    keywords=[
        "artificial intelligence",
        "natural language processing",
        "tokenizer",
        "reasoning trees",
        "constraint-based generation",
        "guided sampling",
        "language models",
    ],
    project_urls={
        "Bug Reports": "https://github.com/cascade-ai/cascade-tokenizer/issues",
        "Source": "https://github.com/cascade-ai/cascade-tokenizer",
    },
)
