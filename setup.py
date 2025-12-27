from setuptools import setup, find_packages

setup(
    name="stockml",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "yfinance>=0.2.0",
        "newsapi-python>=0.2.7",
        "vaderSentiment>=3.3.2",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
    ],
    entry_points={
        "console_scripts": [
            "stockml=cli:main",
        ],
    },
    python_requires=">=3.8",
    author="Your Name",
    description="Stock analysis framework with technical, fundamental, and sentiment analysis",
)
