"""Setup file for financial_gan package."""

from setuptools import setup, find_packages

setup(
    name="financial_gan",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "wandb>=0.12.0",
        "pytest>=6.2.5",
    ],
    author="Lucija Gregov",
    author_email="lgregov@gmail.com",
    description="A Financial Time Series GAN implementation",
    keywords="deep-learning, gan, finance, time-series",
    python_requires=">=3.8",
)