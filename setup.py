from setuptools import setup, find_packages

setup(
    name="solar-forecasting-hk",
    version="0.1.0",
    description="Solar forecasting pipeline using the Gated Unified Model (GUM)",
    author="Your Name",
    packages=find_packages(),  # Automatically finds your 'src' folder as a package
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.23",
        "pandas>=1.5",
        "matplotlib>=3.7",
        "scikit-learn>=1.2",
        "pvlib>=0.10",
        "xarray>=2023.1", 
        "torch>=2.1",
        "torchvision>=0.16",
        "kornia>=0.7",
        "cdsapi"
    ],
)