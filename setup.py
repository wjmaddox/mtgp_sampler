from setuptools import find_packages, setup

setup(
    name="sampling_mtgps",
    version="0.0.1",
    packages=find_packages(),
    install_requires=["gpytorch", "botorch"],
)
