"""
Setup script for demo.

This script installs the demo package and its dependencies.
"""
from setuptools import find_packages, setup

setup(
    name="efficientdet",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "efficientdet=efficientdet.demo:main",
        ]
    },
    package_data={"efficientdet": ["models/*", "dog.jpg"]},
    description="tfLITE",
    include_package_data=True,
    install_requires=open("requirements.txt", "r", encoding="utf-8").readlines(),
)
