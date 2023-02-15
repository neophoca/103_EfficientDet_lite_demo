from setuptools import setup, find_packages

setup(
    name='demo',
    version='0.1.0',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'demo=demo.demo:main',
        ],
    },
    description="tfLITE",
    install_requires=open("requirements.txt").readlines()
)

