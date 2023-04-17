# EfficientDet Lite for Object Detection

This repository demonstrates a simple object detection application using the EfficientDet Lite model with TensorFlow Lite. The application takes an image as input, detects objects within it, and draws bounding boxes and labels around each object.

## Table of Contents

1. [Overview](#overview)
2. [Setup and Installation](#setup-and-installation)
3. [Running the Demo](#running-the-demo)
4. [Running Tests](#running-tests)
5. [Pre-commit Checks](#pre-commit-checks)
6. [Alternative Installation Methods](#alternative-installation-methods)
7. [Docker Usage](#docker-usage)

## Overview

This demo utilizes the TensorFlow Lite version of EfficientDet to perform object detection on images.

## Setup and Installation

Clone this repository:

\```bash
git clone --branch week3 https://github.com/neophoca/103_EfficientDet_lite_demo.git
\```

Install the `build` package and create a wheel file:

\```bash
python -m pip install build
python3 -m build .
pip install dist/efficientdet-0.1.0-py3-none-any.whl
\```

Activate the demo by running:

\```bash
demo
\```

## Running the Demo

Create and activate a virtual environment.

Install the requirements and run the `demo.py` script:

\```bash
python demo/demo.py
\```

This command will execute the `demo.py` script with a sample image and save the result image to the same directory.

## Running Tests

To run the unit tests, execute the following command:

\```bash
pytest .
\```

## Pre-commit Checks

To run pre-commit checks, first install `pre-commit`:

\```bash
pip install pre-commit
\```

Then, run the following command:

\```bash
pre-commit run --all-files
\```

## Alternative Installation Methods

To install the package directly from the GitHub repository:

\```bash
pip install git+https://github.com/neo/103_EfficientDet_lite_demo.git
\```

To start the demo, use the command `demo`.

## Docker Usage

To build the Docker image:

\```bash
docker build -t object-detection-demo .
\```


The object detection model is integrated with Streamlit and can be accessed at `localhost:` in your web browser.
