# Object Detection with EfficientDet TFLite
This is a simple object detection demo that uses Tensorflow Lite. It takes an image as input and detects objects in it, drawing bounding boxes around them and labeling each object.

## Usage
- Create a virtual environment 
- Install the packages listed in the requirements.txt file: pip install -r requirements.txt
- Pick the model version in model_choice = 3 (1 - 4).
- Run the script with python demo.py
The script will create an output image with the same name as the input image but with "_result_tflite" appended to it.

## Displaying an Image

If you want to display an image on the screen, uncomment `show()` method in demo.py. However, if you are running this code in a headless environment, this method will not work.



To install this package from GitHub, you can use the following command: pip3 install git+https://github.com/neophoca/103_EfficientDet_lite_demo.git