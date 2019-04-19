# License Plate Recognition
This project deals with recognizing the license plate number of a vehicle using machine learning.

##Implementation
The characters on the license plate are extracted using OpenCV in Python. These extracted characters are fed to a model for recognizing them, which is based on Convolutional Neural Network (CNN).

##Dataset
The model has been trained on NIST Special Database 19

## Steps for execution
1. Download all the files in the folder
2. Execute the python file "image_upload.py" using command "python3 image_upload.py" on terminal or "python image_upload.py" on cmd
3. The above step will provide a GUI where in the license plate image can be uploaded.
4. The predicted license plate number will be displayed in the GUI.