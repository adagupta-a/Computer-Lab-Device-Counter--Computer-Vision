# Computer Lab Device Counter

A computer vision project that detects and counts computer laboratory devices using a custom-trained YOLOv8 model.

## Features

- Detects CPUs
- Detects monitors
- Detects keyboards
- Detects mice
- Displays bounding boxes around detected devices
- Shows individual device counts
- Interactive Streamlit interface

## Technologies Used

- Python
- YOLOv8 (Ultralytics)
- OpenCV
- Streamlit
- NumPy

## Dataset

Trained using the LabEquipVis computer laboratory image dataset, containing 2,584 annotated images across 10 classes.

## Installation

1. Clone the repository:

   git clone YOUR_REPOSITORY_URL

2. Open the project folder:

   cd Computer-Lab-Device-Counter

3. Create a virtual environment:

   python -m venv venv

4. Activate it:

   source venv/bin/activate

5. Install dependencies:

   pip install -r requirements.txt

6. Run the application:

   python -m streamlit run app.py

## Model

The project uses a custom-trained YOLOv8 model saved as `lab_device_best.pt`.

## Author

Ada Gupta