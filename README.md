# YOLO Personal protective Equipment(PPE) Detection


### 1. Introduction

YOLO PPE Detection is a machine learning project designed to detect personal protective equipment (PPE) on individuals within images. This project leverages two YOLOv8 models: one for detecting people and another for detecting specific PPE on cropped person images.

The goal is to facilitate automatic safety compliance monitoring by recognizing whether people in industrial environments are wearing the required PPE.

### 2. Installation

Install the dependencies using pip:

```bash
pip install ultralytics opencv-python tqdm
```
Install the dependencies using pip:

```bash
pip install ultralytics opencv-python tqdm
```
Then, clone the repository:

```bash
git clone https://github.com/rahulsharmavishwakarma/yolo-ppe-detection.git
cd yolo-ppe-detection
```
### 3. Usage
- Inference
The inference.py script runs the person detection model on full images and the PPE detection model on cropped person images. It takes the following arguments: input_dir, output_dir, person_det_model, and ppe_detection_model.

```bash
python inference.py --input_dir ./datasets/images --output_dir ./results --person_det_model ./models/person_model.pt --ppe_detection_model ./models/ppe_model.pt
```
### 4. Train
To train the YOLOv8 models for either person detection or PPE detection, use the train.py script. You need to pass the dataset directory, model configuration, and training hyperparameters.

```bash
python train.py --data ./datasets/ppe.yaml --epochs 50 --weights yolov8m.pt
```
### 5.Results
After inference, the output (images with bounding boxes and confidence scores) will be saved in the specified output_dir.

- Person Detection: Bounding boxes for each detected person in the full image.
- PPE Detection: Bounding boxes and class names for detected PPE on cropped person images.
