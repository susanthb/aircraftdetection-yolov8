# AircraftDetection using YOLOv8 ✈️  

AircraftDetect is a YOLOv8-based computer vision model for detecting military aircraft in aerial imagery. It can identify different aircraft types in real time, aiding surveillance and reconnaissance efforts.

## 🎯 Features  

- Real-time military aircraft detection  
- Supports multiple aircraft classes (e.g., MQ-9, RQ-4)  
- Optimized for high accuracy with YOLOv8  
- Training and validation pipeline with automated data preprocessing  
- Uses Weights & Biases (WandB) for model tracking  

## 🛠️ Technology Stack  

- **Object Detection**: YOLOv8 (Ultralytics)  
- **Data Processing**: Pandas, NumPy, OpenCV  
- **Training & Logging**: Weights & Biases (WandB)  
- **Machine Learning Framework**: PyTorch  
- **Dataset Management**: Kaggle  

## 📋 Prerequisites  

- Python 3.7+  
- YOLOv8 (Ultralytics)  
- OpenCV  
- Pandas & NumPy  
- WandB Account for logging  

## ⚙️ Installation & Setup  

1. Clone the repository:
```
git clone https://github.com/susanthb/militaryaircraft-detection-yolov8.git

cd militaryaircraft-detection-yolov8
```

2. Install required dependencies
```
pip install ultralytics opencv-python-headless matplotlib pandas numpy wandb

```

3. Set up WandB API key:
```
import wandb
wandb.login(key="your_wandb_api_key")
```

4. Download and preprocess the dataset:

Obtain military aircraft dataset from Kaggle
Extract the dataset into dataset/
Run preprocessing script
     - `python preprocess.py`


5. Configure YOLO training parameters:

Modify config.yaml with dataset paths and model hyperparameters

## 🚀 Usage

Training the Model
To train the YOLOv8 model on the aircraft dataset, run:
```
python train.py --epochs 100 --batch 16 --imgsz 640

```

Running Inference
To detect aircraft in an image or video:
```
python detect.py --source test_image.jpg

```

## 💡 How It Works

The system operates in several key steps:

1.Data Preprocessing: Merges annotation files and converts labels to YOLO format
2.Model Training: Uses YOLOv8 with configured hyperparameters
3.Inference Pipeline: Detects aircraft in real-time using trained weights
4.Performance Logging: Tracks training metrics using WandB

## 🌐 Visualization & Logging

The system logs training progress and evaluation metrics via Weights & Biases (WandB).

Dashboard includes loss curves, precision-recall graphs, and confusion matrices.

## 🛠 Future Enhancements

Improve model accuracy with more diverse datasets
Optimize for edge deployment on drones
Integrate real-time alerting system
