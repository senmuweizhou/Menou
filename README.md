# YOLO-DeepSort

Efficient Real-time Vehicle Tracking via Lightweight YOLOv5 and DeepSORT Integration

**Setup prerequisites**

This project was developed and tested using the following software configurations:

**Software Environment**  
1.Operating System: Windows 11    
2.Python: 3.9.18  
3.PyTorch: 2.1.0  
4.TorchVision: 0.16.0  
5.CUDA Toolkit: 12.1  
6.cuDNN: 8.8.0.1  
7.Matplotlib: 3.8.0  
8.NumPy: 1.26.2    
9.SciPy: 1.11.4  


**Experiment Preparation**  
Configure YOLO Project YAML File  
Modify the dataset and training configuration in the corresponding .yaml file of the YOLO project.  
Set Up Tracking Model in YOLO_Tracking Project  
In the yolo_tracking project, edit the file:  
deep_sort_pytorch/deep_sort/deep/feature_extractor.py  
to configure the feature extraction network used by the Deep SORT tracking module.

**Training and Testing**  
Train YOLO Detection Model  
In the YOLO project, open train.py, specify the YAML configuration file and dataset paths, then run the training script:  
Train and Test the Re-Identification Model for Tracking  
In the yolo_tracking project:  
Use train.py to train the Deep SORT re-identification model on your preprocessed dataset.  
Then, run track.py and load the trained YOLO detection weights to perform object tracking:  


