# Driver Drowsiness Detection Using Multimodal Ensemble

- Most existing driver drowsiness detectors rely heavily on eye-based features (EAR, blink rate) and completely fail when the driver wears sunglasses or has occluded eyes, which is a very common real-world scenario during daytime driving, leading to dangerous false negatives.
- Our system solves this by combining a classical geometric model (using mouth-based MAR & MOE that work even with sunglasses) with a high-accuracy three-branch CNN (face + left/right eyes).

## Dataset
- This project uses the ***YawDD*** dataset (Yawning Detection Dataset). The classes are **alert** and **drowsy**

## Drowsiness Detection Features (Facial Landmarks)

- **EAR (Eye Aspect Ratio)**: Measures eye opening. Lower values indicate closed or blinking eyes.  
  $$
  \text{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \cdot \|p_1 - p_4\|}
  $$  
  where $p_1 … p_6$ are the six eye corner landmarks.

- **MAR (Mouth Aspect Ratio)**: Measures how widely the mouth is open (yawning).  
  $$
  \text{MAR} = \frac{|v_1| + |v_2| + |v_3|}{2 \cdot |h|}
  $$  
  where $v_1, v_2, v_3$ are vertical distances inside the mouth and $h$ is the mouth width.

- **Eye Circularity**: Detects squinting by measuring how circular the eye contour is.  
  $$
  \text{Circularity} = \frac{4\pi \cdot \text{Area(eye contour)}}{\text{Perimeter(eye contour)}^2}
  $$  
  Values close to 1 → wide-open circular eye; lower values → squinted/closed.

- **MOE (Mouth-over-Eye Ratio)**: Highly robust drowsiness indicator, especially when eyes are occluded by sunglasses.  
  $$
  \text{MOE} = \frac{\text{MAR}}{\text{EAR}}
  $$  
  When EAR drops or becomes unreliable (e.g., sunglasses), MAR alone drives MOE high during yawning → strong drowsy signal even with fully covered eyes.


![EAR & MAR](images\ear_mar.png)



## CNN Architecture
- **Backbone:** A ResNet-18 backbone (pretrained on ImageNet) is used for the main face/scene branch to extract global facial features.
- **Eye branches:** Two lightweight CNN branches (one per eye) process tighter eye crops to capture local eye patterns under occlusion.
- **Fusion & classifier:** Features from the three branches are concatenated and passed through fully-connected layers with dropout and a final sigmoid/logit for binary Alert vs Drowsy prediction.


## Folder Structure
```
Driver-Drowsiness-Detection/
├── cnn_resnet18.ipynb              # CNN training 
├── DataPreprocess_FeatureExtract.ipynb  # Dataset extraction & classical feature engineering
├── ensemble_model.ipynb            # Real-time ensemble inference
├── model_evaluation.ipynb          # Evaluation scripts & analysis
├── models/
│   └── drowsiness-97_10.pth        # Trained CNN checkpoint (committed)
│   └── best_drowsiness_model.pkl   # Saved KNN model
│   └── feature_scaler.pkl    
├── README.md                       
```

## Quick Setup & Usage
1. Create a Python environment and install required packages. 

```bash
pip install torch torchvision opencv-python scikit-learn dlib matplotlib numpy
```

2. Open the notebooks in Jupyter / VS Code and run the cells in order:
- `DataPreprocess_FeatureExtract.ipynb` — prepare dataset, extract classical features.
- `cnn_resnet18.ipynb` — training and architecture notes for the CNN branch.
- `model_evaluation.ipynb` — evaluate model performance on held-out data.
- `ensemble_model.ipynb` — demo notebook that runs the real-time ensemble (webcam). 

## Results Summary
- KNN (classical features): ***~72%*** test accuracy on the processed dataset (features extracted from frames like EAR, MAR, Circularity, MOE).
- CNN (ResNet18-based): ***~97%*** accuracy on visible-eyes data.


## Demo Output 
![Output](images\output_ss.png)




Built by: Esha Bidkar, Carol Chopde, Prachi Chavhan, Niharika Hariharan