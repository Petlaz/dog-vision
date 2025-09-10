# Dog Breed Identification Using Deep Learning and Transfer Learning
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rUh8ZLPylROONcYJtT-nW9xGwDfDNxcu?usp=sharing)

Dog Vision is a deep learning project designed to identify dog breeds from images. Using TensorFlow and transfer learning with MobileNetV2, the model predicts the breed of a dog with high accuracy. This project demonstrates practical skills in computer vision, deep learning, and deploying AI models.

## Project Structure

```
dog-vision/
│
├─ dog-vision.ipynb        # Main notebook with model training, evaluation, and predictions
├─ models/                 # Saved model files
├─ images/                 # Custom images for testing
├─ logs/                   # Training logs, TensorBoard logs, etc.
├─ my-dog-photos/          # Personal dog images for testing predictions
├─ test_predictions.csv    # CSV file storing predictions on test set
├─ test/                   # Test dataset images
├─ train/                  # Training dataset images
├─ labels.csv              # CSV file with image filenames and corresponding labels
├─ requirements.txt        # Python dependencies
└─ README.md               # Project overview
```

## Project Overview

- **Problem:** Identify the breed of a dog from an input image.
- **Solution:** A Convolutional Neural Network (CNN) using Transfer Learning (MobileNetV2) for feature extraction and a Dense classifier for predictions.
- **Dataset:** Images of various dog breeds (custom or standard datasets like golden_retriever, boxer, chow, pembroke, samoyed).
- **Frameworks:** TensorFlow, TensorFlow Hub, Keras, Matplotlib, NumPy.
- **Features:**
  - Train and evaluate the model on dog images.
  - Visualize predictions alongside images and confidence scores (top 10 predictions).
  - Custom image predictions (make inference on your own dog photos).
  - Efficient image preprocessing, resizing, and augmentation to improve model generalization.
  - Insightful visualizations of training metrics, accuracy, and prediction results.
  - The model and notebook are structured for easy extension to web or mobile applications.
  - Fully compatible with Google Colab for GPU acceleration.

## Technologies Used
- Python 3.x
- TensorFlow & Keras
- Tensorflow Hub
- NumPy, Pandas, Matplotlib, Seaborn
- Google Colab

## Dataset
The dataset consists of labeled images of different dog breeds. Images were preprocessed and augmented for robust training.

## Installation
1. Open the project in Google Colab using the badge above.
2. Clone the repository:
   ```bash
   git clone https://github.com/Petlaz/dog-vision.git
3. Install required packages:
   ```bash
   !pip install tensorflow numpy pandas matplotlib seaborn

## Visualizations

* The notebook includes:

* Image display with top 10 prediction probabilities.

* Highlighted predictions showing Pred: <breed> (value%) | Truth: <breed>.

* Batch visualization for multiple images.

## Results

- Achieved high accuracy on the validation set.

- Successfully classifies multiple dog breeds in real-time predictions.

## Future Work

* Deploy the model as a web or mobile application.

* Expand dataset to include more breeds.

* Integrate with Gradio or Streamlit for an interactive demo.

## License

MIT License
   
