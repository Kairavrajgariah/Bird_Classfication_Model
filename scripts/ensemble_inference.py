import tensorflow as tf
import numpy as np
import sys
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load models
resnet_model = load_model("models/resnet_model.h5")
effnet_model = load_model("models/efficientnet_model.h5")

# Load class names from train dataset
train_dir = "dataset/train"
class_names = sorted(os.listdir(train_dir))

def preprocess_input_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)

    # Preprocess for both models
    resnet_input = tf.keras.applications.resnet.preprocess_input(img_array.copy())
    effnet_input = tf.keras.applications.efficientnet.preprocess_input(img_array.copy())

    resnet_input = np.expand_dims(resnet_input, axis=0)
    effnet_input = np.expand_dims(effnet_input, axis=0)

    return resnet_input, effnet_input

def ensemble_predict(img_path):
    resnet_input, effnet_input = preprocess_input_image(img_path)

    # Get predictions
    resnet_probs = resnet_model.predict(resnet_input, verbose=0)[0]
    effnet_probs = effnet_model.predict(effnet_input, verbose=0)[0]

    # Average probabilities
    avg_probs = (resnet_probs + effnet_probs) / 2.0
    pred_class_idx = np.argmax(avg_probs)
    confidence = np.max(avg_probs)
    pred_class = class_names[pred_class_idx]

    return pred_class, confidence

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ensemble_inference.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        sys.exit(1)

    pred_class, confidence = ensemble_predict(img_path)
    print(f"Predicted Bird Species: {pred_class} (Confidence: {confidence:.2f})")
