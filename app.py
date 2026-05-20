import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

app = Flask(__name__)

# --- CONFIGURATION ---
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16MB upload
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ⚠️ UPDATE THIS: Match the folder names from your training dataset (alphabetical order)
CLASS_NAMES = ["AMD","Cataract","Normal","Normal"]  # Example class names, update as needed
MODEL_PATH = "amdnet23_v3_1.tflite"

# --- LOAD TFLITE MODEL ---
print("Loading TFLite model...")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Model loaded successfully!")

# --- EXACT PREPROCESSING FROM TRAINING ---
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 1. CLAHE on L channel
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # 2. Bilateral filter
    img = cv2.bilateralFilter(img, 9, 75, 75)

    # 3. Unsharp mask
    blur = cv2.GaussianBlur(img, (0, 0), 3)
    img  = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
    img  = np.clip(img, 0, 255).astype(np.uint8)

    # 4. Gamma correction
    gamma = 1.2
    table = np.array([(i / 255.0) ** (1.0 / gamma) * 255
                      for i in np.arange(256)]).astype("uint8")
    img = cv2.LUT(img, table)

    # 5. Resize to 224x224
    img = cv2.resize(img, (224, 224))

    # 6. Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0

    # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
    return np.expand_dims(img, axis=0)

def get_prediction(image_path):
    # Preprocess
    input_data = preprocess_image(image_path)
    
    # Run Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # Get highest probability class
    predicted_idx = np.argmax(output_data)
        

    confidence = output_data[predicted_idx] * 100

    return CLASS_NAMES[predicted_idx], confidence

# --- FLASK ROUTES ---
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part in the request", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
        
    if file:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Predict
            label, confidence = get_prediction(filepath)
            
            # Pass to result template (using forward slashes for web paths)
            web_filepath = f"/{filepath}".replace('\\', '/') 
            
            return render_template('result.html', 
                                   image_path=web_filepath, 
                                   prediction=label, 
                                   confidence=f"{confidence:.2f}")
        except Exception as e:
            return f"An error occurred during processing: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
