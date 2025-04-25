import os
import cv2
import numpy as np
# Flask Import
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
# Tensor Import
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
# OS import
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Create instances of app
app = Flask(__name__)

# Main Page
@app.route("/")
def index():
    return render_template('./index.html')

# Image Path 
### File store in static folder
train_folder = os.path.join(".", "static", "seg")
test_folder = os.path.join(".", "static", "seg_test")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

### Image choosed from seg_test folder will store in upload folder
app.config['UPLOAD_FOLDER'] = os.path.join(".", "static", "upload")

# Check type of files
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load model
model = load_model("best_model.keras")

# Mapping labels
label_map = {0: 'Cat', 1: 'Dog'}

# Process finding similarity and show results
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    # Get image file
    file = request.files['file']

    if file and allowed_file(file.filename):
        # Get name of image and store the image choosen to upload folder
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Make prediction
        try:
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = img_array / 255.0  # Normalize image
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension    
                
            # Predict the class of the image
            prediction = model.predict(img_array)
            predicted_classes = np.argmax(prediction, axis=1)     
            confidence = float(prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0])

            # Show results in HTML
            return render_template(
                    './result.html',
                    uploaded_image=filepath,
                    name=filename,
                    prediction=label_map[predicted_classes[0]],
                    confidence=f"{confidence*100:.2f}%"
                )
        
        except Exception as e:
            return render_template('index.html', error=str(e))
        
    return redirect(request.url)

# Run app
if __name__ == '__main__':
    app.run(debug=True)