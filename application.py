import os
import torch
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import torchvision.transforms as transforms

from src.models.simple_cnn import SimpleCNN
from src.models.logistic_regression import LogisticRegression
import src.config as cfg

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Load model function
def load_model(model_path, model_class, params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(**params).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


# Preprocessing function
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)


# Inference function
def run_inference(model, image_path):
    input_tensor = preprocess_image(image_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)

    return predicted_class.item()


# Route for homepage
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files or 'model' not in request.form:
            return "No file or model selected", 400

        file = request.files['file']
        model_name = request.form['model'].strip().lower()

        if file.filename == '' or model_name not in ['simple_cnn', 'logistic_regression']:
            return "Invalid input", 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        model_class = SimpleCNN if model_name == 'simple_cnn' else LogisticRegression
        model_dir = os.path.join(cfg.CHECKPOINT_DIR, model_name)
        model_path = os.path.join(model_dir, 'final_model.pth')

        model = load_model(model_path, model_class, cfg.models_config[model_name]['params'])
        prediction = run_inference(model, file_path)

        return jsonify({'filename': filename, 'prediction': prediction})

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
