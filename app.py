from flask import Flask, request, jsonify
from PIL import Image
import io
import torch
from torchvision import transforms
import pickle
import requests
from flask_basicauth import BasicAuth
import time

app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = 'Jiyyo230723'
app.config['BASIC_AUTH_PASSWORD'] = 'Jiyyopass123'
basic_auth = BasicAuth(app)

device = 'cpu'
model_name = 'ImageClassificationModel_28-6-2024'
with open(f'model/{model_name}.pkl', 'rb') as f:    
    model = pickle.load(f)
model.to(device)
model.eval()

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
)

class_names = ['CT-Scan', 'MRI', 'Pic-Face', 'Pic-Mouth', 'Pic-Overall', 'Reports', 'UltraSound', 'X-Ray']

def prepare_image(img):
    img = transform(img).unsqueeze(0)
    return img

@app.route('/predict', methods=['POST'])
@basic_auth.required
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    try:
        start_time = time.time()
        img = Image.open(file.stream)
        img = prepare_image(img)


        with torch.no_grad():
            pred_prob = model(img)
            prob = torch.softmax(pred_prob, dim=1)
            max_prob, pred = torch.max(prob, 1)
            label = class_names[pred.item()]

        end_time = time.time()
        processing_time = end_time - start_time

        return jsonify({'label': label, 'probability': max_prob.item(), 'processing_time': processing_time, 'model_name': model_name})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_from_url', methods=['POST'])
@basic_auth.required
def predict_from_url():
    data = request.json
    if 'url' not in data:
        return jsonify({'error': 'No URL provided'}), 400

    url = data['url']
    try:
        start_time = time.time()
        response = requests.get(url)
        img = Image.open(io.BytesIO(response.content))
        img = prepare_image(img)


        with torch.no_grad():
            pred_prob = model(img)
            prob = torch.softmax(pred_prob, dim=1)
            max_prob, pred = torch.max(prob, 1)
            label = class_names[pred.item()]

        end_time = time.time()
        processing_time = end_time - start_time

        return jsonify({'label': label, 'probability': max_prob.item(), 'processing_time': processing_time, 'model_name': model_name})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
