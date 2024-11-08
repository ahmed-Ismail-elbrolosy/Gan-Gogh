from flask import Flask, request, send_file, send_from_directory
import torch
from model import Generator
from PIL import Image
from io import BytesIO
import torchvision.transforms as transforms

app = Flask(__name__)

# Load the generators for Monet and Actual styles
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G_A2B = Generator(img_channels=3).to(device)
G_B2A = Generator(img_channels=3).to(device)

# Load weights
checkpoint_G_A2B = torch.load('genA.pth.tar', map_location=device)
checkpoint_G_B2A = torch.load('genM.pth.tar', map_location=device)

G_A2B.load_state_dict(checkpoint_G_A2B['state_dict'])
G_B2A.load_state_dict(checkpoint_G_B2A['state_dict'])

G_A2B.eval()
G_B2A.eval()

# Define transformations
def preprocess_image(image):
    transform = transforms.Compose([
        # transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(image).unsqueeze(0).to(device)

def postprocess(tensor):
    transform = transforms.Compose([
        transforms.Normalize((-1, -1, -1), (2, 2, 2)),
        transforms.ToPILImage()
    ])
    image = tensor.squeeze(0).cpu()
    return transform(image)

def translate_image(generator, input_tensor):
    with torch.no_grad():
        return generator(input_tensor)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/transform', methods=['POST'])
def transform():
    if 'image' not in request.files:
        return 'No image file provided', 400

    image = request.files['image']
    try:
        img = Image.open(image).convert('RGB')
    except Exception as e:
        return f"Error processing image: {e}", 400

    input_tensor = preprocess_image(img)
    translated_tensor = translate_image(G_B2A, input_tensor)
    output_image = postprocess(translated_tensor)

    img_byte_arr = BytesIO()
    output_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    print("Image processed and sent back")  # Add this line to log the processing
    return send_file(img_byte_arr, mimetype='image/png')

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)