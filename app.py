from flask import Flask, request, send_file, send_from_directory, jsonify
import torch
from model import Generator
from PIL import Image
from io import BytesIO
import torchvision.transforms as transforms
import cv2
import os
import numpy as np
import warnings
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from datetime import datetime
from moviepy.editor import ImageSequenceClip

project_path: str = os.path.abspath(os.path.dirname(__file__))
# Suppress specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
app = Flask(__name__)

# Check for available GPUs
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"GPUs detected: {num_gpus}")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No GPUs detected, using CPU")

# Load the generators for Monet and Actual styles
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G_B2A = Generator(img_channels=3).to(device)

# Use DataParallel to use multiple GPUs
if torch.cuda.device_count() > 1:
    G_B2A = torch.nn.DataParallel(G_B2A)

checkpoint_G_B2A = torch.load('genM.pth.tar', map_location=device)

G_B2A.load_state_dict(checkpoint_G_B2A['state_dict'])

G_B2A.eval()

# Preprocessing and postprocessing
preprocess = Compose([
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
    ToTensorV2()
])

def preprocess_image(image):
    image = np.array(image)
    image = preprocess(image=image)['image']
    return image.unsqueeze(0)  # Add batch dimension

def postprocess(tensor):
    tensor = tensor.squeeze(0)  # Remove batch dimension
    tensor = tensor * 0.5 + 0.5  # Denormalize
    array = tensor.permute(1, 2, 0).cpu().numpy() * 255  # Change to HWC and scale to [0, 255]
    array = array.astype('uint8')
    return array  # Return numpy array for OpenCV

def translate_image(generator, input_image):
    with torch.no_grad():
        translated_image = generator(input_image.to(device))
    return translated_image

def process_video(input_video_path, output_dir, model):
    # Create a directory with the current timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    output_video_path = os.path.join(output_dir, f'output_video_{timestamp}.mp4')
    input_video_copy_path = os.path.join(output_dir, f'input_video_{timestamp}.mp4')

    # Copy the input video to the output directory
    os.rename(input_video_path, input_video_copy_path)

    # Open the input video
    cap = cv2.VideoCapture(input_video_copy_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {input_video_copy_path}")
        return False, None

    # Get the FPS and frame size
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps == 0:
        print("Error: FPS value is zero")
        cap.release()
        return False, None

    print(f"Video FPS: {fps}, Width: {width}, Height: {height}")

    # Process each frame
    frame_count = 0
    processed_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit if no more frames
        
        # Convert frame to PIL Image for processing
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = preprocess_image(image)
        
        # Apply the model to get the translated frame
        translated_tensor = translate_image(model, input_tensor)
        output_frame = postprocess(translated_tensor)

        # Convert back to BGR for saving
        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
        processed_frames.append(output_frame)

        # Save the processed frame as an image file
        frame_filename = os.path.join(output_dir, f'frame_{frame_count:04d}.png')
        cv2.imwrite(frame_filename, output_frame)

        frame_count += 1
        print(f"Processed frame {frame_count}")
    
    # Release resources
    cap.release()

    if not processed_frames:
        print("Error: No frames processed")
        return False, None

    # Create a video from the processed frames using moviepy
    try:
        clip = ImageSequenceClip([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in processed_frames], fps=fps)
        clip.write_videofile(output_video_path, codec='libx264')
        print(f"Video saved successfully at {output_video_path}")
    except Exception as e:
        print(f"Error creating video: {e}")
        return False, None

    print(f"Processed frames saved in directory: {output_dir}")
    return True, output_video_path, output_dir

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/transform', methods=['POST'])
def transform():
    if 'file' not in request.files:
        return 'No file provided', 400

    file = request.files['file']
    file_type = file.content_type

    if file_type.startswith('image/'):
        try:
            img = Image.open(file).convert('RGB')
            input_tensor = preprocess_image(img)
            translated_tensor = translate_image(G_B2A, input_tensor)
            output_image = postprocess(translated_tensor)
            output_image = Image.fromarray(output_image)

            img_byte_arr = BytesIO()
            output_image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)

            return send_file(img_byte_arr, mimetype='image/png')
        except Exception as e:
            print(f"Error processing image: {e}")
            return f"Error processing image: {e}", 400

    elif file_type.startswith('video/'):
        try:
            video_path = 'input_video.mp4'
            file.save(video_path)
            output_dir = 'processed_frames'
            success, output_path, full_output_dir = process_video(video_path, output_dir, G_B2A)

            if not success or not os.path.exists(output_path):
                return "Error: Processed video file not found", 500

            return jsonify({
                'output_video_path': output_path,
                'full_output_dir': full_output_dir
            })
        except Exception as e:
            print(f"Error processing video: {e}")
            return f"Error processing video: {e}", 500
        finally:
            if os.path.exists(video_path):
                os.remove(video_path)
    else:
        return 'Unsupported file type', 400

@app.route('/processed_frames/<path:filename>')
def serve_processed_file(filename):
    return send_from_directory('processed_frames', filename)

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)