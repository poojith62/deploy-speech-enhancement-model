import os
import time
from flask import Flask, request, render_template, send_file, jsonify
from werkzeug.utils import secure_filename
from final import prediction

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = os.path.abspath('uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure output folder
OUTPUT_FOLDER = os.path.abspath('output')
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    print("Upload route called")
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Call the prediction function
        audio_dir_prediction = app.config['UPLOAD_FOLDER']
        dir_save_prediction = OUTPUT_FOLDER + os.path.sep  # Add separator at the end
        audio_input_prediction = [filename]
        audio_output_prediction = 'denoise_' + filename

        try:
            print(f"Starting prediction with parameters:")
            print(f"Weights: weights")
            print(f"Audio dir: {audio_dir_prediction}")
            print(f"Save dir: {dir_save_prediction}")
            print(f"Input audio: {audio_input_prediction}")
            print(f"Output audio: {audio_output_prediction}")
            
            prediction('weights', audio_dir_prediction, dir_save_prediction, audio_input_prediction, audio_output_prediction)
            
            time.sleep(2)  # Wait for 2 seconds
            
            # Check if the file was created
            output_path = os.path.join(OUTPUT_FOLDER, audio_output_prediction)
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                print(f"Denoised file successfully created at: {output_path}")
                print(f"File size: {file_size} bytes")
                if file_size > 0:
                    return jsonify({
                        'original': f'/audio/original/{filename}',
                        'enhanced': f'/audio/denoised/{audio_output_prediction}'
                    })
                else:
                    print("Error: Denoised file is empty")
                    return jsonify({'error': 'Denoised file is empty'}), 500
            else:
                print(f"Error: Denoised file not found at {output_path}")
                # List contents of the output directory
                print("Contents of output directory:")
                for file in os.listdir(OUTPUT_FOLDER):
                    print(f" - {file}")
                return jsonify({'error': 'Denoised file not created'}), 500
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return jsonify({'error': 'Prediction failed'}), 500

@app.route('/audio/original/<filename>')
def serve_original_audio(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        print(f"Error: Original file not found at {file_path}")
        return jsonify({'error': f'File not found: {filename}'}), 404
    try:
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        print(f"Error serving original file {filename}: {str(e)}")
        return jsonify({'error': 'Error serving file'}), 500

@app.route('/audio/denoised/<filename>')
def serve_denoised_audio(filename):
    print(f"Attempting to serve denoised file: {filename}")
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.exists(file_path):
        print(f"Error: Denoised file not found at {file_path}")
        return jsonify({'error': f'File not found: {filename}'}), 404
    try:
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        print(f"Error serving denoised file {filename}: {str(e)}")
        return jsonify({'error': 'Error serving file'}), 500

if __name__ == '__main__':
    app.run(debug=True)
