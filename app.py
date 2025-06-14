from flask import Flask, request, jsonify, render_template
import librosa
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
    
    # Save the file temporarily
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    try:
        # Load audio file
        y, sr = librosa.load(filepath, sr=None)
        
        # Extract features
        tempo = librosa.beat.tempo(y=y, sr=sr)[0]  # BPM
        chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)  # Chroma (Key)
        mfcc = librosa.feature.mfcc(y=y, sr=sr).mean(axis=1)  # MFCCs
        
        # Clean up (delete the file after processing)
        os.remove(filepath)
        
        return jsonify({
            "tempo": float(tempo),
            "chroma": chroma.tolist(),
            "mfcc": mfcc.tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)