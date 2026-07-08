from fastembed import TextEmbedding

def download_model():
    print("Downloading fastembed ONNX model weights...")
    # This triggers the download and caches it in ~/.cache/fastembed
    TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("Model downloaded and cached successfully!")

if __name__ == "__main__":
    download_model()
