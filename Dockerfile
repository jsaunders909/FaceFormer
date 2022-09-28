FROM pytorch/pytorch
RUN pip install opencv-python scipy pyrender librosa transformers trimesh tqdm numpy soundfile
RUN apt-get update && apt-get install -y libboost-dev libsndfile-dev nvidia-driver-libs:i386