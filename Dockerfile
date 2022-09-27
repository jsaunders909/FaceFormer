FROM pytorch/pytorch
RUN pip install opencv-python scipy pyrender librosa transformers trimesh tqdm numpy
RUN sudo apt-get install libboost-dev