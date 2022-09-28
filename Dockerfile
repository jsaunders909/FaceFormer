FROM pytorch/pytorch
RUN pip install opencv-python scipy pyrender librosa transformers trimesh tqdm numpy soundfile
RUN apt-get update && apt-get install -y libboost-dev libsndfile-dev libgl1 freeglut3-dev
RUN git clone https://github.com/MPI-IS/mesh.git
RUN cd psbody && make all