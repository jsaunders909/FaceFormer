FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
RUN pip install opencv-python scipy pyrender librosa transformers trimesh tqdm numpy soundfile
# RUN apt-get update
RUN apt-get install -y libboost-dev libsndfile-dev libgl1 freeglut3-dev git
RUN git clone https://github.com/MPI-IS/mesh.git
RUN cd mesh && make all