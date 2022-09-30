FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
RUN pip install opencv-python scipy pyrender librosa trimesh tqdm numpy soundfile
RUN pip install transformers==4.6.1
RUN apt-key adv --recv-keys --keyserver keyserver.ubuntu.com `apt-get update 2>&1 | grep -o '[0-9A-Z]\{16\}$' | xargs`
RUN apt-get install -y libboost-dev libsndfile-dev libgl1 freeglut3-dev git
RUN git clone https://github.com/MPI-IS/mesh.git
RUN cd mesh && make all
RUN cd mesh/mesh/cmake && cmake ..