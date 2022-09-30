FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
RUN pip install opencv-python scipy pyrender librosa trimesh tqdm numpy soundfile
RUN pip install transformers==4.6.1
RUN apt-key adv --recv-keys --keyserver keyserver.ubuntu.com `apt-get update 2>&1 | grep -o '[0-9A-Z]\{16\}$' | xargs`
RUN apt-get install -y libboost-dev libsndfile-dev libgl1 freeglut3-dev git
RUN git clone https://github.com/MPI-IS/mesh.git
RUN cd mesh && make all
RUN apt-get install -y build-essential cmake
RUN apt-get install -y python3-dev
RUN apt-get install -y sed
RUN sed -i 's/print numpy.get_include()/print(numpy.get_include())/g' mesh/mesh/cmake/thirdparty.cmake -DPYTHON_EXECUTABLE:FILEPATH=/opt/conda/bin/python
RUN cd mesh/mesh/cmake && cmake ..