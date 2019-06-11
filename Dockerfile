FROM nvidia/cuda:10.0-cudnn7-devel
MAINTAINER Ajaeiya Georgi "gaa40@lri.fr"

# install sys requirements 
RUN \
    apt-get update && apt-get install -y \
    autoconf \
    automake \
    libtool \
    libsm6 \
    libxext6 \
    libxrender1 \
    libfontconfig1 \ 
    build-essential \
    python3-pip \
    python3-dev \
    && pip3 install --upgrade pip

# copy all files from origin folder to the image files
COPY . /app

# set working directory to darknet for a fresh build
WORKDIR /app/darknet

# set GPU=1 if available , set CUDNN and CUDNN_HALF =1 if nvidia is available: Volta, Turing 
RUN \
    sed -i 's/GPU=.*/GPU=1/' Makefile && \
    sed -i 's/CUDNN=.*/CUDNN=1/' Makefile && \
    sed -i 's/CUDNN_HALF=.*/CUDNN_HALF=1/' Makefile && \
    sed -i 's/LIBSO=.*/LIBSO=1/' Makefile && \
    make

# set working directory to app and run
WORKDIR /app
RUN pip3 install -r requirements.txt

# test nvidia driver
CMD nvidia-smi -q

ENTRYPOINT ["python3"]
CMD ["app.py"]
