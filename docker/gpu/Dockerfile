FROM nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04
MAINTAINER Ashik Vetrivelu <vcashik@gmail.com>

RUN apt-get update
RUN apt-get install --no-install-recommends --no-install-suggests -y curl

ENV NVIDIA_RUN http://us.download.nvidia.com/tesla/390.46/NVIDIA-Linux-x86_64-390.46.run
ENV CUDA_RUN https://developer.nvidia.com/compute/cuda/9.1/Prod/local_installers/cuda_9.1.85_387.26_linux

RUN apt-get install --no-install-recommends -y python3-dev \
   git \
   wget \
   unzip \
   liblapack-dev \
   libopenblas-dev \
   python3-numpy \
   build-essential \
   sudo \
   cmake \
   clang-3.6 \
   clang-format-3.6 \
   python3-pip \
   python3-nose \
   python3-wheel \
   python3-tk\
   python3-setuptools \
   kmod \
   python3-scipy

RUN pip3 install --upgrade pip

RUN pip3 install --upgrade numpy

RUN pip3 install --upgrade Cython

RUN cd /root && wget http://www.cmake.org/files/v3.8/cmake-3.8.1.tar.gz && \
  tar -xzf cmake-3.8.1.tar.gz && cd cmake-3.8.1 && \
  ./configure && \
  make -j "$(nproc)" && \
  make install

ENV CUDA_ROOT /usr/local/cuda/bin

RUN pip3 install --upgrade pip
RUN pip3 install --upgrade six
RUN pip3 install --upgrade torch
RUN pip3 install --upgrade torchvision
RUN pip3 install --upgrade pillow
RUN pip3 install --upgrade deepdish
RUN pip3 install --upgrade tqdm
RUN pip3 install --upgrade pycocotools
RUN pip3 install --upgrade nltk
RUN pip3 install --upgrade matplotlib
# RUN pip3 install --upgrade python3-tk


WORKDIR /scene_detection

RUN mkdir -p models
RUN mkdir -p checkpoint
RUN mkdir -p dataset

# ADD download.sh download.sh
# RUN chmod +x download.sh
# RUN ./download.sh

# ADD scripts scripts/
# RUN python3 scripts/build_vocab.py
ADD dataset/ dataset/vocab.pkl

ADD core core/

ADD main.py main.py

RUN rm -rf /tmp/*

ENTRYPOINT ["python3","main.py"]
