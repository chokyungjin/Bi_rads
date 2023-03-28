FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel
# FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
# apt
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get update && apt-get upgrade -y && \
    apt-get -y install bash \
    build-essential \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libomp-dev \
    wget \
    unzip \
    software-properties-common \
    && rm -rf /var/lib/apt/lists