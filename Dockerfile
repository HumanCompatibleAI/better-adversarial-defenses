# Based on OpenAI's mujoco-py Dockerfile

ARG USE_MPI=True

# base stage contains just binary dependencies.
# This is used in the CI build.
FROM nvidia/cuda:10.0-runtime-ubuntu18.04 AS base
ARG DEBIAN_FRONTEND=noninteractive

# installing apt dependencies
RUN echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | debconf-set-selections \
    && apt-get update -q \
    && apt-get install -y --no-install-recommends \
    build-essential cmake \
    curl \
    ffmpeg xvfb \
    git \
    net-tools \
    unzip \
    vim sudo rsync \
    wget \
    htop \
    screen mongodb-server \
    libosmesa6-dev libgl1-mesa-glx libglfw3 python-pygame libsdl1.2-dev pkg-config libfreetype6-dev \
    build-essential autoconf libtool pkg-config python-opengl python-pil \
    python-pyrex python-pyside.qtopengl idle-python2.7 qt4-dev-tools qt4-designer libqtgui4 \
    libqtcore4 libqt4-xml libqt4-test libqt4-script libqt4-network libqt4-dbus python-qt4 python-qt4-gl \
    libgle3 python-dev libgl1-mesa-dev libgl1-mesa-glx libosmesa6-dev python3-scipy libglew-dev patchelf \
    python-dev libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev libsdl1.2-dev libsmpeg-dev \
    python-numpy subversion libportmidi-dev ffmpeg libswscale-dev libavformat-dev libavcodec-dev \
    ttf-mscorefonts-installer \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV LANG C.UTF-8

# installing mujoco
RUN    mkdir -p /root/.mujoco \
    && curl -o mujoco131.zip https://www.roboti.us/download/mjpro131_linux.zip \
    && unzip mujoco131.zip -d /root/.mujoco \
    && curl -o mujoco200.zip https://www.roboti.us/download/mujoco200_linux.zip \
    && unzip mujoco200.zip -d /root/.mujoco \
    && ln -s /root/.mujoco/mujoco200_linux /root/.mujoco/mujoco200
    && rm mujoco131.zip mujoco200.zip

RUN ln -s /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so
RUN touch /root/.mujoco/mjkey.txt
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco200/bin:/root/.mujoco/mjpro131/bin:${LD_LIBRARY_PATH}

# installing conda
RUN wget -c https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh 
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -f && eval "$(/root/miniconda3/bin/conda shell.bash hook)" && conda init

# creating environments
WORKDIR /
RUN git clone --recursive https://github.com/HumanCompatibleAI/better-adversarial-defenses
WORKDIR /better-adversarial-defenses
RUN conda env create -f adv-tf1.yml && conda env create -f adv-tf2.yml

# installing submodules and the project
RUN conda run -n adv-tf1 pip install -e multiagent-competition &&\
    conda run -n adv-tf2 pip install -e multiagent-competition &&\
    conda run -n adv-tf1 pip install -e adversarial-policies &&\
    conda run -n adv-tf2 pip install -e adversarial-policies &&\
    conda run -n adv-tf1 pip install -e . &&\
    conda run -n adv-tf2 pip install -e . &&\
RUN conda run -n adv-tf2 python ray/python/ray/setup-dev.py --yes

# starting Xvfb
CMD nohup Xvfb -screen 0 1024x768x24 & sleep infinity
ENV DISPLAY :0

# running tests
CMD bash test.sh
