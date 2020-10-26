# Based on OpenAI's mujoco-py Dockerfile

# base stage contains just binary dependencies.
# This is used in the CI build.
FROM ubuntu:18.04 AS base
ARG DEBIAN_FRONTEND=noninteractive

# installing apt dependencies
RUN apt-get update -q \
    && apt-get install -y --no-install-recommends \
	autoconf build-essential cmake curl dialog \
	ffmpeg git htop libavcodec-dev libavformat-dev \
	libfreetype6-dev libgl1-mesa-dev libgl1-mesa-glx libgle3 libglew-dev \
	libglfw3 libosmesa6-dev libportmidi-dev \
	libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev libsdl1.2-dev libsmpeg-dev \
	libswscale-dev libtool mongodb-server net-tools patchelf \
	pkg-config rsync screen ca-certificates \
	subversion sudo unzip vim wget xvfb dialog \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV LANG C.UTF-8

# installing mujoco
RUN    mkdir -p /root/.mujoco \
    && curl -o mujoco131.zip https://www.roboti.us/download/mjpro131_linux.zip \
    && unzip mujoco131.zip -d /root/.mujoco \
    && curl -o mujoco200.zip https://www.roboti.us/download/mujoco200_linux.zip \
    && unzip mujoco200.zip -d /root/.mujoco \
    && ln -s /root/.mujoco/mujoco200_linux /root/.mujoco/mujoco200 \
    && rm mujoco131.zip mujoco200.zip

#RUN ln -s /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so
RUN touch /root/.mujoco/mjkey.txt
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco200/bin:/root/.mujoco/mjpro131/bin:${LD_LIBRARY_PATH}

# installing miniconda
# based on https://hub.docker.com/r/continuumio/miniconda3/dockerfile
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc
SHELL ["/bin/bash", "-c"]
ENV PATH /opt/conda/bin:$PATH

# cloning repository
WORKDIR /
RUN git clone --recursive https://github.com/HumanCompatibleAI/better-adversarial-defenses
WORKDIR /better-adversarial-defenses


# creating environments
RUN conda update conda -y \
 && conda env create -f adv-tf1.yml \
 && conda env create -f adv-tf2.yml \
 && ln -s /opt/conda/envs/adv-tf2/fonts/ /root/.fonts \
 && fc-cache -f -v \
 # installing submodules and the project
 && conda run -n adv-tf1 pip install -e multiagent-competition \
 && conda run -n adv-tf2 pip install -e multiagent-competition \
 && conda run -n adv-tf1 pip install -e adversarial-policies \
 && conda run -n adv-tf2 pip install -e adversarial-policies \
 && conda run -n adv-tf1 pip install -e . \
 && conda run -n adv-tf2 pip install -e . \
 && conda run -n adv-tf2 python ray/python/ray/setup-dev.py --yes \
 # cleaning conda cache
 && conda clean --all -y \
 && conda run -n adv-tf1 pip cache purge \
 && rm -rf ~/.cache/pip/

# running tests
CMD bash test.sh
