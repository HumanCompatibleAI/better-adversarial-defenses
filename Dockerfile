# Based on OpenAI's mujoco-py Dockerfile

ARG USE_MPI=True

# base stage contains just binary dependencies.
# This is used in the CI build.
FROM nvidia/cuda:10.0-runtime-ubuntu18.04 AS base
ARG DEBIAN_FRONTEND=noninteractive

RUN echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | debconf-set-selections \
    && apt-get update -q \
    && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ffmpeg \
    git \
    net-tools \
    unzip \
    vim \
    ttf-mscorefonts-installer \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV LANG C.UTF-8

RUN    mkdir -p /root/.mujoco \
    && curl -o mujoco131.zip https://www.roboti.us/download/mjpro131_linux.zip \
    && unzip mujoco131.zip -d /root/.mujoco \
    && rm mujoco131.zip

ENV LD_LIBRARY_PATH /root/.mujoco/mjpro131/bin:${LD_LIBRARY_PATH}
