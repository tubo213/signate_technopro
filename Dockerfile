FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04

SHELL ["/bin/bash", "-c"]
ENV HOME /root
ENV lang="ja_jp.utf-8" language="ja_jp:ja" lc_all="ja_jp.utf-8"
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt-get install -y python3.8
RUN apt-get install -y node-gyp>=0.10.9 libssl-dev 
RUN apt-get install -y npm
RUN apt-get update && apt-get install -y \
build-essential \
libssl-dev \
libffi-dev \
python3.8-dev \
python3.8-venv \
python3-pip \
python3-ipdb \
tzdata \
curl \
nodejs \
gnupg \
vim \
gosu \
ffmpeg \
htop \
git \
tmux \
graphviz \
sudo \
python3-pip

# python3.8 をデフォルトに
RUN mkdir -p $HOME/bin
RUN ln -s -f /usr/bin/python3.8 /usr/bin/python
RUN ln -s -f /usr/bin/python3.8 /usr/bin/python3

# # pyenv
#  ENV PYENV_ROOT $HOME/.pyenv
#  ENV PATH $PYENV_ROOT/bin:$PATH
#  RUN git clone https://github.com/pyenv/pyenv.git $HOME/.pyenv
#  RUN echo 'eval "$(pyenv init -)"' >> ~/.bashrc && \
#      eval "$(pyenv init -)"
# RUN pyenv install 3.8.10 && \
#     pyenv global 3.8.10

# poetry
RUN pip3 install --upgrade pip && \
    pip install --upgrade pip

ENV POETRY_HOME="/.poetry"
ENV PATH="$POETRY_HOME/bin:$PATH"
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python -
ENV PATH /root/.local/bin:$PATH


#############################################
# Library
#############################################
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false && \
    poetry install

# Pytorch
RUN pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Pytorch geometric
RUN pip3 install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cu113.html

# JAX
RUN pip3 install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html


WORKDIR /workspace