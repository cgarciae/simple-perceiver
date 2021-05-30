FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu20.04

RUN apt update && apt install -y python3-pip python-is-python3
RUN pip install poetry && poetry config virtualenvs.in-project true
