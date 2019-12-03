FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:1.2.0-gpu-py36-cu100-ubuntu16.04

RUN pip install pandas torchvision

COPY cifar10_pytorch.py /opt/ml/cifar10_pytorch.py
COPY core.py /opt/ml/core.py
COPY torch_backend.py /opt/ml/torch_backend.py

ENTRYPOINT ["python", "/opt/ml/cifar10_pytorch.py"]