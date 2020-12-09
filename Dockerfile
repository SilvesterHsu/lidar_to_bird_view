FROM ubuntu:16.04

ENV DEBIAN_FRONTEND noninteractive

# python-pcl dependent
RUN apt update -y && \
    apt install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    add-apt-repository ppa:v-launchpad-jochen-sprickerhof-de/pcl -y && \
    apt install python3.5-dev libpcl-dev -y

RUN apt install curl -y && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.5 get-pip.py

RUN pip3 install cython==0.25.2 numpy && \
    pip3 install python-pcl opencv-python Pillow matplotlib tqdm

# lidar to image code
ADD longshaw /longshaw

WORKDIR "longshaw"
VOLUME ["/dataset"]
CMD ["/bin/bash", "-c", "python3.5 lidar2birdview.py"]
