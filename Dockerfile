#
#clone the tensorflow docker image
FROM tensorflow/tensorflow:latest-gpu



# set where python will look for modules
ENV PYTHONPATH /tf
# Set desired Python version
ENV python_version 3.11

# set the working directory in the container
WORKDIR /tf

# Install desired Python version (the current TF image is be based on Ubuntu at the moment)
RUN apt install -y python${python_version}

# Set default version for root user - modified version of this solution: https://jcutrer.com/linux/upgrade-python37-ubuntu1810
RUN update-alternatives --install /usr/local/bin/python python /usr/bin/python${python_version} 1

# Update pip: https://packaging.python.org/tutorials/installing-packages/#ensure-pip-setuptools-and-wheel-are-up-to-date
RUN python -m pip install --upgrade pip setuptools wheel

ADD requirements.txt .
RUN python -m pip install -r requirements.txt
ADD . .
CMD ["python", "Main.py","1"]
