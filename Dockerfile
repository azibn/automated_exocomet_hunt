# Distribution 
FROM ubuntu


# Essentials 
RUN apt-get update && apt-get install -y build-essential wget vim swig curl bzip2

# Set working directory in the container 
WORKDIR /app

# Install Miniforge (I think this takes care of ARM architecture)
## RUN apt-get update && apt-get install -y wget bzip2
RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
RUN bash Miniforge3-Linux-aarch64.sh -b -p /opt/conda
RUN rm Miniforge3-Linux-aarch64.sh

# Add Conda to the PATH
ENV PATH="/opt/conda/bin:$PATH"

# Initialize shell to use conda activate
RUN conda init bash

# Create and activate Conda environment with Mamba
RUN conda install -y mamba -n base -c conda-forge
COPY environment.yml .
RUN mamba env create -n auto-exo -f environment.yml

# Set entrypoint command
#CMD ["conda", "run", "-n", "auto-exo", "/bin/bash"]

## TODO
### Create second environment for PyMVPA for SOM

