###### Distribution #######
FROM ubuntu

###### Essentials #######
RUN apt-get update && apt-get install -y build-essential


####### Install Miniconda ######
RUN wget --quiet \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 


###### Install my work in the container ######
RUN pip install --upgrade pip
COPY environment.yml /
RUN conda update conda
# have packages on "base" (i.e: no need to use another environment in the Docker environment)
RUN conda env update -f environment.yml
# make non-activate conda commands available
ENV PATH=$CONDA_DIR/bin:$PATH
# make conda activate command available from /bin/bash --login shells
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile
# make conda activate command available from /bin/bash --interative shells
RUN conda init bash


COPY . /app
WORKDIR /app
