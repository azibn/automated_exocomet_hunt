FROM continuumio/miniconda3:4.8.2

MAINTAINER Azib Norazman

# Environment setup
COPY environment.yml /
RUN conda env create -f environment.yml
