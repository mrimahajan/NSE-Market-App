# pull official base image
FROM python:3

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# set work directory
WORKDIR /app

# install dependencies
RUN python3 -m pip install --upgrade pip setuptools wheel
RUN pip3 install --upgrade pip
COPY ./requirements.txt /app/
RUN pip3 install -r requirements.txt

# copy project
COPY . /app/
