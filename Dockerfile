
# Use the official lightweight Python image.
# https://hub.docker.com/_/python
# FROM python:3.11-slim

# # Allow statements and log messages to immediately appear in the logs
# ENV PYTHONUNBUFFERED True

# # Copy local code to the container image.
# ENV APP_HOME /app
# WORKDIR $APP_HOME
# COPY . ./

# #If you use the Firebase service, you must write this line
# COPY research-h-firebase-adminsdk-i9dtx-12a1568005.json /app/research-h-firebase-adminsdk-i9dtx-12a1568005.json

# # Install production dependencies.
# #インストールするライブラリはここに記載すればOK
# #ここでは，requirements.txtにインストールするものが記載されていてそれをpipでインストールしている．
# RUN pip install --upgrade pip
# RUN pip install --no-cache-dir -r requirements.txt
# #RUN pip install firebase-admin
# RUN apt update -y && apt upgrade -y
# RUN apt-get install  -y libgl1-mesa-dev libglib2.0

# RUN pip install --no-cache-dir opencv-python

# RUN pip install --no-cache-dir pybase64

# RUN pip install --upgrade Pillow


# RUN pip install google-cloud
# Run pip install google-cloud-aiplatform
# Run pip install protobuf

# # Run pip install google.protobuf
# #requirements.txtというファイルが必要

# # Run the web service on container startup. Here we use the gunicorn
# # webserver, with one worker process and 8 threads.
# # For environments with multiple CPU cores, increase the number of workers
# # to be equal to the cores available.
# # Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.
# CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app 
# #main.pyというファイルが必要
# # ______________________________________________________________________________________



# # FROM pytorch/torchserve:latest-cpu

# # # install dependencies
# # RUN pip3 install transformers

# # # copy model artifacts, custom handler and other dependencies
# # COPY ./custom_text_handler.py /home/model-server/
# # COPY ./index_to_name.json /home/model-server/
# # #modelとその不随ファイルをDockerにコピーしてコピーを保存
# # COPY ./model/$APP_NAME/ /home/model-server/ 

# # # create torchserve configuration file
# # USER root
# # RUN printf "service_envelope=json" >> /home/model-server/config.properties
# # RUN printf "inference_address=http://0.0.0.0:7080" >> /home/model-server/config.properties
# # RUN printf "management_address=http://0.0.0.0:7081" >> /home/model-server/config.properties
# # USER model-server

# # # expose health and prediction listener ports from the image
# # EXPOSE 7080
# # EXPOSE 7081

# # # create model archive file packaging model artifacts and dependencies
# # RUN torch-model-archiver -f  --model-name=$APP_NAME  --version=1.0  --serialized-file=/home/model-server/pytorch_model.bin  --handler=/home/model-server/custom_text_handler.py  --extra-files "/home/model-server/config.json,/home/model-server/tokenizer.json,/home/model-server/training_args.bin,/home/model-server/tokenizer_config.json,/home/model-server/special_tokens_map.json,/home/model-server/vocab.txt,/home/model-server/index_to_name.json"  --export-path=/home/model-server/model-store

# # # run Torchserve HTTP serve to respond to prediction requests
# # CMD ["torchserve",  "--start",  "--ts-config=/home/model-server/config.properties",  "--models",  "$APP_NAME=$APP_NAME.mar",  "--model-store",  "/home/model-server/model-store"]



# __________________________________________________
# Use the official lightweight Python image.
# https://hub.docker.com/_/python
# FROM pytorch/torchserve:latest-cpu
FROM python:3.11-slim

# Allow statements and log messages to immediately appear in the logs
ENV PYTHONUNBUFFERED True

EXPOSE 8080

# Copy local code to the container image.
ENV APP_HOME /app $APP_NAME $PORT
WORKDIR $APP_HOME
COPY . ./

#If you use the Firebase service, you must write this line
COPY research-h-firebase-adminsdk-i9dtx-12a1568005.json /app/research-h-firebase-adminsdk-i9dtx-12a1568005.json

# ここが原因でエラー出てる？？
# APP_NAME = "model"
# PORT = 8080


# Install production dependencies.
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN apt update -y && apt upgrade -y
RUN apt-get install  -y libgl1-mesa-dev libglib2.0

RUN pip install --no-cache-dir opencv-python
RUN pip install --no-cache-dir pybase64
RUN pip install --upgrade Pillow
RUN pip install google-cloud
RUN pip install google-cloud-aiplatform
RUN pip install protobuf
RUN pip install transformers

# #Copy model artifacts, custom handler and other dependencies
# COPY ./custom_text_handler.py /home/model-server/
# COPY ./index_to_name.json /home/model-server/
COPY ./model/ /home/model-server/

# create torchserve configuration file
# USER root
# RUN printf "service_envelope=json" >> /home/model-server/config.properties
# RUN printf "inference_address=http://0.0.0.0:$PORT" >> /home/model-server/config.properties
# RUN printf "management_address=http://0.0.0.0:$PORT" >> /home/model-server/config.properties
# USER model-server

# create model archive file packaging model artifacts and dependencies
# RUN torch-model-archiver -f  --model-name=$APP_NAME  --version=1.0  --serialized-file=/home/model-server/pytorch_model.bin  --handler=/home/model-server/custom_text_handler.py  --extra-files "/home/model-server/config.json,/home/model-server/tokenizer.json,/home/model-server/training_args.bin,/home/model-server/tokenizer_config.json,/home/model-server/special_tokens_map.json,/home/model-server/vocab.txt,/home/model-server/index_to_name.json"  --export-path=/home/model-server/model-store

# run Torchserve HTTP serve to respond to prediction requests
CMD ["torchserve",  "--start",  "--ts-config=/home/ais_lab/myapp-run/config.properties",  "--models",  "${APP_NAME}=${APP_NAME}.mar",  "--model-store",  "/home/model-server/model-store"]
# CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app 
