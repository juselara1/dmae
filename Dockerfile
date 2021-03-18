# Base tensorflow image
FROM tensorflow/tensorflow:2.4.1-gpu

# Install DMAE
RUN apt update --assume-yes
RUN apt install --assume-yes git
RUN apt install --assume-yes vim
RUN git clone -b dev --single-branch https://github.com/juselara1/dmae.git /home/repo/
RUN pip install -r /home/repo/requirements_docker_gpu.txt
RUN pip install --no-deps /home/repo/ 
RUN mkdir /home/dmae

WORKDIR /home/dmae

RUN cp -r /home/repo/examples /home/dmae
RUN rm -rf /home/repo

ADD ./examples/scripts/replication/data /home/dmae/examples/scripts/replication/data/
