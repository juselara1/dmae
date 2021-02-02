# Base tensorflow image
FROM tensorflow/tensorflow:2.4.1-gpu

# Install DMAE
RUN apt install --assume-yes git
RUN apt install --assume-yes vim
RUN git clone -b dev --single-branch https://github.com/juselara1/dmae.git /home/dmae/
RUN pip install -r /home/dmae/requirements_docker_gpu.txt
RUN pip install --no-deps /home/dmae/ 
RUN cp -r /home/dmae/examples /home/
RUN rm -rf /home/dmae
