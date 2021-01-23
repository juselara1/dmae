# Base tensorflow image
FROM tensorflow/tensorflow:2.4.1-gpu

# Install DMAE
RUN pip install tensorflow-addons==0.12.0
RUN pip install matplotlib==3.3.2
RUN pip install scikit-learn==0.23.2
RUN apt install --assume-yes git
RUN git clone -b dev --single-branch https://github.com/juselara1/dmae.git /home/dmae/
RUN pip install /home/dmae/
