FROM tensorflow/tensorflow:1.14.0-py3

RUN mkdir /iemocap
RUN mkdir /iemocap/pkl
RUN mkdir /iemocap/rl-files

ADD . /iemocap/
COPY .netrc /root/.netrc
WORKDIR /iemocap
RUN cd /iemocap
RUN export WANDB_API_KEY=5dd532fa8e46bf1d25a597f96c118bbfe549a807
ENV WANDB_API_KEY=5dd532fa8e46bf1d25a597f96c118bbfe549a807

RUN apt-get update \
&& apt-get upgrade -y \
&& apt-get install -y \
&& apt-get -y install apt-utils gcc libpq-dev libsndfile-dev vim

RUN pip install -r requirements.txt

CMD ${RUN_CMD}