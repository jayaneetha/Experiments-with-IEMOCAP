FROM tensorflow/tensorflow:1.14.0-py3

RUN mkdir /iemocap

ADD . /iemocap/
COPY .netrc /root/.netrc

RUN apt-get update \
&& apt-get upgrade -y \
&& apt-get install -y \
&& apt-get -y install apt-utils gcc libpq-dev libsndfile-dev vim

WORKDIR /iemocap
RUN cd /iemocap
RUN pip install -r requirements.txt
RUN export WANDB_API_KEY=5dd532fa8e46bf1d25a597f96c118bbfe549a807
ENV WANDB_API_KEY=5dd532fa8e46bf1d25a597f96c118bbfe549a807

CMD ${RUN_CMD}