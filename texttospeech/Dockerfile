#FROM python:3.7
FROM centos:centos7

MAINTAINER Jonas Freiknecht "jonas.freiknecht@bwi.de"

RUN /usr/bin/yum --assumeyes install https://centos7.iuscommunity.org/ius-release.rpm
RUN /usr/bin/yum --assumeyes update
RUN /usr/bin/yum --assumeyes install epel-release gcc
RUN /usr/bin/yum --assumeyes install python36u python36pip
RUN /usr/bin/yum --assumeyes install libsndfile

RUN mkdir /app

COPY texttospeech.py /app/
COPY ./requirements.txt /app/

WORKDIR /app

RUN pip3 install -f https://download.pytorch.org/whl/torch_stable.html -r requirements.txt

# centos image
ENTRYPOINT ["python3"]
CMD ["texttospeech.py"]
