FROM centos:centos7

MAINTAINER Jonas Freiknecht "j.freiknecht@googlemail.com"

RUN /usr/bin/yum --assumeyes install https://centos7.iuscommunity.org/ius-release.rpm
RUN /usr/bin/yum --assumeyes update
RUN /usr/bin/yum --assumeyes install epel-release gcc
RUN /usr/bin/yum --assumeyes install python36u python36pip

RUN mkdir /app
RUN mkdir /encodings/

COPY entityextraction.py /app/
COPY ./requirements.txt /app/

WORKDIR /app

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN python3 -m spacy download en_core_web_lg
RUN python3 -m spacy download en_core_web_sm

ENTRYPOINT ["python3"]
CMD ["entityextraction.py"]
