FROM centos:centos7

MAINTAINER Jonas Freiknecht "j.freiknecht@googlemail.com"

RUN /usr/bin/yum --assumeyes install https://centos7.iuscommunity.org/ius-release.rpm
RUN /usr/bin/yum --assumeyes update
RUN /usr/bin/yum --assumeyes install epel-release gcc cmake libx11-dev
RUN /usr/bin/yum --assumeyes install python36u python36pip python36-devel
RUN /usr/bin/yum --assumeyes group install "Development Tools"


RUN mkdir /app
RUN mkdir /app/known

COPY facedetection.py /app/
COPY ./requirements.txt /app/
COPY known/ /app/known/

WORKDIR /app

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

ENTRYPOINT ["python3"]
CMD ["facedetection.py"]

EXPOSE 5000
