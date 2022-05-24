# Ubuntu 20.04
FROM ubuntu:focal

# 切换默认shell为bash
SHELL ["/bin/bash", "-c"]

ADD ./ /asrt_server

WORKDIR /asrt_server

# 最小化源，缩短apt update时间(ca-certificates必须先安装才支持换tsinghua源)
RUN echo 'deb http://archive.ubuntu.com/ubuntu/ focal main restricted' > /etc/apt/sources.list

RUN apt update && apt install -y ca-certificates

RUN echo $'\
deb http://mirrors.aliyun.com/ubuntu/ focal main restricted universe multiverse\
\n# deb-src http://mirrors.aliyun.com/ubuntu/ focal main restricted universe multiverse\n\
deb http://mirrors.aliyun.com/ubuntu/ focal-security main restricted universe multiverse\
\n# deb-src http://mirrors.aliyun.com/ubuntu/ focal-security main restricted universe multiverse\n\
deb http://mirrors.aliyun.com/ubuntu/ focal-updates main restricted universe multiverse\
\n# deb-src http://mirrors.aliyun.com/ubuntu/ focal-updates main restricted universe multiverse\n\
deb http://mirrors.aliyun.com/ubuntu/ focal-proposed main restricted universe multiverse\
\n# deb-src http://mirrors.aliyun.com/ubuntu/ focal-proposed main restricted universe multiverse\n\
deb http://mirrors.aliyun.com/ubuntu/ focal-backports main restricted universe multiverse\
\n# deb-src http://mirrors.aliyun.com/ubuntu/ focal-backports main restricted universe multiverse'\
> /etc/apt/sources.list

RUN apt update && apt install -y python3 python3-pip 

RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip3 install wave scipy matplotlib tensorflow-cpu==2.5.3 numpy==1.19.2 requests flask waitress grpcio==1.34.0 grpcio-tools==1.34.0 protobuf

RUN echo $'cd /asrt_server \n python3 asrserver_http.py & \n python3 asrserver_grpc.py' > /asrt_server/start.sh && chmod +x /asrt_server/start.sh

# refer: https://docs.docker.com/engine/reference/builder/#expose
EXPOSE 20001/tcp 20002/tcp

ENTRYPOINT ["/bin/bash", "/asrt_server/start.sh"]

# https://docs.docker.com/engine/reference/commandline/build/#options
# docker build --progress plain --rm --build-arg TAG=1.3.0 --tag asrt/api_server:1.3.0 .
# https://docs.docker.com/engine/reference/commandline/run/#options
# docker run --rm -it -p 20001:20001 -p 20002:20002 --name asrt -d asrt/api_server:1.3.0
