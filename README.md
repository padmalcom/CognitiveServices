# CognitiveServices
A collection of cognitive services based on python rest services. Can be packaged as docker containers.

Each service can be built and ran by executing the following two commands from each service folder:
```
docker build -t [servicename] .
sudo docker run -p 5000:5000 [servicename]
```

or simply via python by executing:

```
conda create -n [servicename] python=3.6
conda activate [servicename]
python [servicename].py
```
