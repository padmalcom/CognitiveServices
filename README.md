# CognitiveServices
A collection of cognitive services based on python rest services. Can be packaged as docker containers.

![Cognitive Services](https://github.com/padmalcom/CognitiveServices/raw/master/docs/CognitiveServices.png "Cognitive Services")

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

## Backlog
There are multiple services that I plan to add. If you have any wishes feel free to add an issue.
- Add an about function for each service with information about and thanks to all model providers
- Text summarization
- Language detection
- Speech-to-text
- Speaker identification
- ...
