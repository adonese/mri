FROM python:3.6-slim-stretch

ADD requirements.txt /
RUN pip install -r /requirements.txt

ADD . /app
WORKDIR /app

EXPOSE 5010
CMD [ "python" , "app.py"]