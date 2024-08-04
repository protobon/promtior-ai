FROM python:3.11-slim
WORKDIR /src
COPY . /src
RUN pip install -r requirements.txt
CMD ["python3", "server.py"]
EXPOSE 8000