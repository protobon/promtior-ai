FROM python:3.11-slim
WORKDIR /src
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY .env .
COPY db ./db
COPY server.py .
EXPOSE 8000
CMD ["python3", "server.py"]