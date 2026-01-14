FROM runpod/base:0.6.1-cuda12.1.0

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "-u", "handler.py"]
