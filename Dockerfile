FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn8-runtime

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "-u", "handler.py"]
