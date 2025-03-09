FROM python:3.9-slim

WORKDIR /app

COPY app.py mnist_model.h5 requirements.txt ./

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
