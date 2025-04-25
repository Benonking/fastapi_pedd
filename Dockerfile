FROM pytorch/pytorch:latest

WORKDIR /app

COPY requirements.txt .

# Install dependencies directly without upgrading pip

RUN pip install --no-cache-dir --ignore-installed -r requirements.txt


# Install any extras not in requirements.txt
#RUN pip install --no-cache-dir uvicorn[standard] Pillow requests

COPY app app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
