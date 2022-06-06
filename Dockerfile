FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel


COPY requirements.txt .
RUN python -m pip install -r requirements.txt
WORKDIR /app
# COPY . /app
CMD ["main.py"]
ENTRYPOINT [ "python" ]