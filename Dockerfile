FROM python:3.8

WORKDIR /demo
COPY requirements.txt .
COPY . .
COPY .pylintrc .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade setuptools
RUN pip install .
RUN pip install flake8 pytest black isort

EXPOSE 8000
#CMD ["python", "demo/app.py", "--server.enableCORS=false"]
CMD [ "streamlit", "run", "efficientdet/app.py", "--server.port", "8000", "--browser.serverAddress", "0.0.0.0" ]