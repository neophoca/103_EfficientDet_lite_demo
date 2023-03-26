FROM python:3.8

WORKDIR /demo
COPY requirements.txt .
COPY . .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade setuptools
RUN pip install .

EXPOSE 8501
CMD ["python", "demo/demo.py", "--server.enableCORS=false"]
#CMD ["streamlit", "run", "demo/app.py", "--server.enableCORS=false"]