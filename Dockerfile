FROM python:3.8

WORKDIR /demo
COPY requirements.txt .
COPY . .
COPY .pylintrc .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade setuptools
RUN pip install .
RUN pip install flake8 pytest black==19.10b0 isort

EXPOSE 8501
CMD ["python", "demo/demo.py", "--server.enableCORS=false"]
#CMD ["streamlit", "run", "demo/app.py", "--server.enableCORS=false"]