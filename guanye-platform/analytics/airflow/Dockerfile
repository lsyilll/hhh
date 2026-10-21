FROM apache/airflow:3.1.0-python3.12
COPY constraints.txt /tmp/guanye-constraints.txt
RUN pip install --no-cache-dir apache-airflow==3.1.0 pyarrow psycopg2-binary boto3 --constraint /tmp/guanye-constraints.txt && pip check
