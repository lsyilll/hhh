SELECT 'CREATE DATABASE airflow OWNER guanye' WHERE NOT EXISTS(SELECT FROM pg_database WHERE datname='airflow')\gexec
