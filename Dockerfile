# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -U pip setuptools wheel \
	&& pip install --no-cache-dir -r requirements.txt \
	&& pip install --no-cache-dir torch==2.2.2+cpu -i https://download.pytorch.org/whl/cpu

COPY . .
EXPOSE 8501

# Default command launches the dashboard; override for training
CMD ["streamlit", "run", "src/dashboard.py", "--server.address=0.0.0.0", "--server.port=8501"]

