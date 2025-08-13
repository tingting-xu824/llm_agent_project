FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN pip install "fastapi[standard]"

COPY . .

EXPOSE 80

CMD [ "fastapi", "run", "agents/api.py", "--port", "80" ]