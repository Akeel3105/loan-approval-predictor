# Use official Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY model/ model/
COPY streamlit_app/ streamlit_app/
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set Streamlit to run
EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
