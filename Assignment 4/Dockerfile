FROM python:3.10-slim

# Set working directory in the container
WORKDIR /app

# Copy necessary files
COPY app.py score.py best_model.pkl vectorizer.pkl requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port your Flask app runs on
EXPOSE 5000

# Start the Flask app
CMD ["python", "app.py"]