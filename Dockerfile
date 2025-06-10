FROM python:3.11.10-slim-bullseye

# Set environment variables to potentially help with library issues
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /code

# Copy the requirements file into the container
COPY ./requirements_api.txt /code/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /code/requirements.txt

# Copy the application code into the container
COPY ./app /code/app

# Expose the port the app runs on
EXPOSE 8000

# Command to run the Uvicorn server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]