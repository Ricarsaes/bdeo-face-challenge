FROM python:3.9-slim

# Set environment variables to potentially help with library issues
ENV PYTHONUNBUFFERED=1
# If OMP issues persist even in Docker (less likely with clean base but possible if pip installs certain binaries):
# ENV KMP_DUPLICATE_LIB_OK=TRUE
# ENV OMP_NUM_THREADS=1
# ENV MKL_NUM_THREADS=1

# Set the working directory in the container
WORKDIR /code

# Copy the requirements file into the container
COPY ./requirements.txt /code/requirements.txt

# Install Python dependencies
# Using --no-cache-dir to reduce image size
# For CPU only, you can try to find CPU specific wheels for torch/torchvision if smaller
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /code/requirements.txt

# Copy the application code into the container
# This copies the entire app/ directory from host to /code/app in container
COPY ./app /code/app

# Expose the port the app runs on
EXPOSE 8000

# Command to run the Uvicorn server
# Uvicorn will look for app.main:app relative to the WORKDIR (/code)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]