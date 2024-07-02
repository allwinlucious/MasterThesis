# Use an official Python 3.8 image as a base
FROM python:3.8

# Install Jupyter
RUN pip install jupyter

# Copy the requirements.txt file into the container
COPY requirements.txt /tmp/

# Install additional dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy your notebook files into the container
COPY . /home/jovyan/work

# Set the working directory
WORKDIR /home/jovyan/work

# Expose the Jupyter Notebook port
EXPOSE 8888

# Run Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.token=''", "--ServerApp.root_dir='/home/jovyan/work'", "--ServerApp.open_browser=True"]
