# Use an official Python 3.8 image as a base
FROM python:3.8

# Install Jupyter and JupyterLab
RUN pip install jupyter jupyterlab

# Copy the requirements.txt file into the container
COPY requirements.txt /tmp/

# Update package list and install required packages
RUN apt-get update && apt-get install -y \
    texlive-xetex \
    texlive-fonts-recommended \
    texlive-plain-generic \
    dvipng \
    cm-super \
    graphviz \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy Pandoc .deb package
COPY dependencies/pandoc.deb /tmp/
RUN dpkg -i /tmp/pandoc.deb || apt-get install -y -f

# Install nbconvert and additional dependencies
RUN pip install nbconvert
RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy notebook files into the container
COPY . /home/jovyan/work

# Set the working directory
WORKDIR /home/jovyan/work

# Expose the Jupyter port
EXPOSE 8888

# Run JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.token=''", "--ServerApp.root_dir='/home/jovyan/work'", "--ServerApp.open_browser=True"]

