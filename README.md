
# My Thesis Environment

This repository contains the environment setup for my thesis, including all dependencies and a Jupyter Notebook setup using Docker.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed on your system

## Setup Instructions

### Clone the Repository

First, clone the repository to your local machine:

```sh
git clone https://github.com/yourusername/your-repo.git
cd your-repo
```

### Build the Docker Image

Build the Docker image using the provided `Dockerfile`:

```sh
docker build -t my-thesis-env .
```

### Run the Docker Container

Run the Docker container, mapping port 8888 and mounting the current directory:

```sh
docker run -p 8888:8888 -v $(pwd):/home/jovyan/work my-thesis-env
```

### Access Jupyter Notebook

Once the container is running, Jupyter Notebook should automatically open in your default web browser. If it does not, you can manually open your browser and navigate to:

```
http://localhost:8888
```

Since token authentication is disabled, you can access the notebook directly.

### Notes

- The container runs Jupyter Notebook with root privileges, which is allowed by the `--allow-root` flag.
- Your notebooks and all other files in the repository will be accessible inside the container at `/home/jovyan/work`.

## Updating Dependencies

If you need to update the dependencies, modify the `requirements.txt` file and rebuild the Docker image:

```sh
docker build -t my-thesis-env .
```

## Troubleshooting

If you encounter any issues, please check the following:
- Ensure Docker is installed and running correctly on your system.
- Ensure that the Docker image is built without errors.

For further assistance, feel free to open an issue on this repository.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

If you have any questions or suggestions, please contact [your-email@example.com](mailto:your-email@example.com).
