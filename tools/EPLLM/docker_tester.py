"""Docker-based build and test for dependency validation.

Generates Dockerfiles, builds images, runs containers, and captures output.
Inspired by PLLM's DockerHelper but with cleaner resource management.
"""
import os
import docker
from time import sleep


class DockerTester:
    """Manages Docker image building and container execution for testing."""

    def __init__(self, logging=False):
        self.logging = logging
        try:
            self.client = docker.from_env()
        except docker.errors.DockerException as e:
            if "Permission denied" in str(e):
                print("ERROR: Cannot access Docker socket - Permission Denied")
                print("Run with Docker socket access or: chmod 666 /var/run/docker.sock")
            raise

        self.image_name = ""
        self.container_name = ""
        self.dockerfile_name = ""

    def create_dockerfile(self, snippet_path, python_version, modules):
        """Generate a Dockerfile for testing a snippet with given dependencies.

        Args:
            snippet_path: Absolute path to the Python snippet file.
            python_version: Python version string (e.g., '3.8', '2.7').
            modules: Dict of {package_name: version_string}.

        Returns:
            Path to the generated Dockerfile.
        """
        project_dir = os.path.dirname(snippet_path)
        snippet_file = os.path.basename(snippet_path)
        dir_name = os.path.basename(project_dir)

        dockerfile_content = f"FROM python:{python_version}\n"
        dockerfile_content += "WORKDIR /app\n"
        dockerfile_content += 'RUN ["pip","install","--upgrade","pip"]\n'

        for pkg, ver in modules.items():
            if ver and ver != '0.0.0':
                dockerfile_content += (
                    f'RUN ["pip","install","--trusted-host","pypi.python.org",'
                    f'"--default-timeout=100","{pkg}=={ver}"]\n'
                )

        dockerfile_content += f"COPY {snippet_file} /app\n"
        dockerfile_content += f'CMD ["python", "/app/{snippet_file}"]\n'

        # Set naming
        self.image_name = f"test/epllm:{dir_name}_{python_version}"
        self.container_name = f"epllm_{dir_name}_{python_version}"
        self.dockerfile_name = f"Dockerfile-epllm-{python_version}"

        dockerfile_path = os.path.join(project_dir, self.dockerfile_name)
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)

        return dockerfile_path

    def build(self, snippet_path):
        """Build the Docker image.

        Returns:
            (success: bool, output: str) - build result and any error output.
        """
        project_dir = os.path.dirname(snippet_path)
        error_lines = ""

        try:
            for line in self.client.api.build(
                path=project_dir,
                dockerfile=self.dockerfile_name,
                forcerm=True,
                tag=self.image_name
            ):
                decoded = line.decode('utf-8')
                if any(k in decoded for k in ('ERROR', 'errorDetail', 'Could not fetch URL')):
                    error_lines += decoded
                if self.logging:
                    print(decoded, end='')
        except docker.errors.BuildError as e:
            return False, str(e)
        except docker.errors.APIError as e:
            return False, str(e)
        except Exception as e:
            return False, str(e)

        if error_lines:
            return False, error_lines
        return True, ""

    def run(self, timeout=60):
        """Run the built container and capture output.

        Returns:
            (success: bool, output: str) - whether it ran without errors
            and the container logs.
        """
        self._cleanup_container()
        logs = ''

        try:
            container = self.client.containers.create(
                self.image_name, name=self.container_name
            )
            container.start()

            # Wait for completion with timeout
            elapsed = 0
            poll_interval = 3
            initial_wait = 5
            sleep(initial_wait)
            elapsed += initial_wait

            container.reload()
            while container.status == 'running' and elapsed < timeout:
                sleep(poll_interval)
                elapsed += poll_interval
                container.reload()

            if container.status == 'running':
                # Timed out
                container.kill()
                container.reload()

            logs = container.logs().decode('utf-8', errors='replace')
            container.remove(v=True, force=True)

        except docker.errors.ContainerError as e:
            logs = str(e)
        except docker.errors.APIError as e:
            logs = str(e)
        except Exception as e:
            logs = str(e)

        # Check for runtime errors in logs
        runtime_errors = [
            'ImportError', 'ModuleNotFoundError', 'AttributeError',
            'SyntaxError', 'NameError', 'TypeError',
        ]
        has_error = any(err in logs for err in runtime_errors)

        return not has_error, logs

    def _cleanup_container(self):
        """Remove existing container with the same name."""
        try:
            c = self.client.containers.get(self.container_name)
            c.remove(v=True, force=True)
        except (docker.errors.NotFound, docker.errors.APIError):
            pass

    def cleanup(self):
        """Remove container and image."""
        self._cleanup_container()
        try:
            self.client.images.remove(image=self.image_name, force=True)
        except (docker.errors.ImageNotFound, docker.errors.APIError):
            pass
