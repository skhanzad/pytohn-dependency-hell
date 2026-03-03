# Docker build/run helper (standalone EPLLM core)
import docker
from time import sleep
import os

class DockerHelper:
    def __init__(self, logging=False, image_name="", dockerfile_name="", container_name="") -> None:
        self.dockerfile_out = ""
        self.image_name = image_name
        self.dockerfile_name = dockerfile_name
        self.container_name = container_name
        try:
            self.client = docker.from_env()
        except docker.errors.DockerException as e:
            if "Permission denied" in str(e):
                print("\n" + "=" * 80)
                print("ERROR: Cannot access Docker socket - Permission Denied")
                print("=" * 80)
                raise
            raise
        self.logging = logging
        self.previous_error = {"error_message": '', "module": ''}

    def get_project_dir(self, file):
        file = os.path.normpath(file)
        split_path = file.replace("\\", "/").split("/")
        file_path = "/".join(split_path[:-1])
        file_name = split_path[-1]
        dir_name = split_path[-2] if len(split_path) >= 2 else "snippet"
        return file_path, dir_name, file_name

    def create_dockerfile(self, llm_out, file):
        project_dir, dir_name, project_file = self.get_project_dir(file)
        self.dockerfile_out = ""
        self.dockerfile_out += f"FROM python:{llm_out['python_version']}\n"
        self.dockerfile_out += "WORKDIR /app\n"
        self.dockerfile_out += "RUN [\"pip\",\"install\",\"--upgrade\",\"pip\"]\n"
        python_modules = llm_out['python_modules']
        if self.logging:
            print(python_modules)
        for module in python_modules:
            if isinstance(module, dict):
                name, version = module['module'], module['version']
            else:
                name, version = module, python_modules[module]
            ver = version[0] if isinstance(version, (list, tuple)) else version
            self.dockerfile_out += f'RUN ["pip","install","--trusted-host","pypi.python.org","--default-timeout=100","{name}=={ver}"]\n'
        self.dockerfile_out += f"COPY {project_file} /app\n"
        self.dockerfile_out += f'CMD ["python", "/app/{project_file}"]'
        self.image_name = f"test/epllm:{dir_name}_{llm_out['python_version']}"
        self.container_name = f"{dir_name}_{llm_out['python_version']}"
        self.dockerfile_name = f"Dockerfile-llm-{llm_out['python_version']}"
        out_path = os.path.join(project_dir, self.dockerfile_name)
        with open(out_path, "w") as f:
            f.write(self.dockerfile_out)

    def build_dockerfile(self, path, dockerfile=None):
        dockerfile = dockerfile or self.dockerfile_name
        project_dir, _, _ = self.get_project_dir(path)
        error_lines = ""
        for line in self.client.api.build(path=project_dir, dockerfile=dockerfile, forcerm=True, tag=self.image_name):
            decoded_line = line.decode('utf-8')
            if 'ERROR' in decoded_line or 'Could not fetch URL' in decoded_line or 'errorDetail' in decoded_line:
                error_lines += decoded_line
            if self.logging:
                print(decoded_line)
        return (True, "") if error_lines == "" else (False, error_lines)

    def delete_container(self):
        try:
            self.client.containers.get(self.container_name).remove(v=True, force=True)
        except Exception as e:
            if self.logging:
                print(e)

    def delete_image(self):
        try:
            self.client.images.remove(image=self.image_name, force=True)
        except Exception as e:
            if self.logging:
                print(e)

    def run_container_test(self):
        self.delete_container()
        logs = ''
        try:
            self.container = self.client.containers.create(self.image_name, name=self.container_name)
            self.container.start()
            sleep(10)
            while self.container.status == 'running':
                sleep(5)
            if self.logging:
                print(self.container.status)
            logs = self.container.logs()
            self.container.remove(v=True, force=True)
            self.container = None
        except docker.errors.ContainerError as e:
            if self.logging:
                print(e)
            if self.container:
                while self.container.status == 'running':
                    sleep(5)
                logs = self.container.logs()
                self.container.remove(v=True, force=True)
                self.container = None
        return logs.decode('utf-8')
