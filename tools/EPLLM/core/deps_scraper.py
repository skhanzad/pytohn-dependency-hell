import os
import sys
import importlib.util
import sysconfig
import requests


class DepsScraper:
    def __init__(self, logging=False) -> None:
        self.logging = logging

    def is_module_in_standard_library(self, module_name):
        if module_name in sys.builtin_module_names:
            return True
        elif module_name in ('io', 'stringio', 'os'):
            return True
        module_spec = importlib.util.find_spec(module_name)
        if module_spec is None or module_spec.origin is None:
            return False
        std_lib_path = sysconfig.get_paths()['stdlib']
        return module_spec.origin.startswith(std_lib_path)

    def is_package_on_pypi(self, package_name):
        pypi_url = f"https://pypi.org/pypi/{package_name}/json"
        try:
            response = requests.get(pypi_url)
            response.raise_for_status()
            if self.is_module_in_standard_library(package_name):
                return False
            return True
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                return False
            raise

    def append_to_list(self, list, dep):
        if dep not in list:
            list.append(dep)
        return list

    def block_quote(self, block, line):
        if '"""' in line:
            block = not block
        return block

    def clean_deps(self, dep_list):
        imports = []
        for dep in dep_list:
            if dep:
                if dep.istitle():
                    pass
                elif dep[0].isdigit():
                    pass
                else:
                    if not self.is_module_in_standard_library(dep):
                        imports = self.append_to_list(imports, dep)
        return imports

    def find_word_in_file(self, file_path, target_word, folders):
        imports = []
        block_quote = False
        try:
            with open(file_path, 'r') as f:
                for line_number, line in enumerate(f, start=1):
                    block_quote = self.block_quote(block_quote, line)
                    if not block_quote and target_word in line and '#' not in line:
                        stripped = line.strip().split(' ')
                        for i in range(len(stripped)):
                            if i == 0 and stripped[i] == 'import':
                                imports = self.append_to_list(imports, stripped[i + 1])
                            elif i > 0 and stripped[i] == 'import':
                                imports = self.append_to_list(imports, stripped[i - 1])
        except (FileNotFoundError, Exception) as e:
            if self.logging:
                print(f"find_word_in_file: {e}")
        return imports
