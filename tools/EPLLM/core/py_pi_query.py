# PyPI query logic (standalone EPLLM core)
import json
import os
import re
from pathlib import Path
from datetime import datetime

from pypi_json import PyPIJSON

from .github_cruiser_core import GithubCruiserCore
from .deps_scraper import DepsScraper

REF_FILES_DIR = Path(__file__).resolve().parent / "ref_files"


class PyPIQuery:
    def __init__(self, logging=False, base_modules="./modules") -> None:
        self.date_format = '%Y-%m-%d'
        self.output_date_format = '%b %d %Y'
        self.logging = logging
        self.ghc = GithubCruiserCore(logging=False)
        self.deps = DepsScraper(logging=logging)
        self.python_versions = self.ghc.load_json_from_file(str(REF_FILES_DIR / "python_versions.json"))
        os.makedirs(base_modules, exist_ok=True)
        self.base_modules = base_modules

    def check_format(self, python_version):
        # Normalize version specifiers like ">=3.5,<4" to "3.8"
        match = re.search(r'(\d+)\.(\d+)', str(python_version))
        if match:
            python_version = f"{match.group(1)}.{match.group(2)}"
        python_version = python_version.replace('+', '')
        split_version = python_version.split('.')
        if len(split_version) == 1:
            return f"{split_version[0]}.7"
        return f"{split_version[0]}.{split_version[1] if split_version[1] != 'x' else '7'}"

    def read_module_file(self, module, python_version):
        path = f"{self.base_modules}/{module}_{python_version}.txt"
        if os.path.isfile(path):
            with open(path, 'r') as f:
                return f.read()
        module_details = {'python_version': python_version, 'python_modules': [module]}
        self.get_module_specifics(module_details)
        if os.path.isfile(path):
            with open(path, 'r') as f:
                return f.read()
        return ''

    def get_python_dates(self, python_version):
        checked_version = self.check_format(python_version)
        for idx, x in enumerate(self.python_versions):
            if checked_version in x['cycle']:
                if idx > 0:
                    next_date = datetime.strptime(self.python_versions[idx - 1]['releaseDate'], self.date_format).date()
                else:
                    next_date = datetime.now().strftime(self.date_format)
                return datetime.strptime(x['releaseDate'], self.date_format).date(), next_date, checked_version
        return None, None, checked_version

    def get_python_range(self, python_version, pyrange=2):
        checked_version = self.check_format(python_version)
        selected_python = []
        try:
            for idx, x in enumerate(self.python_versions):
                if checked_version in x['cycle']:
                    num_values_each_side = pyrange
                    start_index = max(0, idx - num_values_each_side)
                    end_index = min(len(self.python_versions), idx + num_values_each_side + 1)
                    result = self.python_versions[start_index:end_index]
                    for version in result:
                        selected_python.append(version['cycle'])
                    break
        except Exception as e:
            if self.logging:
                print(f"Unable to get Python version: {e}")
        if len(selected_python) <= 0:
            for i in range(0, pyrange + 1):
                if i == 0:
                    selected_python.append('3.8')
                elif i == 1:
                    selected_python.append('2.7')
                    selected_python.append(f'3.{8 + i}')
                else:
                    selected_python.append(f'3.{8 + i}')
                    selected_python.append(f'3.{8 - i}')
        elif '2.7' not in selected_python:
            selected_python[-1] = '2.7' if len(selected_python) > 0 else selected_python.append('2.7')
        if self.logging:
            print(selected_python)
        return selected_python

    def check_module_name(self, module_name):
        with open(REF_FILES_DIR / "module_link.json") as f:
            known_modules = json.load(f)
        module_list = []
        if isinstance(module_name, str):
            module_name = [module_name]
        for module in module_name:
            if '.' in module:
                module = module.split('.')[0]
            module = module.replace(';', '').replace(',', '')
            if module.lower() in known_modules:
                module_list.append(known_modules[module.lower()]['ref'])
            else:
                module_list.append(module.lower())
        module_list = self.deps.clean_deps(module_list)
        return module_list

    def query_module(self, module_name):
        try:
            with PyPIJSON() as client:
                return client.get_metadata(module_name)
        except Exception:
            return None

    def get_version_from_code(self, python_code):
        if 'cp' not in python_code:
            return python_code
        python_code = python_code[2:]
        return '.'.join(python_code[i] for i in range(len(python_code)))

    def find_modules(self, module_name, start_date, end_date, python_version):
        dpq = self.query_module(module_name)
        stored = []
        if not dpq:
            return stored
        modules_releases = dpq.releases
        latest_release = {'version': '', 'date': datetime.strptime('1981-10-02', self.date_format).date()}
        small_repo = len(modules_releases) <= 5
        for ele in modules_releases:
            release = modules_releases[ele]
            if len(release) > 0:
                store = None
                for release_details in release:
                    if not store and not release_details.get('yanked'):
                        upload_time = datetime.strptime(release_details['upload_time'].split('T')[0], '%Y-%m-%d').date()
                        store = {}
                        if small_repo:
                            store = {'version': ele, 'date': upload_time.strftime(self.output_date_format)}
                        if upload_time >= start_date and upload_time <= end_date:
                            store = {'version': ele, 'date': upload_time.strftime(self.output_date_format)}
                        elif self.get_version_from_code(release_details.get('python_version', '')) == python_version:
                            store = {'version': ele, 'date': upload_time.strftime(self.output_date_format)}
                        elif 'py2' in str(release_details.get('python_version', '')) and '2.' in python_version:
                            store = {'version': ele, 'date': upload_time.strftime(self.output_date_format)}
                        elif 'py3' in str(release_details.get('python_version', '')) and '3.' in python_version:
                            store = {'version': ele, 'date': upload_time.strftime(self.output_date_format)}
                        elif 'source' in str(release_details.get('python_version', '')) and len(stored) <= 20:
                            store = {'version': ele, 'date': upload_time.strftime(self.output_date_format)}
                        if upload_time >= latest_release['date']:
                            latest_release = {'version': ele, 'date': upload_time}
                if store:
                    stored.append(store)
        if len(stored) == 0:
            latest_release['date'] = latest_release['date'].strftime(self.output_date_format)
            stored.append(latest_release)
        return stored

    def get_module_specifics(self, module_details):
        if self.logging:
            print(module_details)
        start_date, end_date, python_version = self.get_python_dates(module_details['python_version'])
        if start_date is None:
            # python_version is already normalized by check_format (e.g. "3.5" from ">=3.5,<4")
            return module_details.get('python_modules', []), python_version
        python_modules = self.check_module_name(module_details['python_modules'])
        modified_modules = []

        def version_key(version):
            return [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', version)]

        for dep in python_modules:
            modules = self.find_modules(dep, start_date, end_date, python_version)
            module_versions = [m['version'] for m in modules]
            modified_modules.append(dep)
            module_versions.sort(key=version_key)
            with open(f"{self.base_modules}/{dep}_{python_version}.txt", "w") as outfile:
                outfile.write(', '.join(module_versions))
        return modified_modules, python_version
