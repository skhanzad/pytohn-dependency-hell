import subprocess
import json
import requests
import time


class GithubCruiserCore:
    def __init__(self, logging=False) -> None:
        self.logging = logging

    def load_json_from_file(self, file_path):
        with open(file_path) as f:
            return json.load(f)
