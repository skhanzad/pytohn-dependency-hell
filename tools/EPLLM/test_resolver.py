import tempfile
import unittest

from EPLLM.memory import SuccessfulVersionMemory
from EPLLM.resolver import SnippetResolver


class StubLLM:
    def __init__(self, version='9.9.9'):
        self.version = version
        self.calls = 0

    def suggest_version(self, *args, **kwargs):
        self.calls += 1
        return self.version


class SuccessfulVersionMemoryTests(unittest.TestCase):
    def test_persists_successful_versions(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            memory = SuccessfulVersionMemory(cache_dir=cache_dir)
            memory.remember_success('requests', '3.8', '2.27.1')

            reloaded = SuccessfulVersionMemory(cache_dir=cache_dir)
            preferred = reloaded.get_preferred_version(
                'requests',
                '3.8',
                ['2.25.0', '2.27.1', '2.31.0'],
            )

            self.assertEqual(preferred, '2.27.1')


class ResolverSelectionTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.resolver = SnippetResolver(
            base_modules=self.tempdir.name,
            use_llm=False,
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def test_choose_version_prefers_success_memory(self):
        self.resolver.success_memory.remember_success(
            'requests', '3.8', '2.27.1'
        )

        chosen = self.resolver._choose_version(
            package='requests',
            versions=['2.25.0', '2.27.1', '2.31.0'],
            python_version='3.8',
            iteration=0,
        )

        self.assertEqual(chosen, '2.27.1')

    def test_choose_version_uses_deterministic_before_llm(self):
        self.resolver.use_llm = True
        self.resolver.llm = StubLLM()

        chosen = self.resolver._choose_version(
            package='numpy',
            versions=['1.0.0', '2.0.0', '3.0.0'],
            python_version='3.8',
            iteration=2,
            excluded={'3.0.0'},
            current_version='3.0.0',
            prefer_older=True,
        )

        self.assertEqual(chosen, '2.0.0')
        self.assertEqual(self.resolver.llm.calls, 0)


if __name__ == '__main__':
    unittest.main()
