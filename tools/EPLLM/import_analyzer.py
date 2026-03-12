"""AST-based import extraction and Python version detection.

Hybrid approach combining:
- PyEGo's AST-based import parsing (deterministic, reliable)
- Syntax-based Python version inference (no LLM needed)
- Comprehensive module name -> PyPI package name mapping (from PLLM)
"""
import ast
import re
import sys
import importlib.util
import sysconfig

# ── Import name -> PyPI package name mapping ─────────────────────────────────
# Sourced from PLLM's module_link.json + common additions
MODULE_LINK = {
    "apiclient": "google-api-python-client",
    "arcpy": "arcgispro-py3",
    "attr": "attrs",
    "avahi-python": "python-avahi",
    "azure-mgmt-azure": "azure-mgmt",
    "bcc": "bcc-python",
    "bio": "biopython",
    "bs4": "beautifulsoup4",
    "c4d": "c4ddev",
    "chess": "python-chess",
    "cjson": "python-cjson",
    "cms": "django-cms",
    "cocoa": "pycocoa",
    "console": "webvirtmgr",
    "compressor": "django-compressor",
    "crypto": "pycryptodome",
    "cryptodome": "pycryptodomex",
    "cv2": "opencv-python",
    "daemon": "python-daemon",
    "dateutil": "python-dateutil",
    "debug_toolbar": "django-debug-toolbar",
    "dns": "dnspython3",
    "docx": "python-docx",
    "editor": "python-editor",
    "elementtree": "citelementtree",
    "ffmpeg": "python-ffmpeg",
    "freetype": "freetype-py",
    "generic": "pygeneric",
    "gi": "pygobject",
    "git": "gitpython",
    "github": "pygithub",
    "googleapiclient": "google-api-python-client",
    "guardian": "django-guardian",
    "haystack": "django-haystack",
    "i3": "i3-py",
    "ib": "ibpy2",
    "imagekit": "django-imagekit",
    "impala": "impyla",
    "jnius": "pyjnius",
    "jose": "python-jose",
    "llama_index": "llama-index",
    "load_dotenv": "dotenv",
    "magic": "python-magic",
    "matlab": "matlabengine",
    "mecab": "mecab-python",
    "mega": "python-mega",
    "memcache": "python-memcached",
    "messaging": "python-messaging",
    "microbit": "yotta",
    "more_itertools": "more-itertools",
    "mosquitto": "paho-mqtt",
    "multipart": "python-multipart",
    "multipart_reader": "multipart-reader",
    "mysqldb": "mysqlclient",
    "naturaltoken": "ntlk",
    "nomad": "python-nomad",
    "nova": "python-novaclient",
    "obelisk": "obelisk-py",
    "objc": "pyobjc",
    "openerp": "openerp-web",
    "openssl": "pyopenssl",
    "osgeo": "gdal",
    "paho": "paho-mqtt",
    "path": "pathpy",
    "PIL": "pillow",
    "pil": "pillow",
    "pipeline": "django-pipeline",
    "pptx": "python-pptx",
    "pygame": "pygame-ce",
    "registration": "django-registration",
    "rest_framework": "djangorestframework",
    "restful_lib": "python-rest-client",
    "securitycenter": "pysecuritycenter",
    "serial": "pyserial",
    "skimage": "scikit-image",
    "sklearn": "scikit-learn",
    "social_auth": "django-social-auth",
    "social-auth": "django-social-auth",
    "socks": "pysocks",
    "storages": "django-storages",
    "tastypie": "django-tastypie",
    "twitter": "python-twitter",
    "usb": "pyusb",
    "visa": "pyvisa",
    "watson": "td-watson",
    "web": "web-py",
    "wordpress_xmlrpc": "python-wordpress-xmlrpc",
    "xmpp": "xmpppy",
    "yaml": "pyyaml",
    "image": "pillow",
    "wx": "wxpython",
    "OpenSSL": "pyopenssl",
    "Crypto": "pycryptodome",
}

# ── Standard library modules (Python 2 & 3 combined) ────────────────────────
# Modules that should NOT be installed via pip
STDLIB_MODULES = {
    # Common across Python 2 & 3
    '__future__', '_thread', 'abc', 'aifc', 'argparse', 'ast', 'asynchat',
    'asyncore', 'atexit', 'base64', 'bdb', 'binascii', 'binhex', 'bisect',
    'builtins', 'bz2', 'calendar', 'cgi', 'cgitb', 'chunk', 'cmath', 'cmd',
    'code', 'codecs', 'codeop', 'collections', 'colorsys', 'compileall',
    'concurrent', 'configparser', 'contextlib', 'contextvars', 'copy',
    'copyreg', 'cProfile', 'crypt', 'csv', 'ctypes', 'curses', 'dataclasses',
    'datetime', 'dbm', 'decimal', 'difflib', 'dis', 'distutils', 'doctest',
    'email', 'encodings', 'enum', 'errno', 'faulthandler', 'fcntl', 'filecmp',
    'fileinput', 'fnmatch', 'formatter', 'fractions', 'ftplib', 'functools',
    'gc', 'getopt', 'getpass', 'gettext', 'glob', 'grp', 'gzip', 'hashlib',
    'heapq', 'hmac', 'html', 'http', 'idlelib', 'imaplib', 'imghdr', 'imp',
    'importlib', 'inspect', 'io', 'ipaddress', 'itertools', 'json', 'keyword',
    'lib2to3', 'linecache', 'locale', 'logging', 'lzma', 'mailbox', 'mailcap',
    'marshal', 'math', 'mimetypes', 'mmap', 'modulefinder', 'multiprocessing',
    'netrc', 'nntplib', 'numbers', 'operator', 'optparse', 'os', 'parser',
    'pathlib', 'pdb', 'pickle', 'pickletools', 'pipes', 'pkgutil', 'platform',
    'plistlib', 'poplib', 'posix', 'posixpath', 'pprint', 'profile', 'pstats',
    'pty', 'pwd', 'py_compile', 'pyclbr', 'pydoc', 'queue', 'quopri',
    'random', 're', 'readline', 'reprlib', 'resource', 'rlcompleter', 'runpy',
    'sched', 'secrets', 'select', 'selectors', 'shelve', 'shlex', 'shutil',
    'signal', 'site', 'smtpd', 'smtplib', 'sndhdr', 'socket', 'socketserver',
    'spwd', 'sqlite3', 'ssl', 'stat', 'statistics', 'string', 'stringprep',
    'struct', 'subprocess', 'sunau', 'symtable', 'sys', 'sysconfig', 'syslog',
    'tabnanny', 'tarfile', 'telnetlib', 'tempfile', 'termios', 'test',
    'textwrap', 'threading', 'time', 'timeit', 'tkinter', 'token', 'tokenize',
    'trace', 'traceback', 'tracemalloc', 'tty', 'turtle', 'turtledemo',
    'types', 'typing', 'unicodedata', 'unittest', 'urllib', 'uu', 'uuid',
    'venv', 'warnings', 'wave', 'weakref', 'webbrowser', 'winreg', 'winsound',
    'wsgiref', 'xdrlib', 'xml', 'xmlrpc', 'zipapp', 'zipfile', 'zipimport',
    'zlib',
    # Python 2 only stdlib
    'BaseHTTPServer', 'CGIHTTPServer', 'ConfigParser', 'Cookie', 'HTMLParser',
    'Queue', 'SimpleHTTPServer', 'SimpleXMLRPCServer', 'SocketServer',
    'StringIO', 'Tkinter', 'UserDict', 'UserList', 'UserString',
    '__builtin__', 'cPickle', 'cStringIO', 'commands', 'cookielib',
    'copy_reg', 'dumbdbm', 'dummy_thread', 'dummy_threading', 'htmlentitydefs',
    'htmllib', 'httplib', 'repr', 'sets', 'thread', 'urllib2', 'urlparse',
    'xmlrpclib',
    # Common built-in names that aren't packages
    'asyncio', 'pathlib', 'typing', 'collections', 'concurrent',
    # Additional well-known stdlib
    'abc', 'array', 'binascii', 'cmath', 'code', 'copy', 'ctypes', 'curses',
    'dis', 'email', 'errno', 'gc', 'getopt', 'grp', 'gzip', 'hashlib',
    'hmac', 'http', 'io', 'json', 'locale', 'logging', 'math', 'mimetypes',
    'multiprocessing', 'os', 'pickle', 'platform', 'pprint', 'queue',
    'random', 're', 'select', 'shelve', 'shlex', 'shutil', 'signal',
    'socket', 'sqlite3', 'ssl', 'string', 'struct', 'subprocess', 'sys',
    'tempfile', 'textwrap', 'threading', 'time', 'traceback', 'types',
    'unittest', 'urllib', 'uuid', 'warnings', 'weakref', 'xml', 'zipfile',
    'zlib',
}

# Google App Engine modules (not installable via pip)
GAE_MODULES = {
    'google.appengine', 'webapp2', 'google.appengine.ext',
    'google.appengine.api', 'google.appengine.runtime',
}


def is_stdlib(module_name):
    """Check if a module is part of the standard library."""
    if not module_name or not isinstance(module_name, str):
        return True

    top = module_name.split('.')[0]

    # Check our comprehensive list first
    if top in STDLIB_MODULES:
        return True

    # Check GAE modules
    if module_name in GAE_MODULES or top in GAE_MODULES:
        return True

    # Check built-in modules
    if top in sys.builtin_module_names:
        return True

    # Try importlib check for the running Python version
    try:
        spec = importlib.util.find_spec(top)
        if spec is not None and spec.origin is not None:
            std_path = sysconfig.get_paths()['stdlib']
            # Ensure it's truly stdlib, not site-packages/dist-packages
            if (spec.origin.startswith(std_path)
                    and 'site-packages' not in spec.origin
                    and 'dist-packages' not in spec.origin):
                return True
    except (ModuleNotFoundError, ValueError):
        pass

    return False


def map_import_to_package(import_name):
    """Map an import name to its PyPI package name.

    Uses MODULE_LINK for known mappings, otherwise lowercases the name.
    Returns None for stdlib/uninstallable modules.
    """
    if not import_name:
        return None

    top = import_name.split('.')[0]

    if is_stdlib(top):
        return None

    # Check known mappings (case-sensitive first, then lowercase)
    if top in MODULE_LINK:
        return MODULE_LINK[top]
    if top.lower() in MODULE_LINK:
        return MODULE_LINK[top.lower()]

    # Default: use the import name as the package name (lowercase)
    return top.lower()


def analyze_imports(filepath):
    """Extract third-party imports from a Python file using AST.

    Returns a list of PyPI package names (not import names).
    Falls back to regex when AST parsing fails (e.g., Python 2 syntax).

    Handles special patterns like:
    - flask.ext.login -> flask-login (old Flask extension style)
    - Second-level imports (e.g., google.cloud -> google-cloud)
    """
    try:
        with open(filepath, 'r', errors='replace') as f:
            source = f.read()
    except (IOError, OSError):
        return []

    raw_imports = set()       # top-level: {'flask', 'numpy', ...}
    full_imports = set()      # full paths: {'flask.ext.login', 'keras.layers', ...}

    # Try AST parsing first (works for valid Python 3 code)
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name:
                        raw_imports.add(alias.name.split('.')[0])
                        full_imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:  # skip relative imports
                    raw_imports.add(node.module.split('.')[0])
                    full_imports.add(node.module)
    except SyntaxError:
        # Fall back to regex for Python 2 code
        raw_imports = _regex_extract_imports(source)

    # Also do regex extraction to catch imports AST might miss
    raw_imports |= _regex_extract_imports(source)

    # Extract full import paths from regex too
    full_imports |= _regex_extract_full_imports(source)

    # Map import names to package names, filtering stdlib
    packages = set()
    for imp in raw_imports:
        pkg = map_import_to_package(imp)
        if pkg:
            packages.add(pkg)

    # Handle special multi-level import patterns (PyEGo-inspired)
    for full_imp in full_imports:
        extra_pkgs = _map_full_import(full_imp)
        packages.update(extra_pkgs)

    return sorted(packages)


def _map_full_import(full_import):
    """Map multi-level imports to their PyPI packages.

    Handles patterns like:
    - flask.ext.login -> flask-login
    - flask.ext.sqlalchemy -> flask-sqlalchemy
    - google.cloud.storage -> google-cloud-storage
    """
    packages = set()
    parts = full_import.split('.')

    # Flask extensions: flask.ext.X -> flask-X
    if len(parts) >= 3 and parts[0] == 'flask' and parts[1] == 'ext':
        ext_name = parts[2]
        packages.add(f"flask-{ext_name}")

    # Google Cloud: google.cloud.X -> google-cloud-X
    if len(parts) >= 3 and parts[0] == 'google' and parts[1] == 'cloud':
        packages.add(f"google-cloud-{parts[2]}")

    # Google API: google.oauth2 -> google-auth
    if len(parts) >= 2 and parts[0] == 'google' and parts[1] == 'oauth2':
        packages.add("google-auth")

    # Zope interface: zope.interface -> zope.interface (dotted package)
    if len(parts) >= 2 and parts[0] == 'zope':
        packages.add(f"zope.{parts[1]}")

    return packages


def _regex_extract_full_imports(source):
    """Extract full dotted import paths from source code."""
    imports = set()
    for line in source.split('\n'):
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        # from X.Y.Z import ...
        m = re.match(r'^from\s+([\w.]+)\s+import\s+', stripped)
        if m:
            imports.add(m.group(1))
        # import X.Y.Z
        m = re.match(r'^import\s+([\w.]+)', stripped)
        if m:
            imports.add(m.group(1))
    return imports


def _regex_extract_imports(source):
    """Extract import names using regex (works for Python 2/3 code)."""
    imports = set()
    in_block_comment = False

    for line in source.split('\n'):
        stripped = line.strip()

        # Track block comments
        if '"""' in stripped or "'''" in stripped:
            count = stripped.count('"""') + stripped.count("'''")
            if count % 2 == 1:
                in_block_comment = not in_block_comment
            continue
        if in_block_comment:
            continue

        # Skip comments
        if stripped.startswith('#'):
            continue

        # Match: import module1, module2
        m = re.match(r'^import\s+(.+)', stripped)
        if m:
            mods = m.group(1).split('#')[0]  # remove inline comments
            for mod in mods.split(','):
                mod = mod.strip().split(' as ')[0].strip()
                if mod and not mod.startswith('.'):
                    imports.add(mod.split('.')[0])
            continue

        # Match: from module import something
        m = re.match(r'^from\s+(\S+)\s+import\s+', stripped)
        if m:
            mod = m.group(1)
            if not mod.startswith('.'):
                imports.add(mod.split('.')[0])

    return imports


def detect_python_versions(filepath):
    """Detect likely Python versions from syntax features.

    Returns a list of Python version strings ordered by likelihood.
    Uses heuristic scoring based on syntax patterns found in the code.

    This replaces LLM-based version detection with deterministic analysis
    inspired by PyEGo's feature-based approach.
    """
    try:
        with open(filepath, 'r', errors='replace') as f:
            source = f.read()
    except (IOError, OSError):
        return ['3.8', '3.7', '3.9', '2.7']

    py2_score = 0
    py3_score = 0
    min_py3_version = None  # minimum required Python 3.x version

    # ── Python 2 indicators ──────────────────────────────────────────────
    py2_patterns = [
        (r'(?m)^\s*print\s+["\']', 3),           # print "hello"
        (r'(?m)^\s*print\s+[a-zA-Z_]', 2),       # print variable
        (r'\braw_input\s*\(', 3),                  # raw_input()
        (r'\bxrange\s*\(', 3),                     # xrange()
        (r'\.has_key\s*\(', 3),                    # dict.has_key()
        (r'\bexecfile\s*\(', 3),                   # execfile()
        (r'\burllib2\b', 3),                       # urllib2
        (r'\bhttplib\b', 2),                       # httplib
        (r'\bConfigParser\b', 2),                  # ConfigParser (Py2 casing)
        (r'\bQueue\.Queue\b', 2),                  # Queue.Queue
        (r'\bcPickle\b', 2),                       # cPickle
        (r'\bcStringIO\b', 2),                     # cStringIO
        (r'\bBaseHTTPServer\b', 2),                # BaseHTTPServer
        (r'\bSimpleHTTPServer\b', 2),              # SimpleHTTPServer
        (r'\burlparse\b', 2),                      # urlparse module
        (r'\bxmlrpclib\b', 2),                     # xmlrpclib
        (r'\bSocketServer\b', 2),                  # SocketServer
        (r'\bthread\b(?!ing)', 1),                 # thread module
        (r'except\s+\w+\s*,\s*\w+\s*:', 3),       # except Exception, e:
        (r'\braise\s+\w+\s*,\s*', 2),             # raise Exception, message
        (r'from\s+__future__\s+import\s+print_function', 2),  # Py2 compat
        (r'\.iteritems\s*\(', 2),                  # dict.iteritems()
        (r'\.itervalues\s*\(', 2),                 # dict.itervalues()
        (r'\.iterkeys\s*\(', 2),                   # dict.iterkeys()
        (r'unicode\s*\(', 2),                      # unicode()
        (r'\blong\s*\(', 1),                       # long()
        (r'from\s+StringIO\s+import', 2),          # from StringIO import
        (r'from\s+UserDict\s+import', 2),          # from UserDict import
    ]

    for pattern, weight in py2_patterns:
        if re.search(pattern, source):
            py2_score += weight

    # ── Python 3 indicators ──────────────────────────────────────────────
    py3_patterns = [
        (r'\basync\s+def\b', 2, '3.5'),            # async def
        (r'\bawait\s+', 2, '3.5'),                  # await
        (r'f"[^"]*\{', 2, '3.6'),                   # f-strings
        (r"f'[^']*\{", 2, '3.6'),                   # f-strings
        (r':\s*\w+\s*=', 1, '3.6'),                 # type hints
        (r'\bsecrets\b', 1, '3.6'),                 # secrets module
        (r'\bfrom\s+dataclasses\s+import', 2, '3.7'),  # dataclasses
        (r'\basyncio\.run\s*\(', 1, '3.7'),         # asyncio.run()
        (r':=', 2, '3.8'),                           # walrus operator
        (r'\bfrom\s+typing\s+import', 1, '3.5'),    # typing module
        (r'\bfrom\s+pathlib\s+import', 1, '3.4'),   # pathlib
        (r'\bfrom\s+enum\s+import', 1, '3.4'),      # enum
        (r'\bfrom\s+contextlib\s+import\s+.*suppress', 1, '3.4'),
        (r'\bfrom\s+concurrent\.futures\s+import', 1, '3.2'),
        (r'\bnonlocal\s+\w', 1, '3.0'),             # nonlocal keyword
        (r'\byield\s+from\b', 1, '3.3'),            # yield from
        (r'from\s+urllib\.request\s+import', 2, '3.0'),  # urllib.request
        (r'from\s+urllib\.parse\s+import', 2, '3.0'),    # urllib.parse
        (r'from\s+configparser\s+import', 2, '3.0'),     # configparser
        (r'from\s+queue\s+import', 1, '3.0'),        # queue module
        (r'from\s+socketserver\s+import', 1, '3.0'),
        (r'from\s+http\.server\s+import', 1, '3.0'),
        (r'from\s+xmlrpc\s+import', 1, '3.0'),
        (r'from\s+io\s+import\s+StringIO', 1, '3.0'),
        (r'\bprint\s*\(', 1, '3.0'),                # print() function
        (r'match\s+\w+\s*:', 1, '3.10'),             # match/case
        (r'case\s+\w+\s*:', 1, '3.10'),
    ]

    for pattern, weight, min_ver in py3_patterns:
        if re.search(pattern, source):
            py3_score += weight
            if min_py3_version is None or min_ver > min_py3_version:
                min_py3_version = min_ver

    # ── Determine version candidates ─────────────────────────────────────
    if py2_score > py3_score and py2_score >= 3:
        # Strong Python 2 signal
        return ['2.7', '3.6', '3.7', '3.8']
    elif py2_score > 0 and py3_score == 0:
        # Some Python 2 signal, no Python 3 signal
        return ['2.7', '3.8', '3.7', '3.6']

    # Python 3 - determine best minor version
    if min_py3_version:
        major_minor = min_py3_version
        minor = int(major_minor.split('.')[1])
        # Build candidate list centered around detected minimum
        candidates = []
        # Preferred version (detected or one above)
        preferred = f"3.{max(minor, 6)}"
        candidates.append(preferred)
        # Add surrounding versions
        for offset in [1, -1, 2, -2]:
            v = minor + offset
            if 4 <= v <= 12:
                ver = f"3.{v}"
                if ver not in candidates:
                    candidates.append(ver)
        if '2.7' not in candidates:
            candidates.append('2.7')
        return candidates

    # Default: try common versions
    return ['3.8', '3.7', '3.9', '3.6', '2.7']


def detect_is_python2(filepath):
    """Quick check if a file is likely Python 2 code."""
    versions = detect_python_versions(filepath)
    return versions[0] == '2.7'
