# ** GENERATED FILE, DO NOT EDIT! **
import re

from setuptools import find_packages, setup

with open('kif_llm_store/__init__.py') as fp:
    text = fp.read()
    (VERSION,) = re.findall(r"__version__\s*=\s*'(.*)'", text)
with open('README.md', 'r') as fp:
    README = fp.read()

setup(
    name='kif-llm-store',
    version=VERSION,
    description='A LLM Store for knowledge integration framework',
    long_description=README,
    long_description_content_type='text/markdown',
    author='IBM',
    author_email='mmachado@ibm.com',
    url='https://github.com/IBM/kif-llm-store',
    license='Apache-2.0',
    python_requires='>=3.9',
    packages=find_packages(exclude=['tests', 'tests.*']),
    package_dir={'kif_llm_store': 'kif_llm_store'},
    install_requires=[
        'requests',
        'kif-lib @ git+https://github.com/IBM/kif@iswc-lm-kbc-2024',
        'wikipedia',
        'nltk',
        'openai',
        'pandas',
        'python-dotenv',
        'sentence-transformers',
        'torch',
        'transformers',
        'types-aiofiles',
        'types-beautifulsoup4',
        'types-requests',
        'yfinance',
        'accelerate',
        'aiofiles',
        'beautifulsoup4',
        'httpx',
        'huggingface-hub',
        'ibm-generative-ai',
    ],
    keywords=[
        'KIF',
        'Knowledge Graph',
        'LLM',
        'Semantic Web',
        'Knowledge Integration',
        'Wikidata',
    ],
    extras_require={
        'docs': [
            'myst_parser',
            'pydata_sphinx_theme',
        ],
        'tests': [
            'ipywidgets',
            'flake8',
            'isort',
            'mypy',
            'pyright',
            'pytest',
            'pytest-cov',
            'pytest-mypy',
            'tox',
        ],
    },
    zip_safe=False,
)
