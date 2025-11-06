"""
Setup configuration for vllm_budget package.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

# Read requirements
def read_requirements(filename):
    req_path = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name='vllm_budget',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='Thinking budget wrapper for vLLM with two-stage generation',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/vllm_budget',
    packages=find_packages(exclude=['tests', 'tests.*']),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.8',
    install_requires=read_requirements('requirements.txt'),
    extras_require={
        'test': read_requirements('requirements-test.txt'),
        'dev': read_requirements('requirements-dev.txt'),
    },
    keywords='vllm llm inference thinking-budget ai ml',
    project_urls={
        'Bug Reports': 'https://github.com/yourusername/vllm_budget/issues',
        'Source': 'https://github.com/yourusername/vllm_budget',
        'Documentation': 'https://github.com/yourusername/vllm_budget#readme',
    },
)