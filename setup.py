try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='simplenlp',
    version='0.0.1',
    description='Simple NLP Tools',
    author='Chia Chi Chang & Willy Kuo',
    author_email='c3h3.tw@gmail.com & waitingkuo0527@gmail.com',
    packages=['simplenlp'],
    install_requires=[
        'scikit-learn',
        'numpy',
        'scipy',
        'jieba',
        'nltk'
    ],
)