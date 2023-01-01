from setuptools import setup, find_packages

setup(
  name = 'T5-rlhf-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.0.1',
  license='MIT',
  description = 'T5 + GAN + Reinforcement Learning with Human Feedback - Pytorch',
  author = 'Muhammad Faris Adi Prabowo',
  author_email = 'muhammadfarisadiprabowo1@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/mfarisadip/T5-rlhf-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention mechanism',
    'reinforcement learning',
    'human feedback',
    'gan'
  ],
  install_requires=[
    'accelerate',
    'beartype',
    'einops>=0.6',
    'torch>=1.6',
    'tqdm'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.8',
  ],
)