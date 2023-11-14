import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='naive_bayes',
    version='0.0.1',
    author='Xueyi Tan',
    author_email='xt93@georgetown.edu',
    description='naive bayes classifier',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    extras_requres={"dev": ["pytest", "flake8", "autopep8"]},
)