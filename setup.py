import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    reqs = f.read().splitlines()

setuptools.setup(
    name='shadowing',
    version='1.0',
    author='Rudy Morel',
    author_email='rmorel@flatironinstitute.org',
    description='Path shadowing Monte Carlo',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/RudyMorel/shadowing',
    install_requires=reqs,
    packages=setuptools.find_packages()
)