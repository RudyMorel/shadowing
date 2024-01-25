import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    reqs = f.read().splitlines()

setuptools.setup(
    name='shadowing',
    version='1.0',
    author='Rudy Morel',
    author_email='rudy.morel@ens.fr',
    description=
    'Path shadowing Monte Carlo simulation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/RudyMorel/shadowing',
    license='MIT',
    install_requires=reqs,
    packages=setuptools.find_packages())