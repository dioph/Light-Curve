import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='litecurve',
    version='0.0.1',
    author='Eduardo Nunes',
    author_email='diofanto.nunes@gmail.com',
    description='Python3 Lightcurve Astrophysical Analysis',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/dioph/litecurve',
    packages=setuptools.find_packages(),
    classifiers=(
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ),
)
