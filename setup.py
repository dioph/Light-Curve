import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='litecurve',
    version='0.0.2',
    author='Eduardo Nunes',
    author_email='diofanto.nunes@gmail.com',
    license='MIT',
    description='Light Curve Analysis with Python3',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/dioph/litecurve',
    packages=setuptools.find_packages(),
    classifiers=(
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
    ),
)
