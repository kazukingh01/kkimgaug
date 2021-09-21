from setuptools import setup, find_packages

packages = find_packages(
        where='.',
        include=['kkimgaug*']
)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='kkimgaug',
    version='2.0.0',
    description='augmentation wrapper package for albumentations',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kazukingh01/kkimgaug",
    author='Kazuki Kume',
    author_email='kazukingh01@gmail.com',
    license='Public License',
    packages=packages,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Private License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'albumentations>=1.0.3',
        'numpy>=1.20.3',
        'opencv-python>=4.5.3.56',
        'opencv-contrib-python>=4.5.3.56',
        'more-itertools>=8.8.0'
    ],
    python_requires='>=3.7'
)
