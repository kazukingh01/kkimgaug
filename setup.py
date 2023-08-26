from setuptools import setup, find_packages

packages = find_packages(
        where='.',
        include=['kkimgaug*']
)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='kkimgaug',
    version='3.0.0',
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
        'albumentations==1.3.1',
        'numpy==1.25.2',
        'opencv-python==4.8.0.76',
        'more-itertools==10.1.0'
    ],
    python_requires='>=3.11.2'
)
