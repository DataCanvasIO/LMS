from setuptools import setup, find_packages

LMS_VERSION = '1.0.0'

setup(
    name='dc-lms',
    version=LMS_VERSION,
    packages=find_packages(),
    include_package_data=True,
    url='https://github.com/DataCanvasIO/LMS',
    license='Apache License 2.0',
    author='datacanvas',
    description='',
    long_description="""
        LMS（Large Model Serving） is an open source tool that provides large model services. 
        See the LMS HOME https://github.com/DataCanvasIO/LMS for details.
    """,
    long_description_content_type="text/markdown",
    entry_points={
        "console_scripts": ['lms = lms.web:main']
    },
    python_requires='>=3.9',
    classifiers=[
                 "License :: OSI Approved :: Apache Software License",
                 "Programming Language :: Python :: 3",
                 "Programming Language :: Python :: 3.9"]
)
