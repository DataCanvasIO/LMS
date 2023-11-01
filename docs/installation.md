Installation
==

# Build

## Prerequisites

To build the LMS components, you will first need the following software packages.

**System Package**

* Python>=3.9.7
* make
* git

**PyPI Package**

* build
* setuptools

## Downloading and Building

```bash
git clone git@gitlab.datacanvas.com:APS-OpenSource/lms.git
cd lms
python setup.py bdist_wheel
```

## Installation

```bash
pip3 install dist/lms-$(VERSION)-py3-none-any.whl
```

## Setting

```bash
nohup lms_web start &
```

```bash
lms join <LMS_WEB_HOST_NAME>:18080 --token=default
nohup lmsd &
```