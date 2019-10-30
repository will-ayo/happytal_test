from setuptools import setup

setup(
    name="happytal_webservice",
    version="0.1.0",
    author="William NGAUV",
    author_email="william.ngauv@gadz.org",
    description="Recommender System WebService for happytal",
    license="MIT",
    # Include additional files into the package
    # include_package_data=True,
    # Details
    url="http://github.com/will-ayo/happytal_webservice",

    long_description=open("README.md").read(),

    # Dependent packages (distributions)
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: Unix"

    ],
)

