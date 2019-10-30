# happytal test

Happytal recommender system

### Goal
Identifying mislabelled documents using a machine learning algorithm.

#### Details
There's no algorithm as this is not the main focus of the test

##### Installation
* Install Python 3.7 first from:
 '''
 https://conda.io/miniconda.html
 '''
* Clone the repository :
> git clone http://github.com/will-ayo/happytal_test.git
* Navigate to the cloned folder and execute this to install the dependencies :
> pip install --user -r requirements.txt

##### How to run it locally
Navigate to the cloned repository folder and execute this command from the shell :
```
gunicorn --bind 0.0.0.0:5000 --threads 2 app:app
```

#### Swagger doc
After launching the server, enter this url in your browser : localhost:5000

### Dependencies
* pandas v0.21+
* numpy v1.13+
* scikit-learn v0.19+
* xgboost v0.7+
* flask v0.12+
* flask-restplus v0.10+
* gunicorn v19.7+

