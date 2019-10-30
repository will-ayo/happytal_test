from flask_restplus import Api

from .predictions import api as apipred
from .training import api as apitrain

api = Api(
    title='happytal recommender system',
    version='0.1',
    description='Placeholder description'
)

api.add_namespace(apitrain)
api.add_namespace(apipred)


__version__="0.1"