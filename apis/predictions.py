import pandas as pd
from flask_restplus import fields, Resource, reqparse, Namespace

from happytal_recsys import HappytalRecSys

api = Namespace('predictions', description='predict ratings')

# Request Parser
parser = reqparse.RequestParser(bundle_errors=True)
parser.add_argument(
    'new_data',
    type=list,
    required=True,
    location='json',
    help="new_item ?")

# Fields documentation
doc_fields = api.model('new_rating', {
    "product_rating": fields.Integer(attribute='product_rating',
                                     required=True,
                                  description="Note de satisfaction donnée par le client sur le produit"),
    "product_price": fields.Float(attribute='product_price',
                                      required=True,
                                      description="Prix en euros du produit"),
    "product_category": fields.String(attribute='product_category',
                                        required=True,
                                        description="Catégorie du produit"),
    "product_id": fields.Integer(attribute='product_id',
                                   required=True,
                                   description="Identifiant du produit"),
    "etat": fields.String(attribute='etat'),
})

predictions_fields = api.model('Predictions', {
    'result': fields.String(required=True,
                            description='Nouvelles notes'),
    'list_items': fields.Nested(doc_fields,
                             required=True,
                             description='liste des nouveaux items'),
})


@api.route('/')
class TypoDetection(Resource):
    @api.marshal_with(predictions_fields, code=201)
    @api.response(201, 'Predictions successfully computed.',
                  predictions_fields)
    @api.expect(parser)
    def post(self):
        """
        Returns mislabelled document

        Use this method to check if your documents contains mislabelled
        documents.

        * Send a JSON object with the documents metadata in the request body.

        ```
        {
          "documents": [
          {
            "":""
          },
          {
            "":""
          },
         ]
        }
        ```

        """
        args = parser.parse_args()
        result = self.get_result(args)
        return result, 201

    @staticmethod
    def get_result(args):
        df = pd.DataFrame(args['documents'])

        happytal_recsys = HappytalRecSys()
        outliers = happytal_recsys.predict_model(df)

        if outliers.shape[0] > 0:
            result = "pli KO"

        else:
            result = "pli OK"

        return {'result': result,
                'docs_ko': outliers.to_dict(orient='records')}
