import logging
from time import time

import pandas as pd
import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.externals import joblib
from xgboost import callback

from preprocessing import Preprocessing

logging.basicConfig(
    filename="happytal_recsys.log",
    level=logging.DEBUG,
    format='%(asctime)s - (%(threadName)-9s) %(name)s.%(funcName)s '
           '+ %(lineno)s: %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

folder_name = ""


class HappytalRecSys(object):
    def __init__(self):
        self.status = ""

    def model_tuning(self, df):
        """
        Tuning model's hyperparameters with a Bayesian Optimization
        :param df:
        :return:
        """
        start = time()

        # Passing the hyperparameters we'll seek to optimize in a function
        def xgb_evaluate(min_child_weight,
                         colsample_bytree,
                         max_depth,
                         subsample,
                         gamma,
                         alpha,
                         max_delta_step):
            params['min_child_weight'] = int(min_child_weight)
            params['colsample_bytree'] = max(min(colsample_bytree, 1), 0)
            params['max_depth'] = int(max_depth)
            params['subsample'] = max(min(subsample, 1), 0)
            params['gamma'] = max(gamma, 0)
            params['alpha'] = max(alpha, 0)
            params['max_delta_step'] = max(int(max_delta_step), 0)

            cv_result = xgb.cv(params,
                               xgtrain,
                               num_boost_round=num_rounds,
                               nfold=5,
                               seed=random_state,
                               callbacks=[callback.early_stop(50)])
            return cv_result['test-auc-mean'].values[-1]

        # Converting the dataset to a XGBoost Matrix
        xgtrain = xgb.DMatrix(df.drop('target', 1), label=df.target)
        num_rounds = 3000
        random_state = 2016
        num_iter = 25
        init_points = 5
        params = {
            'nthread': 4,
            'eta': 0.1,
            'silent': 1,
            'eval_metric': 'auc',
            'verbose_eval': True,
            'seed': random_state
        }

        xgb_bayesopt = BayesianOptimization(xgb_evaluate,
                                            {'min_child_weight': (1, 20),
                                             'colsample_bytree': (0.1, 1),
                                             'max_depth': (5, 10),
                                             'subsample': (0.5, 1),
                                             'gamma': (0, 10),
                                             'alpha': (0, 10),
                                             'max_delta_step': (0, 10),
                                             })

        logger.debug("Tuning model's hyperparameters")
        xgb_bayesopt.maximize(init_points=init_points, n_iter=num_iter)

        logger.debug("Tuning's done, best params : %s"
                     % xgb_bayesopt.res['max']['max_params'])

        # Updating our hyperparameters
        params.update(xgb_bayesopt.res['max']['max_params'])
        params['min_child_weight'] = int(params['min_child_weight'])
        params['max_depth'] = int(params['max_depth'])
        params['max_delta_step'] = int(params['max_delta_step'])
        logger.debug('Time elapsed : %0.3fs' % (time() - start))

        return params

    def fit_model(self, df):
        start = time()
        logger.debug('Fitting model to data')

        # Preprocessing the dataset (cleaning, reformatting)
        pp = Preprocessing()

        self.status = "preprocessing : cleaning, reformatting"

        pp.fit(df)

        train = pp.transform(df)
        logger.debug('DataFrame shape : %s, %s'
                     % (train.shape[0], train.shape[1]))


        logger.debug('Target distribution: %s'
                     % (train['product_rating'].value_counts()))

        # Tuning model's hyperparameters with a Bayesian Optimizer
        self.status = "tuning model's hyperparameters"
        params = self.model_tuning(train)

        # Fitting data to our model with updated parameters
        self.status = "fitting data to the model"
        xgbclf = xgb.XGBClassifier(**params)
        xgbclf.fit(train.drop('target', 1), train.target)
        logger.debug('Time elapsed : %0.3fs' % (time() - start))

        # Saving model
        joblib.dump(xgbclf, folder_name + "xgb.model")
        joblib.dump(pp, folder_name + 'preprocessing.model')
        logger.debug('Model for XGBoost & preprocessing data saved.')

    def predict_model(self, df_test):
        """
        Prediction on test dataset
        :param df_test:
        :return:
        """
        start = time()
        logger.debug('Computing predictions')

        # Loading files needed for the prediction
        xgbmodel = joblib.load(folder_name + "./xgb.model")
        pp = joblib.load(folder_name + './preprocessing.model')

        # Preprocessing data
        self.status = "preprocessing : cleaning, reformatting"
        test = pp.clean_df(df_test)
        test = pp.transform(test)


        # Computing predictions
        self.status = "computing predictions"
        test['product_rating'] = xgbmodel.predict(test)

        logger.debug('Predictions done.')
        logger.debug('Time elapsed : %0.3fs' % (time()-start))

        return test



__version__ = "0.1"
