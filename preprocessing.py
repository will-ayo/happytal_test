from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import logging

logging.basicConfig(
    filename="preprocessing.log",
    level=logging.DEBUG,
    format='%(asctime)s %(name)s.%(funcName)s +%(lineno)s:'
           '%(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

folder_name = ""


class Preprocessing(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cols = None

    def transform(self, df):
        """

        :param df:
        :return:
        """
        logger.debug('Preprocessing transform method')
        df = self.clean_df(df)
        df = pd.concat([df, pd.DataFrame(columns=self.cols)], axis=1)
        return df

    def clean_df(self, df):
        """
        reformatting date, cleaning out data
        :param df:
        :return df:
        """
        logger.debug('clean_df method')
        return df

    def replace_value(self, df):
        """
        Cleaning misspelled words
        :param df:
        :return:
        """
        logger.debug('Replacing misspelled words')

        dict_values = {
            'var1': {
                r'(^old1.*$)|(^old2.*$)': 'new'
            },
            'var2': {
                'old': 'new'
            },
        }
        df = df.replace(to_replace=dict_values, regex=True)

        return df

    def fit(self, df):
        """
        :param df:
        :return:
        """
        logger.debug('Fitting data to preprocessing algorithm')

        self.cols = self.clean_df(df).columns

        return self


__version__ = "0.1"
