import ctypes
import inspect
import threading

import pandas as pd
from flask import abort
from flask_restplus import fields, Resource, Namespace, reqparse

from happytal_recsys import HappytalRecSys

pd.options.mode.chained_assignment = None


# Initializing RESTful API
api = Namespace('training', description="model training")
clf = HappytalRecSys()

class TrainingThread(threading.Thread):
    """
    Initialize the training inside a new thread
    """
    def __init__(self, name, daemon):
        threading.Thread.__init__(self)
        self.name = name
        self.daemon = daemon

    def run(self):
        print("Training started.")
        print("ThreadID : %s " % self.ident)
        df = pd.read_csv('order_data.csv',
                         sep=';')

        clf.fit_model(df)



# Fields documentation
training_fields = api.model(
    'Training',
    {'status': fields.String(required=False, description='Training status')})
training_thread = api.model(
    'Status',
    {'process_alive_count': fields.Integer(required=True,
                                           description='Threads count'),
     'process_name': fields.String(required=True,
                                   description='Threads name'),
     'training_process_id': fields.Integer(required=False,
                                           description='Training thread id'),
     'progress': fields.String(required=False,
                               description='Training progression')})

# Request Parser
parser = reqparse.RequestParser(bundle_errors=True)
parser.add_argument(
    'order',
    type=str,
    required=True,
    location='json',
    help="")


def _async_raise(tid, exctype):
    '''
    Raises an exception in the threads with id tid
    :param tid: Thread ID
    :param exctype: Exception Type
    :return: raise
    '''
    if not inspect.isclass(exctype):
        raise TypeError("Only types can be raised (not instances)")
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid),
                                                  ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # "if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, 0)
        raise SystemError("PyThreadState_SetAsyncExc failed")


@api.route('/')
class TypoDetection(Resource):
    @api.marshal_with(training_fields)
    @api.response(200, 'Training status')
    @api.expect(parser)
    def post(self):
        """
        Launch model training

        Train your model with preloaded data.
        Pass "start" to start training your model or "stop" to cancel it.


        ```
        {
        "order" : "string"
        }
        ```


        """
        args = parser.parse_args()
        if args['order'].lower() == 'start':
            for t in threading.enumerate():
                if isinstance(t, TrainingThread):
                    return {'status': 'Training already in progress'}
            else:
                training = TrainingThread(name="Training", daemon=True)
                training.start()
                return {'status': 'Starting model training'}

        elif args['order'].lower() == 'stop':
            for t in threading.enumerate():
                if isinstance(t, TrainingThread):
                    _async_raise(t.ident, SystemExit)
                else:
                    pass
            return {'status': 'Stopping existing training thread'}

        else:
            abort(400)

    @api.marshal_with(training_thread)
    @api.response(200, 'OK')
    def get(self):
        """
        Training status


        """
        thread_list = []
        for t in threading.enumerate():
            if isinstance(t, TrainingThread):
                return {'process_alive_count': threading.active_count(),
                        'process_name': t.name,
                        'training_process_id': t.ident,
                        'progress': clf.status}
            else:
                thread_list.append(t.name)
        return {'process_alive_count': threading.active_count(),
                'process_name': thread_list}
