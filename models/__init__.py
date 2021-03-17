from models.tf_v1_model import TFV1Model

from core.registry import Registry


MODEL = Registry('MODEL')


from models.q_model import *
from models.ac_model import *
from models.q_keras_model import *