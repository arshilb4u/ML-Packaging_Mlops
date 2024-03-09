from sklearn.pipeline import Pipeline
from prediction_model.config import config
from prediction_model.preprocessing import preprocessing as pp
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

classification_pipeline = Pipeline(
    [
        ('Mean Imputation', pp.MeanImputation(variable = config.NUM_FEATURES)),
        ('Mode Imputation' , pp.ModeImputation(variable = config.CAT_FEATURES)),
        ('Domain Processing' , pp.DomainProcessing(variable_to_modify = config.FEATURES_TO_MODIFY,variable_to_add = config.FEATURE_TO_ADD)),
        ('Drop Feature', pp.DropColumns(variable_to_drop = config.DROP_FEATURES)),
        ('LabelEncoders', pp.CustomLabelEncoders(variable = config.FEATURES_TO_ENCODE)),
        ('LogTransform', pp.LogTransform(variable = config.LOG_FEATURES)),
        ('MinMaxScale', MinMaxScaler()),
        ('LogisticRegression', LogisticRegression(random_state=0))
    ]
)