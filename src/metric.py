import tensorflow as tf
import pandas as pd


def auc_roc(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value


if __name__ == '__main__':
    sample_true = pd.Series([0, 1, 1, 0, 1, 0, 0])
    sample_pred = pd.Series([0, 1, 0, 0, 1, 0, 1])

    print(auc_roc(sample_true, sample_pred))
