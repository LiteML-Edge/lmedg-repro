import os, random
import tensorflow as tf
import numpy as np

def set_global_seed(seed: int = 42):
    # Python and hashing
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Deterministic TensorFlow operations where supported
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # when using a GPU
    # RNGs
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # Threads: reduce sources of nondeterminism from parallel execution
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except Exception:
        pass



