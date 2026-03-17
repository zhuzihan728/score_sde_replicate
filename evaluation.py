# import jax
import numpy as np
from scipy import linalg
import six
import tensorflow as tf
import tensorflow_gan as tfgan
import tensorflow_hub as tfhub

INCEPTION_TFHUB = 'https://tfhub.dev/tensorflow/tfgan/eval/inception/1'
INCEPTION_OUTPUT = 'logits'
INCEPTION_FINAL_POOL = 'pool_3'
_DEFAULT_DTYPES = {
  INCEPTION_OUTPUT: tf.float32,
  INCEPTION_FINAL_POOL: tf.float32
}
INCEPTION_DEFAULT_IMAGE_SIZE = 299


def evaluation_metrics(real_stats, gen_stats, splits=10):
    g_logits, g_pool3 = gen_stats['logits'], gen_stats['pool_3']
    preds = np.exp(g_logits-np.max(g_logits, axis=1, keepdims=True))
    preds = preds/np.sum(preds, axis=1, keepdims=True)
    scores = []
    chunk_size = preds.shape[0]//splits
    for i in range(splits):
        p = preds[i*chunk_size:(i+1)*chunk_size, :]
        kl_m = np.mean(p, axis=0, keepdims=True)
        kl_d = p*(np.log(p+1e-6)-np.log(kl_m+1e-6))
        kl_avg = np.mean(np.sum(kl_d, axis=1))
        scores.append(np.exp(kl_avg))
        
    is_mean, is_std = np.mean(scores), np.std(scores)
    
    mean_gen, cov_gen = np.mean(g_pool3, axis=0), np.cov(g_pool3, rowvar=False)
    
    mean_real, cov_real = np.mean(real_stats['pool_3'], axis=0), np.cov(real_stats['pool_3'], rowvar=False)
    
    d = mean_gen - mean_real
    cov_mean, _ = linalg.sqrtm(cov_gen.dot(cov_real), disp=False)
    if not np.isfinite(cov_mean).all():
        offset = np.eye(cov_gen.shape[0])*1e-6
        cov_mean = linalg.sqrtm((cov_gen+offset).dot(cov_real+offset))
    
    cov_mean = cov_mean.real if np.iscomplexobj(cov_mean) else cov_mean

    tr = np.trace(cov_mean)
    fid_score = d.dot(d)+np.trace(cov_gen)+np.trace(cov_real)-2*tr

    return is_mean, is_std, fid_score

def get_inception_model():
    return tfhub.load(INCEPTION_TFHUB)

def load_dataset_stats(config):
    if config.data.dataset == 'cifar10':
        filename = 'stats/cifar10_stats.npz'
    else:
        raise ValueError('Dataset stats not found')
    
    with tf.io.gfile.GFile(filename, 'rb') as fin:
        return np.load(fin)
    
def get_classifier_fn(output_fields, inception_model, return_tensor=False):
    if isinstance(output_fields, six.string_types):
        output_fields = [output_fields]
    
    def classifier_fn(images):
        output = inception_model(images)
        if output_fields:
            output = {x: output[x] for x in output_fields}
        if return_tensor:
            assert len(output) == 1
            output= list(output.values())[0]
        return tf.nest.map_structure(tf.compat.v1.layers.flatten, output)
    
    return classifier_fn

@tf.function
def run_inception_jit(inputs, inception_model, num_batches=1):
    inputs = (tf.cast(inputs, tf.float32)-127.5)/127.5
    return tfgan.eval.run_classifier_fn(
        inputs, 
        num_batches = num_batches, 
        classifier_fn = get_classifier_fn(None, inception_model),
        dtypes = _DEFAULT_DTYPES
    )

@tf.function
def run_inception(input_tensor, inception_model, num_batches=1):
    res = run_inception_jit(input_tensor, inception_model, num_batches)
    return {'pool_3': res['pool_3'], 'logits': res['logits']}



