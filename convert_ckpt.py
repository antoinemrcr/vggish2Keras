import requests
import numpy as np
from scipy.io import wavfile
import six
import tensorflow as tf
import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim
from pydub import AudioSegment
from scipy.io import wavfile
from vggish_keras import get_vggish_keras

checkpoint_file = 'vggish_weights.ckpt'

sess = tf.Session()
tf.Graph().as_default()
# Define the model in inference mode, load the checkpoint, and
# locate input and output tensors.
vggish_slim.define_vggish_slim(training=False)
vggish_slim.load_vggish_slim_checkpoint(sess, 'vggish_model.ckpt')
features_tensor = sess.graph.get_tensor_by_name(
    vggish_params.INPUT_TENSOR_NAME)
embedding_tensor = sess.graph.get_tensor_by_name(
    vggish_params.OUTPUT_TENSOR_NAME)

pproc = vggish_postprocess.Postprocessor('vggish_pca_params.npz')



weights = {}
operations = sess.graph.get_operations()
for op in operations:
    name = op.name
    if 'read' in name:
        name2 = name.replace('vggish/','').replace('/read','').replace('conv3/','').replace('conv4/','').replace('/fc1','')
        name2_layer, name2_type = name2.split('/')
        if name2_type == 'weights':
            weights[name2_layer] = []
            weights[name2_layer].append(sess.run(op.values())[0])

for op in operations:
    name = op.name
    if 'read' in name:
        name2 = name.replace('vggish/','').replace('/read','').replace('conv3/','').replace('conv4/','').replace('/fc1','')
        name2_layer, name2_type = name2.split('/')
        if name2_type == 'biases':
            weights[name2_layer].append(sess.run(op.values())[0])

model = get_vggish_keras()
for layer in model.layers:
    if layer.name in list(weights.keys()):
        layer.set_weights(weights[layer.name])
model.save_weights(checkpoint_file)

