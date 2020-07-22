import tensorflow as tf
import numpy as np

class LayerWithConstantParameters(tf.keras.layers.Layer):
    """Layer with constant parameters added at initialization."""
    
    def add_parameter(self, value, name, dtype=None):
        if dtype is None:
            dtype = self.dtype
        if isinstance(value, int) or isinstance(value, float):
            value = np.array([value])
        return self.add_weight(shape=value.shape,
                              initializer=tf.keras.initializers.Constant(value),
                              trainable=False,
                              name="param/" + name,
                              dtype=dtype)


class ObservationPreprocessingLayer(LayerWithConstantParameters):
    """Normalizing observations as in
    https://github.com/HumanCompatibleAI/multiagent-competition/blob/72c342c4178cf189ea336a743f74e445faa6183a/gym_compete/policy.py#L77
    
    Args:
        obs_mean: mean of observation (to subtract)
        obs_std: std of observation (to divide by)
        clip_value: clip observations after normalization
        
    Returns:
        Callable which does the processing.
    """
    
    def __init__(self, obs_mean, obs_std, clip_value):
        super(ObservationPreprocessingLayer, self).__init__()
        #print('obsmean', type(obs_mean), obs_mean.shape)
        #print('tf eager', tf.executing_eagerly())
        self.mean = self.add_parameter(name='mean', value=obs_mean)
        self.std = self.add_parameter(name='std', value=obs_std)
        self.clip = self.add_parameter(name='clip', value=clip_value)

    def call(self, inputs):
        #print(inputs, self.mean, self.std, self.clip)
        inputs = tf.cast(inputs, tf.float32)
        return tf.clip_by_value((inputs - self.mean) / self.std, -self.clip, self.clip)
    
class ValuePostprocessingLayer(LayerWithConstantParameters):
    """Normalizing values as in
    https://github.com/HumanCompatibleAI/multiagent-competition/blob/72c342c4178cf189ea336a743f74e445faa6183a/gym_compete/policy.py#L128
    
    Args:
        value_mean: mean value to add
        value_std: multiply NN output by this
        
    Returns:
        Callable which does the processing.
    """
    def __init__(self, value_mean, value_std):
        super(ValuePostprocessingLayer, self).__init__()
        self.mean = self.add_parameter(name='mean', value=value_mean)
        self.std = self.add_parameter(name='std', value=value_std)

    def call(self, inputs):
        return inputs * self.std + self.mean
    
class DiagonalNormalSamplingLayer(tf.keras.layers.Layer):
    """Sampling from a Diagonal Normal Distribution."""
    def call(self, inputs):
        # inputs have shape (-1, dim, 2)
        assert len(inputs.shape) == 3, "Expected 3d tensor got shape %s" % inputs.shape
        assert inputs.shape[2] == 2, "Expect mean/std, got shape %s" % inputs.shape
        means = inputs[:, :, 0]
        logstds = inputs[:, :, 1]
        out = tf.random.normal(shape=tf.shape(inputs)[:2],
                               mean=means, stddev=tf.exp(logstds))
        return out
    
class UnconnectedVariableLayer(tf.keras.layers.Layer):
    """Layer which outputs a trainable variable on a call."""
    def __init__(self, shape, name):
        super(UnconnectedVariableLayer, self).__init__(name=name)
        self.v = self.add_weight(
            shape=(1,) + shape, initializer=tf.keras.initializers.Zeros(),
            trainable=True, name="var/" + name)
        
    def call(self, inputs):
        return tf.repeat(self.v, axis=0, repeats=tf.shape(inputs)[0])