import tensorflow as tf
import numpy as np
import baselines.common.tf_util as U
from baselines.a2c.utils import fc
from tensorflow.python.ops import math_ops
import tensorflow_probability as tfp


class Pd(object):
    """
    A particular probability distribution
    """

    def flatparam(self):
        raise NotImplementedError

    def mode(self):
        raise NotImplementedError

    def neglogp(self, x):
        # Usually it's easier to define the negative logprob
        raise NotImplementedError

    def kl(self, other):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def logp(self, x):
        return - self.neglogp(x)

    def get_shape(self):
        return self.flatparam().shape

    @property
    def shape(self):
        return self.get_shape()

    def __getitem__(self, idx):
        return self.__class__(self.flatparam()[idx])


class PdType(object):
    """
    Parametrized family of probability distributions
    """

    def pdclass(self):
        raise NotImplementedError

    def pdfromflat(self, flat):
        return self.pdclass()(flat)

    def pdfromlatent(self, latent_vector):
        raise NotImplementedError

    def param_shape(self):
        raise NotImplementedError

    def sample_shape(self):
        raise NotImplementedError

    def sample_dtype(self):
        raise NotImplementedError

    def param_placeholder(self, prepend_shape, name=None):
        return tf.placeholder(dtype=tf.float32, shape=prepend_shape + self.param_shape(), name=name)

    def sample_placeholder(self, prepend_shape, name=None):
        return tf.placeholder(dtype=self.sample_dtype(), shape=prepend_shape + self.sample_shape(), name=name)


class CategoricalPdType(PdType):
    def __init__(self, ncat):
        self.ncat = ncat

    def pdclass(self):
        return CategoricalPd

    def pdfromlatent(self, latent_vector, init_scale=1.0, init_bias=0.0):
        pdparam = fc(latent_vector, 'pi', self.ncat, init_scale=init_scale, init_bias=init_bias)
        return self.pdfromflat(pdparam), pdparam

    def param_shape(self):
        return [self.ncat]

    def sample_shape(self):
        return []

    def sample_dtype(self):
        return tf.int32


class MultiCategoricalPdType(PdType):
    def __init__(self, nvec):
        self.ncats = nvec

    def pdclass(self):
        return MultiCategoricalPd

    def pdfromflat(self, flat):
        return MultiCategoricalPd(self.ncats, flat)

    def param_shape(self):
        return [sum(self.ncats)]

    def sample_shape(self):
        return [len(self.ncats)]

    def sample_dtype(self):
        return tf.int32


class DiagGaussianPdType(PdType):
    def __init__(self, size):
        self.size = size

    def pdclass(self):
        return DiagGaussianPd

    def param_shape(self):
        return [2 * self.size]

    def sample_shape(self):
        return [self.size]

    def sample_dtype(self):
        return tf.float32

    def pdfromflat(self, flat, transformation=None, inv_transformation=None, tanh=False):
        return self.pdclass()(flat, transformation, inv_transformation, tanh)

    def pdfromlatent(self, latent_vector, init_scale=1.0, init_bias=0.0, init_logstd=0.0, transformation=None,
                     inv_transformation=None, tanh=False, trainable_variance=False):
        mean = fc(latent_vector, 'pi', self.size, init_scale=init_scale, init_bias=init_bias)
        if init_logstd is None:
            logstd = tf.get_variable(name='pi/logstd', shape=[1, self.size], initializer=tf.zeros_initializer(),
                                     trainable=trainable_variance)
        elif np.isscalar(init_logstd):
            logstd = tf.get_variable(name='pi/logstd', initializer=tf.constant([init_logstd] * self.size),
                                     trainable=trainable_variance)
        else:
            logstd = tf.get_variable(name='pi/logstd', initializer=tf.constant(init_logstd),
                                     trainable=trainable_variance)
        pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
        return self.pdfromflat(pdparam, transformation, inv_transformation, tanh=tanh), mean


class BetaPdType(PdType):
    def __init__(self, size):
        self.size = size

    def pdclass(self):
        return BetaMultivariatePd

    def param_shape(self):
        return [self.size]

    def sample_shape(self):
        return [self.size]

    def sample_dtype(self):
        return tf.float32

    def pdfromflat(self, flat):
        return self.pdclass()(flat)

    def pdfromlatent(self, latent_vector, init_scale=1.0, init_bias=0.0, trainable_variance=False):
        pdparam = fc(latent_vector, 'pi', self.size, init_scale=init_scale, init_bias=init_bias)
        mode = self.pdfromflat(pdparam).mode()
        return self.pdfromflat(pdparam), mode


class GaussianVectorPdType(PdType):
    def __init__(self, size):
        self.size = size

    def pdclass(self):
        return GaussianVectorPd

    def param_shape(self):
        return [2 * self.size]

    def sample_shape(self):
        return [self.size]

    def sample_dtype(self):
        return tf.float32


class MultiGaussianVectorPdType(PdType):
    def __init__(self, size):
        self.size = size

    def pdclass(self):
        return MultiGaussianVectorPd

    def param_shape(self):
        return [2 * self.size]

    def sample_shape(self):
        return [self.size]

    def sample_dtype(self):
        return tf.float32

    def pdfromflat(self, flat, layer_lens):
        return self.pdclass()(flat, layer_lens)


class CholeskyGaussianPdType(PdType):
    def __init__(self, size):
        self.size = size

    def pdclass(self):
        return CholeskyGaussianPd

    def param_shape(self):
        return [self.size * (self.size + 3) // 2]

    def sample_shape(self):
        return [self.size]

    def sample_dtype(self):
        return tf.float32

    def pdfromflat(self, flat):
        return self.pdclass()(flat, self.size)


class BernoulliPdType(PdType):
    def __init__(self, size):
        self.size = size

    def pdclass(self):
        return BernoulliPd

    def param_shape(self):
        return [self.size]

    def sample_shape(self):
        return [self.size]

    def sample_dtype(self):
        return tf.int32

    def pdfromlatent(self, latent_vector, init_scale=1.0, init_bias=0.0):
        pdparam = fc(latent_vector, 'pi', self.size*2, init_scale=init_scale, init_bias=init_bias)
        return self.pdfromflat(pdparam), pdparam


# WRONG SECOND DERIVATIVES
# class CategoricalPd(Pd):
#     def __init__(self, logits):
#         self.logits = logits
#         self.ps = tf.nn.softmax(logits)
#     @classmethod
#     def fromflat(cls, flat):
#         return cls(flat)
#     def flatparam(self):
#         return self.logits
#     def mode(self):
#         return U.argmax(self.logits, axis=-1)
#     def logp(self, x):
#         return -tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, x)
#     def kl(self, other):
#         return tf.nn.softmax_cross_entropy_with_logits(other.logits, self.ps) \
#                 - tf.nn.softmax_cross_entropy_with_logits(self.logits, self.ps)
#     def entropy(self):
#         return tf.nn.softmax_cross_entropy_with_logits(self.logits, self.ps)
#     def sample(self):
#         u = tf.random_uniform(tf.shape(self.logits))
#         return U.argmax(self.logits - tf.log(-tf.log(u)), axis=-1)

class CategoricalPd(Pd):
    def __init__(self, logits):
        self.logits = logits

    def flatparam(self):
        return self.logits

    def mode(self):
        return tf.argmax(self.logits, axis=-1)

    @property
    def mean(self):
        return tf.nn.softmax(self.logits)

    def neglogp(self, x):
        # return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=x)
        # Note: we can't use sparse_softmax_cross_entropy_with_logits because
        #       the implementation does not allow second-order derivatives...
        if x.dtype in {tf.uint8, tf.int32, tf.int64}:
            # one-hot encoding
            x_shape_list = x.shape.as_list()
            logits_shape_list = self.logits.get_shape().as_list()[:-1]
            for xs, ls in zip(x_shape_list, logits_shape_list):
                if xs is not None and ls is not None:
                    assert xs == ls, 'shape mismatch: {} in x vs {} in logits'.format(xs, ls)

            x = tf.one_hot(x, self.logits.get_shape().as_list()[-1])
        else:
            # already encoded
            assert x.shape.as_list() == self.logits.shape.as_list()

        return tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.logits,
            labels=x)

    def kl(self, other):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        a1 = other.logits - tf.reduce_max(other.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        ea1 = tf.exp(a1)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        z1 = tf.reduce_sum(ea1, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=-1)

    def entropy(self):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)

    def sample(self):
        u = tf.random_uniform(tf.shape(self.logits), dtype=self.logits.dtype)
        return tf.argmax(self.logits - tf.log(-tf.log(u)), axis=-1)

    def renyi(self, other, alpha=2.):
        assert isinstance(other, CategoricalPd)
        ea0 = tf.exp(self.logits)
        ea1 = tf.exp(other.logits)
        softmax0 = ea0 / tf.reduce_sum(ea0, axis=-1, keepdims=True)
        softmax1 = ea1 / tf.reduce_sum(ea1, axis=-1, keepdims=True)
        return (1 / (alpha - 1)) * tf.log(tf.reduce_sum(tf.pow(softmax0, alpha) * tf.pow(softmax1, 1 - alpha), axis=-1))

    @classmethod
    def fromflat(cls, flat):
        return cls(flat)


class MultiCategoricalPd(Pd):
    def __init__(self, nvec, flat):
        self.flat = flat
        self.categoricals = list(map(CategoricalPd, tf.split(flat, nvec, axis=-1)))

    def flatparam(self):
        return self.flat

    def mode(self):
        return tf.cast(tf.stack([p.mode() for p in self.categoricals], axis=-1), tf.int32)

    def neglogp(self, x):
        return tf.add_n([p.neglogp(px) for p, px in zip(self.categoricals, tf.unstack(x, axis=-1))])

    def kl(self, other):
        return tf.add_n([p.kl(q) for p, q in zip(self.categoricals, other.categoricals)])

    def entropy(self):
        return tf.add_n([p.entropy() for p in self.categoricals])

    def sample(self):
        return tf.cast(tf.stack([p.sample() for p in self.categoricals], axis=-1), tf.int32)

    @classmethod
    def fromflat(cls, flat):
        raise NotImplementedError


class DiagGaussianPd(Pd):
    def __init__(self, flat, transformation=None, inv_transformation=None, tanh=False):
        self.flat = flat
        mean, logstd = tf.split(axis=len(flat.shape) - 1, num_or_size_splits=2, value=flat)
        self.mean = mean
        self.logstd = logstd
        self.std = tf.exp(logstd)
        self.logN = None
        if transformation is None or inv_transformation is None:
            transformation = inv_transformation = lambda x: x
        elif not tanh:
            self.logN = tfp.distributions.LogNormal(self.mean, self.std)
        self.transformation = transformation
        self.inv_transformation = inv_transformation
        self.tanh = tanh

    def flatparam(self):
        return self.flat

    def mode(self):
        return self.transformation(self.mean)

    def neglogp(self, x):
        x = self.inv_transformation(x)
        neglogp = 0.5 * tf.reduce_sum(tf.square((x - self.mean) / self.std), axis=-1) \
                  + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[-1]) \
                  + tf.reduce_sum(self.logstd, axis=-1)
        if self.tanh:
            neglogp += tf.reduce_sum(tf.log(1 - tf.pow(tf.tanh(x), 2)), axis=-1)
        return neglogp

    def independent_logps(self, x):
        self.inv_transformation(x)
        logps = - (0.5 * tf.square((x - self.mean) / self.std)
                   + 0.5 * np.log(2. * np.pi)
                   + self.logstd)
        if self.tanh:
            logps -= tf.log(1 - tf.pow(tf.tanh(x), 2))
        return logps

    def kl(self, other):
        assert isinstance(other, DiagGaussianPd)
        return tf.reduce_sum(other.logstd - self.logstd + (tf.square(self.std) + tf.square(self.mean - other.mean)) / (
                    2.0 * tf.square(other.std)) - 0.5, axis=-1)

    def entropy(self):
        if self.logN is not None:
            return self.logN.entropy()
        return tf.reduce_sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)

    def sample(self):
        x = self.mean + self.std * tf.random_normal(tf.shape(self.mean))
        return self.transformation(x)

    def sample_symmetric(self):
        noise = tf.random_normal(tf.shape(self.mean))
        return self.transformation((self.mean + self.std * noise, self.mean - self.std * noise))

    def renyi(self, other, alpha=2.):
        tol = 1e-45
        assert isinstance(other, DiagGaussianPd)
        var_alpha = alpha * tf.square(other.std) + (1. - alpha) * tf.square(self.std)
        return alpha / 2. * tf.reduce_sum(tf.square(self.mean - other.mean) / (var_alpha + tol), axis=-1) - \
               1. / (2 * (alpha - 1)) * (tf.log(tf.reduce_prod(var_alpha, axis=-1) + tol) -
                                         tf.log(tf.reduce_prod(tf.square(self.std), axis=-1) + tol) * (1 - alpha)
                                         - tf.log(tf.reduce_prod(tf.square(other.std), axis=-1) + tol) * alpha)

    @classmethod
    def fromflat(cls, flat):
        return cls(flat)


class BetaPd(Pd):
    def __init__(self, flat):
        self.flat = flat
        alpha, beta = tf.split(axis=-1, num_or_size_splits=2, value=flat)
        self.alpha = 1. + tf.exp(alpha)
        self.beta = 1. + tf.exp(beta)
        self.beta_dist = tfp.distributions.Beta(self.alpha, self.beta)

    def flatparam(self):
        return self.flat

    def mean(self):
        return self.beta_dist.mean()

    def mode(self):
        return self.beta_dist.mode()

    #def gamma(self, x):
        #return tf.exp(tf.lgamma(x))

    #def beta_func(self):
        #return (self.gamma(self.alpha) * self.gamma(self.beta))/self.gamma(self.alpha + self.beta)

    #def digamma(self, x):
        #return tf.math.digamma(x)

    def entropy2(self):
        return tf.math.log(self.beta_func()) - (self.alpha - 1)*self.digamma(self.alpha) \
             - (self.beta - 1)*self.digamma(self.beta) + (self.alpha + self.beta - 2)*self.digamma(self.alpha + self.beta)

    def entropy(self):
        return self.beta_dist.entropy()


    def cross_entropy(self, other):
        assert isinstance(other, BetaPd)
        return self.beta_dist.cross_entropy(other.beta_dist)
        # return tf.math.log(other.beta_func()) - (other.alpha - 1)*self.digamma(self.alpha) \
        #      - (other.beta - 1)*self.digamma(self.beta) + (other.alpha + other.beta - 2)*self.digamma(self.alpha + self.beta)

    def kl(self, other):
        assert isinstance(other, BetaPd)
        return self.beta_dist.kl_divergence(other.beta_dist)
        #return self.cross_entropy(other) - self.entropy()

    def kl2(self, other):
        assert isinstance(other, BetaPd)
        return self.cross_entropy(other) - self.entropy()

    def neglogp(self, x):
        log_p = self.beta_dist.log_prob(x)
        return - log_p
        #return tf.math.log(self.beta_func()) - (self.alpha - 1)*tf.math.log(x) - (self.beta - 1)*tf.math.log(1 - x)

    def sample(self):
        s = self.beta_dist.sample()
        return s
        #return tf.random_gamma((), self.alpha)/(tf.random_gamma((), self.alpha) + tf.random_gamma((), self.beta))

    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

class BetaMultivariatePd(Pd):
    def __init__(self, flat):
        self.flat = flat
        self.flat1, self.flat2 = tf.split(axis=- 1, num_or_size_splits=2, value=self.flat)
        self.dist1 = BetaPd(self.flat1)
        self.dist2 = BetaPd(self.flat2)

    def flatparam(self):
        return self.flat

    def beta_params(self):
        return [self.dist1.alpha, self.dist1.beta, self.dist2.alpha, self.dist2.beta]

    def entropy(self):
        return self.dist1.entropy() + self.dist2.entropy()

    def entropy2(self):
        return self.dist1.entropy2() + self.dist2.entropy2()

    def kl(self, other):
        assert isinstance(other, BetaMultivariatePd)
        return self.dist1.kl(other.dist1) + self.dist2.kl(other.dist2)

    def kl2(self, other):
        assert isinstance(other, BetaMultivariatePd)
        return self.dist1.kl2(other.dist1) + self.dist2.kl2(other.dist2)

    def neglogp(self, x):
        first = tf.expand_dims(x[:, 0], axis=-1)
        second = tf.expand_dims(x[:, 1], axis=-1)
        neglogp = self.dist1.neglogp(first) + self.dist2.neglogp(second)
        return neglogp

    def sample(self):
        s1 = self.dist1.sample()
        s2 = self.dist2.sample()
        sample = tf.concat([s1, s2], axis=1)
        return sample

    def mean(self):
        m = tf.concat([self.dist1.mean(), self.dist2.mean()], axis=1)
        return m

    def mode(self):
        m = tf.concat([self.dist1.mode(), self.dist2.mode()], axis=1)
        return m

class GaussianVectorPd(Pd):
    def __init__(self, flat):
        self.flat = flat
        mean, logstd = tf.split(axis=len(flat.shape) - 1, num_or_size_splits=2, value=flat)
        self.mean = mean
        self.logstd = logstd
        self.std = tf.exp(logstd)

    def flatparam(self):
        return self.flat

    def mode(self):
        return self.mean

    def neglogp(self, x):
        return 0.5 * tf.square((x - self.mean) / self.std) \
               + 0.5 * np.log(2.0 * np.pi) \
               + self.logstd

    def independent_logps(self, x):
        return - (0.5 * tf.square((x - self.mean) / self.std)
                  + 0.5 * np.log(2. * np.pi)
                  + self.logstd)

    def kl(self, other):
        assert isinstance(other, DiagGaussianPd)
        return other.logstd - self.logstd + (tf.square(self.std) + tf.square(self.mean - other.mean)) / (
                    2.0 * tf.square(other.std)) - 0.5

    def entropy(self):
        return self.logstd + .5 * np.log(2.0 * np.pi * np.e)

    def sample(self):
        return self.mean + self.std * tf.random_normal(tf.shape(self.mean))

    def sample_symmetric(self):
        noise = tf.random_normal(tf.shape(self.mean))
        return (self.mean + self.std * noise, self.mean - self.std * noise)

    def renyi(self, other, alpha=2.):
        tol = 1e-24
        assert isinstance(other, GaussianVectorPd)
        var_alpha = alpha * tf.square(other.std) + (1. - alpha) * tf.square(self.std)
        return alpha / 2. * tf.square(self.mean - other.mean) / (var_alpha + tol) + \
               other.logstd - self.logstd + \
               0.5 / (alpha - 1) * (2 * other.logstd - tf.log(var_alpha + tol))

    @classmethod
    def fromflat(cls, flat):
        return cls(flat)


class MultiGaussianVectorPd(Pd):
    def __init__(self, flat, layer_lens):
        self.flat = flat
        self.layer_lens = layer_lens
        mean, logstd = tf.split(axis=len(flat.shape) - 1, num_or_size_splits=2, value=flat)
        self.flat_mean = mean
        self.flat_logstd = logstd
        self.means = tf.split(mean, layer_lens)
        self.logstds = tf.split(logstd, layer_lens)

    def flatparam(self):
        return self.flat

    def mode(self):
        return tf.concat(self.means, axis=0)

    def neglogp(self, x_flat):
        xs = tf.split(x_flat, self.layer_lens, axis=1)
        return tf.stack([0.5 * tf.reduce_sum(tf.square((x - selfmean) / tf.exp(selflogstd)), axis=-1) \
                         + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[-1]) \
                         + tf.reduce_sum(selflogstd, axis=-1) for selfmean, selflogstd, x
                         in zip(self.means, self.logstds, xs)], axis=1)

    def kl(self, other):
        assert isinstance(other, MultiGaussianVectorPd)
        return tf.stack([tf.reduce_sum(
            otherlogstd - selflogstd + (tf.exp(2 * selflogstd) + tf.square(selfmean - othermean)) / (
                        2.0 * tf.exp(2 * otherlogstd)) - 0.5, axis=-1)
                         for selfmean, selflogstd, othermean, otherlogstd in
                         zip(self.means, self.logstds, other.means, other.logstds)],
                        axis=0)

    def entropy(self):
        return tf.stack(
            [tf.reduce_sum(selflogstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1) for selflogstd in self.logstds],
            axis=0)

    def sample(self):
        return self.flat_mean + tf.exp(self.flat_logstd) * tf.random_normal(tf.shape(self.flat_mean))

    def sample_symmetric(self):
        noise = tf.random_normal(tf.shape(self.flat_mean))
        return (self.flat_mean + tf.exp(self.flat_logstd) * noise, self.flat_mean - tf.exp(self.flat_logstd) * noise)

    def renyi(self, other, alpha=2.):
        tol = 1e-45
        assert isinstance(other, MultiGaussianVectorPd)
        renyis = []
        for selfmean, selflogstd, othermean, otherlogstd in zip(self.means, self.logstds, other.means, other.logstds):
            var_alpha = alpha * tf.exp(2 * otherlogstd) + (1. - alpha) * tf.exp(2 * selflogstd)
            renyis.append(alpha / 2. * tf.reduce_sum(tf.square(selfmean - othermean) / (var_alpha + tol), axis=-1) - \
                          1. / (2 * (alpha - 1)) * (tf.log(tf.reduce_prod(var_alpha, axis=-1) + tol) -
                                                    tf.log(tf.reduce_prod(tf.exp(2 * selflogstd), axis=-1) + tol) * (
                                                                1 - alpha)
                                                    - tf.log(
                        tf.reduce_prod(tf.exp(otherlogstd), axis=-1) + tol) * alpha))
        return tf.stack(renyis, axis=0)


# Single distribution: use for higher order policy or on single state
class CholeskyGaussianPd(Pd):
    """d-dimensional multivariate Gaussian distribution"""

    def __init__(self, flat, size):
        """Params:
            flat: the d(d+3)/2 necessary parameters in a flat tensor.
                For a d-dimensional Gaussian, first d parameters for the mean,
                then d(d+1)/2 parameters for the nonzero elements of upper triangular
                standard deviation matrix L. The diagonal entries are exponentiated, while the others are kept
                as is. The covariance matrix is then LL^T. Initializing all standard deviation parameters to
                zero produces an identity covariance matrix.
            size: dimension d
        """
        l = flat.shape[-1].value
        if size * (size + 3) != 2 * l:
            raise ValueError('Multivariate Gaussian: parameter size does not match dimension')
        self.size = size

        # Build std matrix
        mask = np.triu_indices(size)
        mask = mask[0] * size + mask[1]
        mask = np.expand_dims(mask, -1)
        mean, std_params = tf.split(axis=0,
                                    num_or_size_splits=[size, l - size],
                                    value=flat)
        self.mean = mean
        self.std = tf.scatter_nd(mask, std_params, shape=[size ** 2])
        self.std = tf.reshape(self.std, shape=[size, size])
        self.std = (tf.ones(shape=[self.size, self.size]) - tf.eye(self.size)) * self.std + \
                   tf.eye(self.size) * tf.exp(self.std)
        self.cov = tf.matmul(self.std, self.std, transpose_b=True)
        # Distribution properties
        self.log_det_cov = 2 * tf.reduce_sum(tf.log(tf.matrix_diag_part(self.std)))
        self._entropy = 0.5 * (self.size +
                               self.size * tf.log(tf.constant(2 * np.pi)) +
                               self.log_det_cov)

    def flatparam(self):
        return self.flat

    def mode(self):
        return self.mean

    def neglogp(self, x):
        delta = tf.expand_dims(x - self.mean, axis=-1)
        stds = 0 * delta + self.std
        half_quadratic = tf.matrix_triangular_solve(stds,
                                                    delta, lower=False)
        quadratic = tf.matmul(half_quadratic, half_quadratic, transpose_a=True)

        return 0.5 * (self.log_det_cov + quadratic + self.size * tf.log(2 * tf.constant(np.pi)))

    def kl(self, other):
        assert isinstance(other, CholeskyGaussianPd)
        assert self.size == other.size
        std_mix = tf.matrix_triangular_solve(other.std, self.std, lower=False)
        trace_mix = tf.trace(tf.matmul(std_mix, std_mix, transpose_b=True))
        delta = self.mean - other.mean
        delta = tf.expand_dims(delta, axis=-1)
        half_quadratic = tf.matrix_triangular_solve(other.std, delta, lower=False)
        quadratic = tf.matmul(half_quadratic, half_quadratic, transpose_a=True)
        return 0.5 * (self.log_det_cov - other.log_det_cov + trace_mix +
                      quadratic - self.size)

    def entropy(self):
        return self._entropy

    def sample(self):
        noise = tf.random_normal([self.size])
        return self.mean + tf.einsum('n,nm->m', noise, self.std)

    def sample_symmetric(self):
        noise = tf.random_normal([self.size])
        return (self.mean + tf.einsum('n,nm->m', noise, self.std),
                self.mean - tf.einsum('n,nm->m', noise, self.std))

    # TODO: test
    def renyi(self, other, alpha=2.):
        assert isinstance(other, CholeskyGaussianPd)
        assert self.size == other.size
        delta = self.mean - other.mean
        delta = tf.expand_dims(delta, axis=-1)
        mix_cov = alpha * other.cov + (1 - alpha) * self.cov
        mix_std = tf.cholesky(mix_cov)
        half_quadratic = tf.matrix_triangular_solve(mix_std, delta, lower=True)
        quadratic = tf.matmul(half_quadratic, half_quadratic, transpose_a=True)
        log_det_mix = 2 * tf.reduce_sum(tf.log(tf.matrix_diag_part(mix_std)))
        return 0.5 * alpha * quadratic - 1. / (2 * (alpha - 1)) * (log_det_mix -
                                                                   (1 - alpha) * self.log_det_cov -
                                                                   alpha * other.log_det_cov)

    @classmethod
    def fromflat(cls, flat):
        return cls(flat)


class BernoulliPd(Pd):
    def __init__(self, logits):
        self.logits = logits
        self.ps = tf.sigmoid(logits)

    def flatparam(self):
        return self.logits

    @property
    def mean(self):
        return self.ps

    def mode(self):
        return tf.round(self.ps)

    def neglogp(self, x):
        return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=tf.to_float(x)),
                             axis=-1)

    def kl(self, other):
        return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=other.logits, labels=self.ps),
                             axis=-1) - tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.ps), axis=-1)

    def entropy(self):
        return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.ps), axis=-1)

    def sample(self):
        u = tf.random_uniform(tf.shape(self.ps))
        return tf.to_float(math_ops.less(u, self.ps))

    @classmethod
    def fromflat(cls, flat):
        return cls(flat)


def make_pdtype(ac_space, beta):
    from gym import spaces
    if isinstance(ac_space, spaces.Box):
        assert len(ac_space.shape) == 1
        if beta is True:
            return BetaPdType(ac_space.shape[0])
        else:
            return DiagGaussianPdType(ac_space.shape[0])
    elif isinstance(ac_space, spaces.Discrete):
        return CategoricalPdType(ac_space.n)
    elif isinstance(ac_space, spaces.MultiDiscrete):
        return MultiCategoricalPdType(ac_space.nvec)
    elif isinstance(ac_space, spaces.MultiBinary):
        return BernoulliPdType(ac_space.n)
    else:
        raise NotImplementedError


def shape_el(v, i):
    maybe = v.get_shape()[i]
    if maybe is not None:
        return maybe
    else:
        return tf.shape(v)[i]


@U.in_session
def test_probtypes():
    np.random.seed(0)

    pdparam_diag_gauss = np.array([-.2, .3, .4, -.5, .1, -.5, .1, 0.8])
    diag_gauss = DiagGaussianPdType(pdparam_diag_gauss.size // 2)  # pylint: disable=E1101
    validate_probtype(diag_gauss, pdparam_diag_gauss)

    pdparam_categorical = np.array([-.2, .3, .5])
    categorical = CategoricalPdType(pdparam_categorical.size)  # pylint: disable=E1101
    validate_probtype(categorical, pdparam_categorical)

    nvec = [1, 2, 3]
    pdparam_multicategorical = np.array([-.2, .3, .5, .1, 1, -.1])
    multicategorical = MultiCategoricalPdType(nvec)  # pylint: disable=E1101
    validate_probtype(multicategorical, pdparam_multicategorical)

    pdparam_bernoulli = np.array([-.2, .3, .5])
    bernoulli = BernoulliPdType(pdparam_bernoulli.size)  # pylint: disable=E1101
    validate_probtype(bernoulli, pdparam_bernoulli)


def validate_probtype(probtype, pdparam):
    N = 100000
    # Check to see if mean negative log likelihood == differential entropy
    Mval = np.repeat(pdparam[None, :], N, axis=0)
    M = probtype.param_placeholder([N])
    X = probtype.sample_placeholder([N])
    pd = probtype.pdfromflat(M)
    calcloglik = U.function([X, M], pd.logp(X))
    calcent = U.function([M], pd.entropy())
    Xval = tf.get_default_session().run(pd.sample(), feed_dict={M: Mval})
    logliks = calcloglik(Xval, Mval)
    entval_ll = - logliks.mean()  # pylint: disable=E1101
    entval_ll_stderr = logliks.std() / np.sqrt(N)  # pylint: disable=E1101
    entval = calcent(Mval).mean()  # pylint: disable=E1101
    assert np.abs(entval - entval_ll) < 3 * entval_ll_stderr  # within 3 sigmas

    # Check to see if kldiv[p,q] = - ent[p] - E_p[log q]
    M2 = probtype.param_placeholder([N])
    pd2 = probtype.pdfromflat(M2)
    q = pdparam + np.random.randn(pdparam.size) * 0.1
    Mval2 = np.repeat(q[None, :], N, axis=0)
    calckl = U.function([M, M2], pd.kl(pd2))
    klval = calckl(Mval, Mval2).mean()  # pylint: disable=E1101
    logliks = calcloglik(Xval, Mval2)
    klval_ll = - entval - logliks.mean()  # pylint: disable=E1101
    klval_ll_stderr = logliks.std() / np.sqrt(N)  # pylint: disable=E1101
    assert np.abs(klval - klval_ll) < 3 * klval_ll_stderr  # within 3 sigmas
    print('ok on', probtype, pdparam)
