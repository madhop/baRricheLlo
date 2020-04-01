import tensorflow as tf
import numpy as np
from baselines.common import tf_util
from baselines.a2c.utils import fc
from baselines.common.distributions import make_pdtype
from baselines.common.input import observation_placeholder, encode_observation
from baselines.common.tf_util import adjust_shape
from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.common.models import get_network_builder

import gym


def build_gradient(logits, vars, ob_ph, n_actions=3):

    action_ph = tf.placeholder(tf.int32, [None], name='targets_placeholder')
    action_selected = tf.one_hot(action_ph, n_actions)
    # out = tf.reduce_sum(tf.reduce_sum(tf.log(self.logits+1e-5)*action_selected, axis=1))
    out = tf.reduce_sum(tf.log(tf.reduce_sum(logits * action_selected, axis=1)))
    gradients = tf.gradients(out, vars)
    compute_gradients = tf_util.function(
        inputs=[ob_ph, action_ph],
        outputs=gradients
    )
    return compute_gradients

class PolicyWithValue(object):
    """
    Encapsulates fields and methods for RL policy and value function estimation with shared parameters
    """

    def __init__(self, observations, action_space, latent, estimate_q=False, vf_latent=None, sess=None,
                 pos_actions=False, init_logstd=0.0, init_bias=0.0, tanh=False, beta=False,
                 trainable_variance=False, **tensors):
        """
        Parameters:
        ----------
        env             RL environment

        observations    tensorflow placeholder in which the observations will be fed

        latent          latent state from which policy distribution parameters should be inferred

        vf_latent       latent state from which value function should be inferred (if None, then latent is used)

        sess            tensorflow session to run calculations in (if None, default session is used)

        **tensors       tensorflow tensors for additional attributes such as state or mask

        """

        self.X = observations
        self.state = tf.constant([])
        self.initial_state = None
        self.__dict__.update(tensors)

        vf_latent = vf_latent if vf_latent is not None else latent

        vf_latent = tf.layers.flatten(vf_latent)
        latent = tf.layers.flatten(latent)
        self.action_space = action_space
        self.beta = beta
        self.pdtype = make_pdtype(action_space, self.beta)
        if tanh:
            def transformation(x):
                return (tf.tanh(x) + 1) * action_space.high / 2

            def inv_transformation(x):
                x = x * 2. / action_space.high
                x = x - 1.
                x = tf.clip_by_value(x, -1., 1.)
                x = tf.atanh(x)
                return x

        elif pos_actions:
            def transformation(x):
                return tf.exp(x)

            def inv_transformation(x):
                return tf.log(x)
        else:
            def transformation(x):
                return x

            def inv_transformation(x):
                return x

        if self.beta is True:
            self.pd, self.pi = self.pdtype.pdfromlatent(latent, init_scale=0.01, init_bias=init_bias)
        else:
            self.pd, self.pi = self.pdtype.pdfromlatent(latent, init_scale=0.01, init_logstd=init_logstd,
                                                        init_bias=init_bias, transformation=transformation,
                                                         inv_transformation=inv_transformation, tanh=tanh)


        self.stochastic = tf.placeholder(dtype=tf.bool, shape=())

        self.action = tf_util.switch(self.stochastic, self.pd.sample(), self.pd.mode())
        self.neglogp = self.pd.neglogp(self.action)
        #self.logits = tf.nn.softmax(self.pd.flatparam())
        self.logits = self.pd.flatparam()
        self.sess = sess
        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pi/pi')
        try:
            self.action_ph = tf.placeholder(tf.int64, [None], name='targets_placeholder')
            action_selected = tf.one_hot(self.action_ph, self.action_space)

        #out = tf.reduce_sum(tf.reduce_sum(tf.log(self.logits+1e-5)*action_selected, axis=1))
            out = tf.reduce_sum(tf.log(tf.reduce_sum(self.logits*action_selected, axis=1)))
            gradients = tf.gradients(out, self.vars)
        except:
            self.action_ph = tf.placeholder(dtype=tf.float32, shape=(None,) + action_space.shape,
                                            name='targets_placeholder')
            gradients = tf.gradients(-self.pd.neglogp(self.action_ph), self.vars)

        self.compute_gradients = tf_util.function(
            inputs=[self.X, self.action_ph],
            outputs=gradients
        )
        self.debug = tf_util.function(
            inputs=[self.X, self.action_ph],
            outputs=[gradients, self.logits, self.action_ph]
        )
        if estimate_q:
            assert isinstance(action_space, gym.spaces.Discrete)
            self.q = fc(vf_latent, 'q', action_space.n)
            self.vf = self.q
        else:
            self.vf = fc(vf_latent, 'vf', 1)
            self.vf = self.vf[:, 0]

    def _evaluate(self, variables, observation, stochastic, beta, **extra_feed):
        sess = self.sess or tf.get_default_session()
        feed_dict = {self.X: adjust_shape(self.X, observation),
                     self.stochastic: stochastic}
        for inpt_name, data in extra_feed.items():
            if inpt_name in self.__dict__.keys():
                inpt = self.__dict__[inpt_name]
                if isinstance(inpt, tf.Tensor) and inpt._op.type == 'Placeholder':
                    feed_dict[inpt] = adjust_shape(inpt, data)

        return sess.run(variables, feed_dict)

    def step(self, observation, stochastic=False, **extra_feed):
        """
        Compute next action(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        (action, value estimate, next state, negative log likelihood of the action under current policy parameters) tuple
        """

        a, v, state, neglogp, logits = self._evaluate([self.action, self.vf, self.state, self.neglogp, self.logits],
                                                      observation, stochastic, self.beta, **extra_feed)
        if state.size == 0:
            state = None
        return a, v, state, neglogp, logits

    def value(self, ob, *args, **kwargs):
        """
        Compute value estimate(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        value estimate
        """
        return self._evaluate(self.vf, ob, *args, **kwargs)

    def save(self, save_path):
        tf_util.save_state(save_path, sess=self.sess)

    def load(self, load_path):
        tf_util.load_state(load_path, sess=self.sess)

def build_policy(ob_space, ac_space, policy_network, value_network=None, pos_actions=True, normalize_observations=False,
                 estimate_q=False, init_logstd=0.0, init_bias=0.0, tanh=False, beta=True,
                 trainable_variance=False, **policy_kwargs):
    if isinstance(policy_network, str):
        network_type = policy_network
        policy_network = get_network_builder(network_type)(**policy_kwargs)

    def policy_fn(nbatch=None, nsteps=None, sess=None, observ_placeholder=None, beta=beta):

        X = observ_placeholder if observ_placeholder is not None else observation_placeholder(ob_space,
                                                                                              batch_size=nbatch)

        extra_tensors = {}

        if normalize_observations and X.dtype == tf.float32:
            encoded_x, rms = _normalize_clip_observation(X)
            extra_tensors['rms'] = rms
        else:
            encoded_x = X

        encoded_x = encode_observation(ob_space, encoded_x)

        with tf.variable_scope('pi', reuse=tf.AUTO_REUSE):
            policy_latent, recurrent_tensors = policy_network(encoded_x)

            if recurrent_tensors is not None:
                # recurrent architecture, need a few more steps
                nenv = nbatch // nsteps
                assert nenv > 0, 'Bad input for recurrent policy: batch size {} smaller than nsteps {}'.format(nbatch,
                                                                                                               nsteps)
                policy_latent, recurrent_tensors = policy_network(encoded_x, nenv)
                extra_tensors.update(recurrent_tensors)

        _v_net = value_network

        if _v_net is None or _v_net == 'shared':
            vf_latent = policy_latent
        else:
            if _v_net == 'copy':
                _v_net = policy_network
            else:
                assert callable(_v_net)

            with tf.variable_scope('vf', reuse=tf.AUTO_REUSE):
                vf_latent, _ = _v_net(encoded_x)

        policy = PolicyWithValue(
            action_space=ac_space,
            observations=X,
            latent=policy_latent,
            vf_latent=vf_latent,
            sess=sess,
            estimate_q=estimate_q,
            pos_actions=pos_actions,
            init_logstd=init_logstd,
            init_bias=init_bias,
            tanh=tanh,
            beta=beta,
            trainable_variance=trainable_variance,
            **extra_tensors
        )
        return policy

    return policy_fn


def _normalize_clip_observation(x, clip_range=[-5.0, 5.0]):
    rms = RunningMeanStd(shape=x.shape[1:])
    norm_x = tf.clip_by_value((x - rms.mean) / rms.std, min(clip_range), max(clip_range))
    return norm_x, rms
