import os
import tensorflow as tf
from module import discriminator, generator_gatedcnn, domain_classifier
from datetime import datetime
from utils import *
import numpy as np
from preprocess import *


class StarGANVC(object):

    def __init__(self,
                 num_features,
                 frames=FRAMES,
                 discriminator=discriminator,
                 generator=generator_gatedcnn,
                 classifier=domain_classifier,
                 mode='train',
                 log_dir='./log'):
        super().__init__()
        self.num_features = num_features

        self.input_shape = [None, num_features, frames, 1]
        self.label_shape = [None, SPEAKERS_NUM]

        self.mode = mode
        self.log_dir = log_dir

        self.discriminator = discriminator
        self.generator = generator_gatedcnn
        self.classifier = classifier

        self.build_model()

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        if self.mode == 'train':
            self.train_step = 0
            now = datetime.now()
            self.log_dir = os.path.join(log_dir, now.strftime('%Y%m%d-%H%M%S'))
            self.writer = tf.summary.FileWriter(self.log_dir, tf.get_default_graph())
            self.generator_summaries, self.discriminator_summaries, self.domain_classifier_summaries = self.summary()

    def build_model(self):
        # Placeholders for real training samples
        self.input_real = tf.placeholder(tf.float32, self.input_shape, name='input_real')
        self.target_real = tf.placeholder(tf.float32, self.input_shape, name='target_real')

        self.source_label = tf.placeholder(tf.float32, self.label_shape, name='source_label')
        self.target_label = tf.placeholder(tf.float32, self.label_shape, name='target_label')

        self.generated_forward = self.generator(self.input_real, self.target_label, reuse=False, scope_name='generator')
        self.generated_back = self.generator(self.generated_forward, self.source_label, reuse=True, scope_name='generator')

        #Cycle loss
        self.cycle_loss = l1_loss(self.input_real, self.generated_back)

        #Identity loss
        self.identity_map = self.generator(self.input_real, self.source_label, reuse=True, scope_name='generator')
        self.identity_loss = l1_loss(self.input_real, self.identity_map)

        self.discrimination_real = self.discriminator(self.target_real, self.target_label, reuse=False, scope_name='discriminator')

        #combine discriminator and generator
        self.discirmination = self.discriminator(self.generated_forward, self.target_label, reuse=True, scope_name='discriminator')

        self.generator_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.discirmination), logits=self.discirmination))

        # Discriminator adversial loss

        self.discirmination_fake = self.discriminator(self.generated_forward, self.target_label, reuse=True, scope_name='discriminator')

        self.discrimination_real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.discrimination_real), logits=self.discrimination_real))

        self.discrimination_fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.discirmination_fake), logits=self.discirmination_fake))

        self.discrimator_loss = self.discrimination_fake_loss + self.discrimination_real_loss

        #domain classify loss

        self.domain_out_real = self.classifier(self.target_real, reuse=False, scope_name='classifier')

        self.domain_out_fake = self.classifier(self.generated_forward, reuse=True, scope_name='classifier')

        #domain_out_xxx [batchsize, 1,1,4], need to convert label[batchsize, 4] to [batchsize, 1,1,4]
        target_label_reshape = tf.reshape(self.target_label, [-1, 1, 1, SPEAKERS_NUM])

        self.domain_fake_loss = cross_entropy_loss(self.domain_out_fake, target_label_reshape)
        self.domain_real_loss = cross_entropy_loss(self.domain_out_real, target_label_reshape)

        # self.domain_loss = self.domain_fake_loss + self.domain_real_loss

        # Place holder for lambda_cycle and lambda_identity
        self.lambda_cycle = tf.placeholder(tf.float32, None, name='lambda_cycle')
        self.lambda_identity = tf.placeholder(tf.float32, None, name='lambda_identity')
        self.lambda_classifier = tf.placeholder(tf.float32, None, name='lambda_classifier')

        self.generator_loss_all = self.generator_loss + self.lambda_cycle * self.cycle_loss + \
                                self.lambda_identity * self.identity_loss +\
                                 self.lambda_classifier * self.domain_fake_loss

        # Categorize variables because we have to optimize the three sets of the variables separately
        trainable_variables = tf.trainable_variables()
        self.discriminator_vars = [var for var in trainable_variables if 'discriminator' in var.name]
        self.generator_vars = [var for var in trainable_variables if 'generator' in var.name]
        self.classifier_vars = [var for var in trainable_variables if 'classifier' in var.name]
        # for var in self.discriminator_vars:
        #     print(var.name)
        # for var in self.generator_vars:
        #     print(var.name)
        # for var in self.classifier_vars:
        #     print(var.name)

        #optimizer
        self.generator_learning_rate = tf.placeholder(tf.float32, None, name='generator_learning_rate')
        self.discriminator_learning_rate = tf.placeholder(tf.float32, None, name='discriminator_learning_rate')
        self.classifier_learning_rate = tf.placeholder(tf.float32, None, name="domain_classifier_learning_rate")

        self.discriminator_optimizer = tf.train.AdamOptimizer(
            learning_rate=self.discriminator_learning_rate, beta1=0.5).minimize(
                self.discrimator_loss, var_list=self.discriminator_vars)

        self.generator_optimizer = tf.train.AdamOptimizer(
            learning_rate=self.generator_learning_rate, beta1=0.5).minimize(
                self.generator_loss_all, var_list=self.generator_vars)

        self.classifier_optimizer = tf.train.AdamOptimizer(learning_rate=self.classifier_learning_rate).minimize(
            self.domain_real_loss, var_list=self.classifier_vars)

        # test
        self.input_test = tf.placeholder(tf.float32, self.input_shape, name='input_test')
        self.target_label_test = tf.placeholder(tf.float32, self.label_shape, name='target_label_test')

        self.generation_test = self.generator(self.input_test, self.target_label_test, reuse=True, scope_name='generator')

    def train(self, input_source, input_target, source_label, target_label,  lambda_cycle=1.0, lambda_identity=1.0, lambda_classifier=1.0, \
    generator_learning_rate=0.0001, discriminator_learning_rate=0.0001, classifier_learning_rate=0.0001):

        generation_f, _, generator_loss, _, generator_summaries = self.sess.run(
            [self.generated_forward, self.generated_back, self.generator_loss, self.generator_optimizer, self.generator_summaries], \
            feed_dict = {self.lambda_cycle: lambda_cycle, self.lambda_identity: lambda_identity, self.lambda_classifier:lambda_classifier ,\
            self.input_real: input_source, self.target_real: input_target,\
             self.source_label:source_label, self.target_label:target_label, \
             self.generator_learning_rate: generator_learning_rate})

        self.writer.add_summary(generator_summaries, self.train_step)

        discriminator_loss, _, discriminator_summaries = self.sess.run(\
        [self.discrimator_loss, self.discriminator_optimizer, self.discriminator_summaries], \
            feed_dict = {self.input_real: input_source, self.target_real: input_target , self.target_label:target_label,\
            self.discriminator_learning_rate: discriminator_learning_rate})

        self.writer.add_summary(discriminator_summaries, self.train_step)

        domain_classifier_real_loss, _, domain_classifier_summaries = self.sess.run(\
        [self.domain_real_loss, self.classifier_optimizer, self.domain_classifier_summaries],\
        feed_dict={self.input_real: input_source, self.target_label:target_label, self.target_real:input_target, \
        self.classifier_learning_rate:classifier_learning_rate}
        )
        self.writer.add_summary(domain_classifier_summaries, self.train_step)

        self.train_step += 1

        return generator_loss, discriminator_loss, domain_classifier_real_loss

    def summary(self):
        with tf.name_scope('generator_summaries'):
            cycle_loss_summary = tf.summary.scalar('cycle_loss', self.cycle_loss)
            identity_loss_summary = tf.summary.scalar('identity_loss', self.identity_loss)

            generator_loss_summary = tf.summary.scalar('generator_loss', self.generator_loss)
            generator_summaries = tf.summary.merge([cycle_loss_summary, identity_loss_summary, generator_loss_summary])

        with tf.name_scope('discriminator_summaries'):
            discriminator_loss_summary = tf.summary.scalar('discriminator_loss', self.discrimator_loss)
            discriminator_summaries = tf.summary.merge([discriminator_loss_summary])

        with tf.name_scope('domain_classifier_summaries'):
            domain_real_loss = tf.summary.scalar('domain_real_loss', self.domain_real_loss)
            domain_fake_loss = tf.summary.scalar('domain_fake_loss', self.domain_fake_loss)
            domain_classifer_summaries = tf.summary.merge([domain_real_loss, domain_fake_loss])

        return generator_summaries, discriminator_summaries, domain_classifer_summaries

    def test(self, inputs, label):
        generation = self.sess.run(self.generation_test, feed_dict={self.input_test: inputs, self.target_label_test: label})

        return generation

    def save(self, directory, filename):
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.saver.save(self.sess, os.path.join(directory, filename))

        return os.path.join(directory, filename)

    def load(self, filepath):
        self.saver.restore(self.sess, filepath)


if __name__ == '__main__':
    starganvc = StarGANVC(36)