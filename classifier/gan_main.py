import tensorflow as tf
import reader
import nets
import numpy as np
import params as p


with tf.variable_scope('input_batching'):
    data_folders = ["car", "dog", "graph", "human"]

    batch = reader.get_batch(10, data_folders)

    image = batch[0]
    tf.summary.image("input images", image)
    classes = tf.squeeze(batch[1])

    gen_classes = np.zeros((p.gan_samples, p.classes), dtype=np.float32)
    gen_classes[:,1] = 1

    disc_classes = np.zeros((p.gan_samples, p.classes), dtype=np.float32)
    disc_classes[:,4] = 1

    gen_classes = tf.constant(gen_classes)
    disc_classes = tf.constant(disc_classes)

    inp = tf.random_uniform(shape=[3,256,256,3])

    gen = nets.make_generator(inp)
    tf.summary.image("generated images", gen)
    image = tf.concat([gen, image], 0)

    disc_classes = tf.concat([disc_classes, classes], 0)
    gen_classes = tf.concat([gen_classes, classes], 0)

    net_output_disc = nets.make_network(image)

    gen_loss = tf.losses.softmax_cross_entropy(onehot_labels=gen_classes,
                                           logits=net_output_disc)

    disc_loss = tf.losses.softmax_cross_entropy(onehot_labels=disc_classes,
                                           logits=net_output_disc)
    diff = (disc_classes - tf.nn.softmax(net_output_disc))/2
    a = tf.abs(diff)
    class_error_disc = tf.reduce_sum(a)

    diff = (gen_classes - tf.nn.softmax(net_output_disc))/2
    a = tf.abs(diff)
    class_error_gen = tf.reduce_sum(a)


    tf.summary.scalar("SCE loss for classifier", disc_loss)
    tf.summary.scalar("SCE loss for generator", gen_loss)
    tf.summary.scalar("classification_error_discriminator", class_error_disc)
    tf.summary.scalar("classification_error_generated", class_error_gen)

with tf.name_scope('Global_step'):
    global_step = tf.Variable(0, dtype=tf.int32)


gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "input_batching/generator")
disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "input_batching/discriminator")
train_step_gen = tf.train.AdamOptimizer(0.001).minimize(gen_loss, global_step=global_step, var_list=gen_vars)
train_step_disc = tf.train.AdamOptimizer(0.001).minimize(disc_loss, global_step=global_step, var_list=disc_vars)

merged = tf.summary.merge_all()


print("Model constructed!")

sess = tf.Session()

print("Variables initialized!")
import os
import sys

net_name = 'dcgh_gan1'
folder_name = './networks/%s'%net_name
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

saver = tf.train.Saver()
writer = tf.summary.FileWriter("output/"+net_name, sess.graph)
coordinate = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coordinate)


sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
if "-new" not in sys.argv:
    print('loading network.. ', end='')

    if '-best' in sys.argv:
        saver.restore(sess, folder_name + '/best_valid.cpt')
        print('Starting from best net.. ', end='')
        print('Done.')
    else:
        saver.restore(sess, folder_name + '/latest.cpt')
        print('Starting from latest net.. ', end='')
        print('Done.')

else:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())


train_mode = "disc"

min_loss = 100
#print([n.name for n in tf.get_default_graph().as_graph_def().node])
import time
from tensorflow.python.framework import graph_util
with sess.as_default():
    while True:
        if train_mode == "disc":
            for i in range(2000):
                start_time=time.time()
                cse, c, _t = sess.run([disc_loss, class_error_disc, train_step_disc])
                #print(no, classes_actual)
                #print(cse)
                print('Disc. training set took %f seconds!'%(time.time()-start_time))
                if (i%10 == 0):
                    print("step %d, SCE loss: %g, classification error: %s" % (i, cse, c))
                    writer.add_summary(sess.run(merged), global_step=sess.run(global_step))

                if (i%100 == 99):
                    start = time.time()
                    saver.save(sess, folder_name + '/latest.cpt')
            train_mode = "gen"
        elif train_mode == "gen":
            for i in range(1000):
                start_time=time.time()
                cse, c, _t = sess.run([gen_loss, class_error_gen, train_step_gen])
                #print(no, classes_actual)
                #print(cse)
                print('Gen. training set took %f seconds!'%(time.time()-start_time))
                if (i%10 == 0):
                    print("step %d, SCE loss: %g, classification error: %s" % (i, cse, c))
                    writer.add_summary(sess.run(merged), global_step=sess.run(global_step))

                if (i%100 == 99):
                    start = time.time()
                    saver.save(sess, folder_name + '/latest.cpt')
            train_mode = "disc"

writer.close()

#saver.save(sess, './networks/%sEND.cpt'%net_name)
