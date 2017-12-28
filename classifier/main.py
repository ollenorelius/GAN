import tensorflow as tf
import reader
import nets


with tf.variable_scope('input_batching'):
    data_folders = ["frank", "not_frank"]
    data_folders = ["car", "dog", "graph", "human"]

    batch = reader.get_batch(10, data_folders)

    image = batch[0]
    tf.summary.image("input images", image)
    classes = tf.squeeze(batch[1])

    net_output = nets.make_network(image)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=classes,
                                           logits=net_output)
    diff = (classes - tf.nn.softmax(net_output))/2
    a = tf.abs(diff)
    class_error = tf.reduce_sum(a)
    print(image.shape)
    print(net_output.shape)
    print(loss)
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("classification_error", class_error)

with tf.name_scope('Global_step'):
    global_step = tf.Variable(0, dtype=tf.int32)

train_step = tf.train.AdamOptimizer(0.001).minimize(loss, global_step=global_step)

merged = tf.summary.merge_all()


print("Model constructed!")

sess = tf.Session()

print("Variables initialized!")
import os
import sys

net_name = 'test_net2'
folder_name = './networks/%s'%net_name
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

saver = tf.train.Saver()
writer = tf.summary.FileWriter("output/"+net_name, sess.graph)
coordinate = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coordinate)


if True:
    print('loading network.. ', end='')
    try:
        if '-best' in sys.argv:
            saver.restore(sess, folder_name + '/best_valid.cpt')
            print('Starting from best net.. ', end='')
            print('Done.')
        else:
            saver.restore(sess, folder_name + '/latest.cpt')
            print('Starting from latest net.. ', end='')
            print('Done.')
    except:
        print('Couldn\'t load net, creating new! E:(%s)'%sys.exc_info()[0])
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
else:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

test = tf.constant(1)

min_loss = 100
#print([n.name for n in tf.get_default_graph().as_graph_def().node])
import time
from tensorflow.python.framework import graph_util
with sess.as_default():
    for i in range(200000):
        start_time=time.time()
        cse, c, _t = sess.run([loss, class_error, train_step])
        #print(no, classes_actual)
        #print(cse)
        print('training set took %f seconds!'%(time.time()-start_time))
        if (i%10 == 0):
            no, classes_actual = sess.run([net_output, classes])
            print("step %d, CSE loss: %g, classification error: %s" % (i, cse, c))
            writer.add_summary(sess.run(merged), global_step=sess.run(global_step))

        if (i%1000 == 0):
            start = time.time()
            saver.save(sess, folder_name + '/latest.cpt')

writer.close()

saver.save(sess, './networks/%sEND.cpt'%net_name)
