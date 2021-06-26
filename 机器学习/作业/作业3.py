#_*_ coding:utf-8_*_
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import collections


def read_data(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    content = [content[i].split() for i in range(len(content))]
    content = np.array(content)
    content = np.reshape(content, [-1, ])
    return content


def build_dataset(words):
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary


training_file = 'belling_the_cat.txt'
training_data = read_data(training_file)
dictionary, reverse_dictionary = build_dataset(training_data)
vocab_size = len(dictionary)

learning_rate = 0.001
training_iters = 50000
n_input = 3
n_hidden = 512
display_step = 10

x = tf.placeholder("float", [None, n_input, 1])
y = tf.placeholder("float", [None, vocab_size])
keep_prob = tf.placeholder(tf.float32, [])


def cell():
    cell=rnn.GRUCell(n_hidden,reuse=tf.get_variable_scope().reuse)
    return rnn.DropoutWrapper(cell,output_keep_prob=keep_prob)

def build_Cell(x,layer=3):

    stacked_rnn = []
    stacked_bw_rnn = []
    for i in range(layer):
        stacked_rnn.append(cell())
        stacked_bw_rnn.append(cell())
    mcell = rnn.MultiRNNCell(stacked_rnn)
    mcell_bw = rnn.MultiRNNCell(stacked_bw_rnn)
    outputs,_,_=rnn.stack_bidirectional_dynamic_rnn([mcell], [mcell_bw], x, dtype=tf.float32)
    outputs=tf.transpose(outputs, [1, 0, 2])
    print(outputs)
    output=tf.contrib.layers.fully_connected(outputs[-1],vocab_size,activation_fn=None)
    print(output)
    return output

pred = build_Cell(x)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()


with tf.Session() as session:
    session.run(init)
    step = 0
    offset = random.randint(0,n_input+1)
    end_offset = n_input + 1
    acc_total = 0
    loss_total = 0


    while step < training_iters:

        if offset > (len(training_data)-end_offset):
            offset = random.randint(0, n_input+1)

        symbols_in_keys = [ [dictionary[ str(training_data[i])]] for i in range(offset, offset+n_input) ]

        symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, n_input,1])

        symbols_out_onehot = np.zeros([vocab_size], dtype=float)
        symbols_out_onehot[dictionary[str(training_data[offset+n_input])]] = 1.0
        symbols_out_onehot = np.reshape(symbols_out_onehot,[1,-1])

        session.run(optimizer,feed_dict={x: symbols_in_keys, y: symbols_out_onehot,keep_prob:0.75})
        acc, loss, onehot_pred = session.run([accuracy, cost, pred],feed_dict={x: symbols_in_keys, y: symbols_out_onehot,keep_prob:1})

        loss_total += loss
        acc_total += acc
        if (step+1) % display_step == 0:
            print("Iter= " + str(step+1) + ", Average Loss= " +
                  "{:.6f}".format(loss_total/display_step) + ", Average Accuracy= " +
                  "{:.2f}%".format(100*acc_total/display_step))
            acc_total = 0
            loss_total = 0
            symbols_in = [training_data[i] for i in range(offset, offset + n_input)]
            symbols_out = training_data[offset + n_input]
            symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]
            print("%s - [%s] vs [%s]" % (symbols_in,symbols_out,symbols_out_pred))
        step += 1
        offset += (n_input+1)


    while True:
        prompt = "%s words: " % n_input
        sentence = input(prompt)
        sentence = sentence.strip()
        words = sentence.split(' ')
        if len(words) != n_input:
            continue
        try:
            symbols_in_keys = [dictionary[str(words[i])] for i in range(len(words))]
            for i in range(32):
                keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
                onehot_pred = session.run(pred, feed_dict={x: keys,keep_prob:1.0})
                onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
                sentence = "%s %s" % (sentence,reverse_dictionary[onehot_pred_index])
                symbols_in_keys = symbols_in_keys[1:]
                symbols_in_keys.append(onehot_pred_index)
            print(sentence)
        except:
            print("Word not in dictionary")