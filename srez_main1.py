import os
os.chdir('/home/a/TF/srez')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#os.environ['CUDA_VISIBLE_DEVICES'] = ''

import pdb

import srez_demo
import srez_input
import srez_model
import srez_train

import os.path
import random
import numpy as np
import numpy.random
import scipy.io

import tensorflow as tf

import sys

import datetime

FLAGS = tf.app.flags.FLAGS

# Configuration (alphabetically)
tf.app.flags.DEFINE_integer('batch_size', 16,
                            "Number of samples per batch.")

tf.app.flags.DEFINE_string('checkpoint_dir', 'checkpoint',
                           "Output folder where checkpoints are dumped.")

tf.app.flags.DEFINE_integer('checkpoint_period', 10000,
                            "Number of batches in between checkpoints")

tf.app.flags.DEFINE_string('dataset', 'dataset1', "Path to the dataset directory.")

tf.app.flags.DEFINE_float('epsilon', 1e-8,
                          "Fuzz term to avoid numerical instability")

tf.app.flags.DEFINE_string('run', 'train',
                            "Which operation to run. [demo|train]")

tf.app.flags.DEFINE_float('gene_l1_factor', 1  , "Multiplier for generator L1 loss term")
#tf.app.flags.DEFINE_float('gene_l1_factor', .90, "Multiplier for generator L1 loss term")

tf.app.flags.DEFINE_float('learning_beta1', 0.9, #0.5,
                          "Beta1 parameter used for AdamOptimizer")

# tf.app.flags.DEFINE_float('learning_rate_start', 0.0002,"Starting learning rate used for AdamOptimizer")
tf.app.flags.DEFINE_float('learning_rate_start', 0.002,"Starting learning rate used for AdamOptimizer")
#tf.app.flags.DEFINE_float('learning_rate_start', 0.00001, #0.00020,"Starting learning rate used for AdamOptimizer")

tf.app.flags.DEFINE_integer('learning_rate_half_life', 5000,
                            "Number of batches until learning rate is halved")

tf.app.flags.DEFINE_bool('log_device_placement', False,
                         "Log the device where variables are placed.")

tf.app.flags.DEFINE_integer('sample_size', 64,
                            "Image sample size in pixels. Range [64,128]")

tf.app.flags.DEFINE_integer('summary_period', 200,
                            "Number of batches between summary data dumps")

tf.app.flags.DEFINE_integer('random_seed', 0,
                            "Seed used to initialize rng.")

#tf.app.flags.DEFINE_integer('test_vectors', 16,
tf.app.flags.DEFINE_integer('test_vectors', 16,
                            """Number of features to use for testing""")
                            
tf.app.flags.DEFINE_string('train_dir', 'train',
                           "Output folder where training logs are dumped.")

#tf.app.flags.DEFINE_integer('train_time', 20,"Time in minutes to train the model")
tf.app.flags.DEFINE_integer('train_time', 180,"Time in minutes to train the model")

tf.app.flags.DEFINE_integer('DataH', 64*64,"DataH")
tf.app.flags.DEFINE_integer('DataW', 64*64,"DataW")
tf.app.flags.DEFINE_integer('channelsIn', 64*64,"channelsIn")
tf.app.flags.DEFINE_integer('LabelsH', 64*64,"LabelsH")
tf.app.flags.DEFINE_integer('LabelsW', 64*64,"LabelsW")
tf.app.flags.DEFINE_integer('channelsOut', 64*64,"channelsOut")

tf.app.flags.DEFINE_string('SessionName', 'SessionName', "Which operation to run. [demo|train]")



def getParam(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


import myParams
myParams.init()
# defaults
myParams.myDict['DataH']=1
myParams.myDict['MapSize']=3
myParams.myDict['learning_rate_half_life']=5000
myParams.myDict['learning_rate_start']=0.002

ParamFN='/home/a/TF/Params.txt'
ParamsD = {}
with open(ParamFN) as f:
    for line in f:
        print(line)
        (key, val) = line.split()
        ParamsD[key] = getParam(val)
        myParams.myDict[key]=ParamsD[key]

#pdb.set_trace()
#FLAGS.nChosen=2045
#FLAGS.random_seed=0
#tf.app.flags.random_seed=0
# FullData=scipy.io.loadmat('/home/a/TF/ImgsMC1.mat')
# FullData=scipy.io.loadmat('/home/a/TF/CurChunk.mat')
# Data=FullData['Data']
# Labels=FullData['Labels']
# #pdb.set_trace()
# nSamples=Data.shape[0]
# DataH = Data.shape[1]
# if Data.ndim<3 :
#     DataW = 1
# else:
#     DataW = Data.shape[2]

# if Data.ndim<4 :
#     channelsIn=1
# else :
#     channelsIn = Data.shape[3]

# LabelsH = Labels.shape[1]
# LabelsW = Labels.shape[2]
# if Labels.ndim<4 :
#     channelsOut=1
# else :
#     channelsOut = Labels.shape[3]

# FLAGS.DataH=DataH
# FLAGS.DataW=DataW
# FLAGS.channelsIn=channelsIn
# FLAGS.LabelsH=LabelsH
# FLAGS.LabelsW=LabelsW
# FLAGS.channelsOut=channelsOut

# del FullData
# del Data
# del Labels


# pdb.set_trace()

# MatlabParams=scipy.io.loadmat('/home/a/TF/MatlabParams.mat')

FLAGS.dataset='dataKnee'
#FLAGS.dataset='dataFaceP4'

# SessionNameBase=MatlabParams['SessionName'][0];
SessionNameBase= myParams.myDict['SessionNameBase']

#SessionName=MatlabParams['SessionName'][0] + '__' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

SessionName= SessionNameBase + '_'+FLAGS.dataset + '__' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


FLAGS.checkpoint_dir=SessionName+'_checkpoint'
FLAGS.train_dir=SessionName+'_train'

FLAGS.SessionName=SessionName


remaining_args = FLAGS([sys.argv[0]] + [flag for flag in sys.argv if flag.startswith("--")])
assert(remaining_args == [sys.argv[0]])

def prepare_dirs(delete_train_dir=False):
    # Create checkpoint dir (do not delete anything)
    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)
    
    # Cleanup train dir
    if delete_train_dir:
        if tf.gfile.Exists(FLAGS.train_dir):
            tf.gfile.DeleteRecursively(FLAGS.train_dir)
        tf.gfile.MakeDirs(FLAGS.train_dir)

    # Return names of training files
    if not tf.gfile.Exists(FLAGS.dataset) or \
       not tf.gfile.IsDirectory(FLAGS.dataset):
        raise FileNotFoundError("Could not find folder `%s'" % (FLAGS.dataset,))

    filenames = tf.gfile.ListDirectory(FLAGS.dataset)
    filenames = sorted(filenames)
    random.shuffle(filenames)
    filenames = [os.path.join(FLAGS.dataset, f) for f in filenames]

    return filenames


def setup_tensorflow():
    print("setup_tensorflow")
    # Create session
    #config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    FLAGS
    sess = tf.Session(config=config)

    # Initialize rng with a deterministic seed
    with sess.graph.as_default():
        tf.set_random_seed(FLAGS.random_seed)
        
    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)

    #summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
    print("setup_tensorflow end")
    return sess, summary_writer

def _demo():
    # Load checkpoint
    if not tf.gfile.IsDirectory(FLAGS.checkpoint_dir):
        raise FileNotFoundError("Could not find folder `%s'" % (FLAGS.checkpoint_dir,))

    # Setup global tensorflow state
    sess, summary_writer = setup_tensorflow()

    # Prepare directories
    filenames = prepare_dirs(delete_train_dir=False)

    # Setup async input queues
    features, labels = srez_input.setup_inputs(sess, filenames)

    # Create and initialize model
    [gene_minput, gene_moutput,
     gene_output, gene_var_list,
     disc_real_output, disc_fake_output, disc_var_list] = \
            srez_model.create_model(sess, features, labels)

    # Restore variables from checkpoint
    saver = tf.train.Saver()
    filename = 'checkpoint_new.txt'
    filename = os.path.join(FLAGS.checkpoint_dir, filename)
    saver.restore(sess, filename)

    # Execute demo
    srez_demo.demo1(sess)

class TrainData(object):
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)

def _train():
    # Setup global tensorflow state
    sess, summary_writer = setup_tensorflow()

    # Prepare directories
    all_filenames = prepare_dirs(delete_train_dir=True)

    # Separate training and test sets
    train_filenames = all_filenames[:-FLAGS.test_vectors]
    test_filenames  = all_filenames[-FLAGS.test_vectors:]

    # TBD: Maybe download dataset here

    #pdb.set_trace()
    # Setup async input queues
    train_features, train_labels = srez_input.setup_inputs(sess, train_filenames)
    test_features,  test_labels  = srez_input.setup_inputs(sess, test_filenames)

    print('train_features %s' % (train_features))
    print('train_labels %s' % (train_labels))


    # Add some noise during training (think denoising autoencoders)
    AddNoise=False
    if AddNoise:
        noise_level = .03
        noisy_train_features = train_features + tf.random_normal(train_features.get_shape(), stddev=noise_level)
    else:
        noisy_train_features = train_features

    # Create and initialize model
    [gene_minput, gene_moutput,
     gene_output, gene_var_list,
     disc_real_output, disc_fake_output, disc_var_list] = \
            srez_model.create_model(sess, noisy_train_features, train_labels)
    
    # gene_VarNamesL=[];
    # for line in gene_var_list: gene_VarNamesL.append(line.name+'           ' + str(line.shape.as_list()))
    # gene_VarNamesL.sort()

    # for line in gene_VarNamesL: print(line)
    # # var_23 = [v for v in tf.global_variables() if v.name == "gene/GEN_L020/C2D_weight:0"][0]

    # for line in sess.graph.get_operations(): print(line)
    # Gen3_ops=[]
    # for line in sess.graph.get_operations():
    #     if 'GEN_L003' in line.name:
    #         Gen3_ops.append(line)

    #     LL=QQQ.outputs[0]
        
    #     for x in Gen3_ops: print(x.name +'           ' + str(x.outputs[0].shape))

    
    # pdb.set_trace()

    gene_loss = srez_model.create_generator_loss(disc_fake_output, gene_output, train_features,train_labels)
    disc_real_loss, disc_fake_loss = \
                     srez_model.create_discriminator_loss(disc_real_output, disc_fake_output)
    disc_loss = tf.add(disc_real_loss, disc_fake_loss, name='disc_loss')
    
    (global_step, learning_rate, gene_minimize, disc_minimize) = \
            srez_model.create_optimizers(gene_loss, gene_var_list, disc_loss, disc_var_list)

    # Train model
    train_data = TrainData(locals())
    
    #pdb.set_trace()
    # ggg: to restore session
    RestoreSession=False
    if RestoreSession:
        saver = tf.train.Saver()
        filename = 'checkpoint_new.txt'
        filename = os.path.join(FLAGS.checkpoint_dir, filename)
        saver.restore(sess, filename)

    srez_train.train_model(train_data)

def main(argv=None):
    print("aaa")
    _train()
    # Training or showing off?
    #_train()
    #if FLAGS.run == 'demo':
    #    _demo()
    #elif FLAGS.run == 'train':
    #    _train()

if __name__ == '__main__':
  tf.app.run()

#print("asd40")
_train()
#setup_tensorflow()
#tf.app.run()

#print("asd5")
