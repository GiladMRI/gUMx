import numpy as np
import os.path
import scipy.misc
import tensorflow as tf
import time
import scipy.io
import pdb
import myParams

FLAGS = tf.app.flags.FLAGS

def _summarize_progress(train_data, feature, label, gene_output, batch, suffix, max_samples=8):
    td = train_data

    size = [label.shape[1], label.shape[2]]

    #nearest = tf.image.resize_nearest_neighbor(feature, size)
    #nearest = tf.maximum(tf.minimum(nearest, 1.0), 0.0)

    #bicubic = tf.image.resize_bicubic(feature, size)
    #bicubic = tf.maximum(tf.minimum(bicubic, 1.0), 0.0)

    gene_outputx=np.sqrt(np.power(gene_output[:,:,:,0],2)+np.power(gene_output[:,:,:,1],2))
    Theta=np.arctan2(gene_output[:,:,:,1],gene_output[:,:,:,0])/(2*np.pi)+0.5;
    labelx=np.sqrt(np.power(label[:,:,:,0],2)+np.power(label[:,:,:,1],2))
    labelx[0]=Theta[0];

    clipped = tf.maximum(tf.minimum(gene_outputx, 1.0), 0.0)

    #image   = tf.concat([nearest, bicubic, clipped, label], 2)
    image   = tf.concat([clipped, labelx], 2)

    image=tf.reshape(image,[image.shape[0], image.shape[1], image.shape[2], 1])
    # pdb.set_trace()
    image   = tf.concat([image,image,image], 3)

    image = image[0:max_samples,:,:,:]
    #image = tf.concat([image[i,:,:,:] for i in range(max_samples)], 0)
    image1 = tf.concat([image[i,:,:,:] for i in range(int(max_samples/2))], 0)
    image2 = tf.concat([image[i,:,:,:] for i in range(int(max_samples/2),max_samples)], 0)
    image  = tf.concat([image1, image2], 1)
    image = td.sess.run(image)

    filename = 'batch%06d_%s.png' % (batch, suffix)
    filename = os.path.join(FLAGS.train_dir, filename)
    scipy.misc.toimage(image, cmin=0., cmax=1.).save(filename)
    print("    Saved %s" % (filename,))

def _save_checkpoint(train_data, batch,G_LossV):
    td = train_data

    oldname = 'checkpoint_old.txt'
    newname = 'checkpoint_new.txt'

    oldname = os.path.join(FLAGS.checkpoint_dir, oldname)
    newname = os.path.join(FLAGS.checkpoint_dir, newname)

    # Delete oldest checkpoint
    try:
        tf.gfile.Remove(oldname)
        tf.gfile.Remove(oldname + '.meta')
    except:
        pass

    # Rename old checkpoint
    try:
        tf.gfile.Rename(newname, oldname)
        tf.gfile.Rename(newname + '.meta', oldname + '.meta')
    except:
        pass

    # Generate new checkpoint
    saver = tf.train.Saver()
    saver.save(td.sess, newname)

    print("    Checkpoint saved")

    # save all:
    TrainSummary={}
    filenamex = 'TrainSummary_%06d.mat' % (batch)
    filename = os.path.join(FLAGS.train_dir, filenamex)
    VLen=td.gene_var_list.__len__()
    var_list=[]
    for i in range(0, VLen): 
        var_list.append(td.gene_var_list[i].name);
        tmp=td.sess.run(td.gene_var_list[i])
        s1=td.gene_var_list[i].name
        print("Saving  %s" % (s1))
        s1=s1.replace(':','_')
        s1=s1.replace('/','_')
        TrainSummary[s1]=tmp
    
    TrainSummary['var_list']=var_list
    TrainSummary['G_LossV']=G_LossV

    scipy.io.savemat(filename,TrainSummary)

    print("saved to %s" % (filename))

def train_model(train_data):
    td = train_data

    summaries = tf.summary.merge_all()
    RestoreSession=False
    if not RestoreSession:
        td.sess.run(tf.global_variables_initializer())

    # lrval       = FLAGS.learning_rate_start
    lrval       = myParams.myDict['learning_rate_start']
    start_time  = time.time()
    last_summary_time  = time.time()
    lsat_checkpoint_time  = time.time()
    done  = False
    batch = 0

    print("lrval %f" % (lrval))

    # assert FLAGS.learning_rate_half_life % 10 == 0

    # Cache test features and labels (they are small)
    test_feature, test_label = td.sess.run([td.test_features, td.test_labels])

    G_LossV=np.zeros((1000000), dtype=np.float32)
    filename = os.path.join(FLAGS.train_dir, 'TrainSummary.mat')
    
    feed_dictOut = {td.gene_minput: test_feature}
    gene_output = td.sess.run(td.gene_moutput, feed_dict=feed_dictOut)
    _summarize_progress(td, test_feature, test_label, gene_output, batch, 'out')

    feed_dict = {td.learning_rate : lrval}
    opsx = [td.gene_minimize, td.gene_loss]
    _, gene_loss = td.sess.run(opsx, feed_dict=feed_dict)

    # opsy = [td.gene_loss]
    # gene_loss = td.sess.run(opsy, feed_dict=feed_dict)

    # ops = [td.gene_minimize, td.disc_minimize, td.gene_loss, td.disc_real_loss, td.disc_fake_loss]
    # _, _, gene_loss, disc_real_loss, disc_fake_loss = td.sess.run(ops, feed_dict=feed_dict)

    batch += 1

    gene_output = td.sess.run(td.gene_moutput, feed_dict=feed_dictOut)
    _summarize_progress(td, test_feature, test_label, gene_output, batch, 'out')

    feed_dict = {td.learning_rate : lrval}
    # ops = [td.gene_minimize, td.disc_minimize, td.gene_loss, td.disc_real_loss, td.disc_fake_loss]

    opsx = [td.gene_minimize, td.gene_loss]
    _, gene_loss = td.sess.run(opsx, feed_dict=feed_dict)

    batch += 1

    gene_output = td.sess.run(td.gene_moutput, feed_dict=feed_dictOut)
    _summarize_progress(td, test_feature, test_label, gene_output, batch, 'out')

    # load model
    #saver.restore(sess,tf.train.latest_checkpoint('./'))
    # running model on data:test_feature
    RunOnData=False
    if RunOnData:
        filenames = tf.gfile.ListDirectory('DataAfterpMat')
        filenames = sorted(filenames)
        #filenames = [os.path.join('DataAfterpMat', f) for f in filenames]
        Ni=len(filenames)
        OutBase=FLAGS.SessionName+'_OutMat'
        tf.gfile.MakeDirs(OutBase)

        #pdb.set_trace()

        for index in range(Ni):
            print(index)
            print(filenames[index])
            CurData=scipy.io.loadmat(os.path.join('DataAfterpMat', filenames[index]))
            Data=CurData['CurData']
            Data=Data.reshape((1,64,64,1))
            test_feature=np.kron(np.ones((16,1,1,1)),Data)
            #test_feature = np.array(np.random.choice([0, 1], size=(16,64,64,1)), dtype='float32')


            feed_dictOut = {td.gene_minput: test_feature}
            gene_output = td.sess.run(td.gene_moutput, feed_dict=feed_dictOut)

            filenameOut=os.path.join(OutBase, filenames[index][:-4] + '_out.mat') 

            SOut={}
            SOut['X']=gene_output[0]
            scipy.io.savemat(filenameOut,SOut)

    # pdb.set_trace()

    #_summarize_progress(td, test_feature, test_label, gene_output, batch, 'out')
    # to get value of var:
    # ww=td.sess.run(td.gene_var_list[1])
    
    ifilename=os.path.join('RealData', 'a.mat')
    RealData=scipy.io.loadmat(ifilename)
    RealData=RealData['Data']
    RealData=RealData.reshape((RealData.shape[0],RealData.shape[1],1,1))
    Real_feature=RealData
    Real_dictOut = {td.gene_minput: Real_feature}

    # LearningDecayFactor=np.power(2,(-1/FLAGS.learning_rate_half_life))
    LearningDecayFactor=np.power(2,(-1/myParams.myDict['learning_rate_half_life']))

    # train_time=FLAGS.train_time
    train_time=myParams.myDict['train_time']
    while not done:
        batch += 1
        gene_loss = disc_real_loss = disc_fake_loss = -1.234

        # Update learning rate
        lrval=lrval*LearningDecayFactor;

        

        #print("batch %d gene_l1_factor %f' " % (batch,FLAGS.gene_l1_factor))
        if batch==2000:
            FLAGS.gene_l1_factor=0.9
        
        RunDiscriminator= FLAGS.gene_l1_factor < 0.999

        feed_dict = {td.learning_rate : lrval}
        if RunDiscriminator:
            ops = [td.gene_minimize, td.disc_minimize, td.gene_loss, td.disc_real_loss, td.disc_fake_loss]
            _, _, gene_loss, disc_real_loss, disc_fake_loss = td.sess.run(ops, feed_dict=feed_dict)
        else:
            ops = [td.gene_minimize, td.gene_loss]
            _, gene_loss = td.sess.run(ops, feed_dict=feed_dict)
        
        
        G_LossV[batch]=gene_loss
        
        if batch % 10 == 0:

            # pdb.set_trace()

            # Show we are alive
            elapsed = int(time.time() - start_time)/60
            print('Progress[%3d%%], ETA[%4dm], Batch [%4d], G_Loss[%3.3f], D_Real_Loss[%3.3f], D_Fake_Loss[%3.3f]' %
                  (int(100*elapsed/train_time), train_time - elapsed, batch, gene_loss, disc_real_loss, disc_fake_loss))

            # Finished?            
            current_progress = elapsed / train_time
            if current_progress >= 1.0:
                done = True
            
            # Update learning rate
            # if batch % FLAGS.learning_rate_half_life == 0:
            #     lrval *= .5

        if batch % FLAGS.summary_period == 0:
            # Show progress with test features
            # feed_dict = {td.gene_minput: test_feature}
            gene_output = td.sess.run(td.gene_moutput, feed_dict=feed_dictOut)
            
            gene_RealOutput = td.sess.run(td.gene_moutput, feed_dict=Real_dictOut)
            gene_output[0]=gene_RealOutput[0]
            
            _summarize_progress(td, test_feature, test_label, gene_output, batch, 'out')

            last_summary_time  = time.time()
    
            
        if batch % FLAGS.checkpoint_period == 0:
            lsat_checkpoint_time  = time.time()
            # Save checkpoint
            _save_checkpoint(td, batch,G_LossV)

    _save_checkpoint(td, batch,G_LossV)
    
    print('Finished training!')