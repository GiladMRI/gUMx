import tensorflow as tf
import pdb
import numpy as np

import myParams


FLAGS = tf.app.flags.FLAGS

def setup_inputs(sess, filenames, image_size=None, capacity_factor=3):

    if image_size is None:
        image_size = FLAGS.sample_size
    
    #pdb.set_trace()

    #label_bytes = 39
    #image_bytes = 218*1 * 178 * 3
    #record_bytes = label_bytes + image_bytes
    #reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    reader = tf.TFRecordReader()

    # Read each JPEG file
    #reader = tf.WholeFileReader()
    #filenames=['dataset1/a3.tfrecords', 'dataset1/a1.tfrecords']
    filename_queue = tf.train.string_input_producer(filenames)
    key, value = reader.read(filename_queue)
    # channelsIn = FLAGS.channelsIn
    # channelsOut = FLAGS.channelsOut
    # DataH = FLAGS.DataH
    # DataW = FLAGS.DataW
    # LabelsH = FLAGS.LabelsH
    # LabelsW = FLAGS.LabelsW

    channelsIn=myParams.myDict['channelsIn']
    channelsOut=myParams.myDict['channelsOut']
    DataH=myParams.myDict['DataH']
    DataW=myParams.myDict['DataW']
    LabelsH=myParams.myDict['LabelsH']
    LabelsW=myParams.myDict['LabelsW']

    #image = tf.image.decode_jpeg(value, channels=channels, name="dataset_image")

    #print('1')
    featuresA = tf.parse_single_example(
        value,
        features={
            'DataH': tf.FixedLenFeature([], tf.int64),
            'DataW': tf.FixedLenFeature([], tf.int64),
            'channelsIn': tf.FixedLenFeature([], tf.int64),
            'LabelsH': tf.FixedLenFeature([], tf.int64),
            'LabelsW': tf.FixedLenFeature([], tf.int64),
            'channelsOut': tf.FixedLenFeature([], tf.int64),
            'data_raw': tf.FixedLenFeature([], tf.string),
            'labels_raw': tf.FixedLenFeature([], tf.string)
            #'X': tf.FixedLenFeature([218*1* 178* 3], tf.string),
        })
    #image = tf.decode_raw(featuresA['image_raw'], tf.int8)
    feature = tf.decode_raw(featuresA['data_raw'], tf.float32)
    labels = tf.decode_raw(featuresA['labels_raw'], tf.float32)
    #height=featuresA['height'];
    #height=3667*2
    #height=FLAGS.nChosen*2
    #record_bytes=features['X']

    print('setup_inputs')
    print('Data   H,W,#ch: %d,%d,%d -> Labels H,W,#ch %d,%d,%d' % (DataH,DataW,channelsIn,LabelsH,LabelsW,channelsOut))
    print('------------------')
    #readery = tf.TFRecordReader()
    #example = tf.train.Example()
    #keyy, valuey = readery.read(filename_queue)
    #for serialized_example in tf.python_io.tf_record_iterator('test1.tfrecords'):
    #    example = tf.train.Example()
    #    example.ParseFromString(serialized_example)
    #    x_1 = np.array(example.features.feature['X'].float_list.value)
    #    break
    #example.ParseFromString(value)
    #x_1 = np.array(example.features.feature['X'].float_list.value)

    #print('2')
    #image = tf.decode_raw(featuresA['X'], tf.float32)
    #image = tf.decode_raw(featuresA['X'], tf.int8)
    #value.set_shape([None, record_bytes])    
    #dataset = tf.data.TFRecordDataset(value)

    # pdb.set_trace()

    #for serialized_example in tf.python_io.tf_record_iterator('test1.tfrecords'):
    #    example = tf.train.Example()
    #    example.ParseFromString(serialized_example)
    #    x_1 = np.array(example.features.feature['X'].float_list.value)
        #y_1 = np.array(example.features.feature['Y'].float_list.value)
    #    break

    #readerX = tf.TFRecordReader()
    #_, serialized_example = readerX.read(value)
    #record_bytes = tf.decode_raw(value, tf.int8)
    #record_bytes = tf.decode_raw(value[0:(image_bytes+1)], tf.float32)
    #pdb.set_trace()
    #record_bytes = tf.decode_raw(tf.strided_slice(value, [label_bytes],[label_bytes + image_bytes]), tf.float32)
    #image = tf.reshape(tf.strided_slice(record_bytes, [label_bytes],[label_bytes + image_bytes]),[218*4, 178, 3])
    #image = tf.reshape(record_bytes,[218, 178, 3])
    #print('33')
    #image = tf.reshape(image,[218, 178, 3])
    
    #image = tf.reshape(image,[1,64*64*2, channelsIn])
    #labels = tf.reshape(labels,[64, 64, channelsOut])

    #image = tf.reshape(image, [1, height, channelsIn])
    feature = tf.reshape(feature, [DataH, DataW, channelsIn])
    labels = tf.reshape(labels, [image_size, image_size, channelsOut])
    
    #print('44')
    #example.ParseFromString(serialized_example)
    #x_1 = np.array(example.features.feature['X'].float_list.value)

    # Convert from [depth, height, width] to [height, width, depth].
    #result.uint8image = tf.transpose(depth_major, [1, 2, 0])

    feature.set_shape([None, None, channelsIn])
    labels.set_shape([None, None, channelsOut])

    
    #print('channels %d' % (channels))

    #print('filename_queue %s' % (filename_queue))

    # print('image %s' % (image))

    #pdb.set_trace()


    # Crop and other random augmentations
    #image = tf.image.random_flip_left_right(image)
    #image = tf.image.random_saturation(image, .95, 1.05)
    #image = tf.image.random_brightness(image, .05)
    #image = tf.image.random_contrast(image, .95, 1.05)

    #print('55')
    #wiggle = 8
    #off_x, off_y = 25-wiggle, 60-wiggle
    #crop_size = 128
    #crop_size_plus = crop_size + 2*wiggle
    #print('56')
    #image = tf.image.crop_to_bounding_box(image, off_y, off_x, crop_size_plus, crop_size_plus)
    #print('57')
    #image = tf.image.crop_to_bounding_box(image, 1, 2, crop_size, crop_size)
    #image = tf.random_crop(image, [crop_size, crop_size, 3])

    #image = tf.reshape(image, [1, crop_size, crop_size, 3])
    #feature = tf.reshape(image, [ 1, height, channelsIn])
    feature = tf.reshape(feature, [DataH, DataW, channelsIn])
    feature = tf.cast(feature, tf.float32) #/255.0

    #print('66')
    
    labels = tf.reshape(labels, [LabelsH, LabelsW, channelsOut])
    label = tf.cast(labels, tf.float32) #/255.0

    # pdb.set_trace()
    #if crop_size != image_size:
    #    image = tf.image.resize_area(image, [image_size, image_size])

    # The feature is simply a Kx downscaled version
    #K = 1
    #downsampled = tf.image.resize_area(image, [image_size//K, image_size//K])

    #print('77')
    #feature = tf.reshape(downsampled, [image_size//K, image_size//K, 3])
    #feature = tf.reshape(downsampled, [image_size//K, image_size//K, 3])
    #label   = tf.reshape(image,       [image_size,   image_size,     3])

    #feature = tf.reshape(image,     [image_size,    image_size,     channelsIn])
    #feature = tf.reshape(image,     [1, image_size*image_size*2,     channelsIn])
    #label   = tf.reshape(labels,    [image_size,    image_size,     channelsOut])

    # Using asynchronous queues
    features, labels = tf.train.batch([feature, label],
                                      batch_size=FLAGS.batch_size,
                                      num_threads=4,
                                      capacity = capacity_factor*FLAGS.batch_size,
                                      name='labels_and_features')

    #print('88')
    tf.train.start_queue_runners(sess=sess)
    
    #print('99')
    return features, labels
