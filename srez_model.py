import numpy as np
import tensorflow as tf
import scipy.io

FLAGS = tf.app.flags.FLAGS

import myParams

class Model:
    """A neural network model.

    Currently only supports a feedforward architecture."""
    
    def __init__(self, name, features):
        self.name = name
        self.outputs = [features]

    def _get_layer_str(self, layer=None):
        if layer is None:
            layer = self.get_num_layers()
        
        return '%s_L%03d' % (self.name, layer+1)

    def _get_num_inputs(self):
        return int(self.get_output().get_shape()[-1])

    def _glorot_initializer(self, prev_units, num_units, stddev_factor=1.0):
        """Initialization in the style of Glorot 2010.

        stddev_factor should be 1.0 for linear activations, and 2.0 for ReLUs"""
        stddev  = np.sqrt(stddev_factor / np.sqrt(prev_units*num_units))
        return tf.truncated_normal([prev_units, num_units],
                                    mean=0.0, stddev=stddev)

    def _glorot_initializer_conv2d(self, prev_units, num_units, mapsize, stddev_factor=1.0):
        """Initialization in the style of Glorot 2010.

        stddev_factor should be 1.0 for linear activations, and 2.0 for ReLUs"""

        stddev  = np.sqrt(stddev_factor / (np.sqrt(prev_units*num_units)*mapsize*mapsize))
        return tf.truncated_normal([mapsize, mapsize, prev_units, num_units],
                                    mean=0.0, stddev=stddev)

    def get_num_layers(self):
        return len(self.outputs)

    def add_batch_norm(self, scale=False):
        """Adds a batch normalization layer to this model.

        See ArXiv 1502.03167v3 for details."""

        # TBD: This appears to be very flaky, often raising InvalidArgumentError internally
        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            out = tf.contrib.layers.batch_norm(self.get_output(), scale=scale)
        
        self.outputs.append(out)
        return self

    def add_flatten(self):
        """Transforms the output of this network to a 1D tensor"""

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            batch_size = int(self.get_output().get_shape()[0])
            out = tf.reshape(self.get_output(), [batch_size, -1])

        self.outputs.append(out)
        return self

    # ggg
    def add_reshapeTo4D(self, height, width):
        """Transforms the output of this network to a 1D tensor"""

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            batch_size = int(self.get_output().get_shape()[0])
            out = tf.reshape(self.get_output(), [batch_size, height, width, -1])

        self.outputs.append(out)
        return self

    # ggg
    def add_Mult2D(self, stddev_factor=1.0):
        """Adds a dense linear layer to this model.

        Uses Glorot 2010 initialization assuming linear activation."""
        
        assert len(self.get_output().get_shape()) == 4, "Previous layer must be 4-dimensional (batch, H, W, channels)"

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):

            batch_size = int(self.get_output().get_shape()[0])
            H = int(self.get_output().get_shape()[1])
            W = int(self.get_output().get_shape()[2])
            channels = int(self.get_output().get_shape()[3])

            prev_units = self._get_num_inputs()
            
            # Weight term
            initw   = self._glorot_initializer(H, W,
                                               stddev_factor=stddev_factor)
            weight  = tf.get_variable('M2D_weight', initializer=initw)

            # Bias term
            #initb   = tf.constant(0.0, shape=[num_units])
            #bias    = tf.get_variable('bias', initializer=initb)

            # Output of this layer
            #out     = tf.matmul(self.get_output(), weight) + bias

            weight = tf.reshape(weight, [1,H,W,1])
            weight = tf.tile(weight,[batch_size,1,1,channels])
            out    = tf.multiply(self.get_output(), weight)

        self.outputs.append(out)
        return self

    # ggg
    def add_Mult2DComplexRI(self, stddev_factor=1.0):
        """Adds a dense linear layer to this model.

        Uses Glorot 2010 initialization assuming linear activation."""
        
        assert len(self.get_output().get_shape()) == 4, "Previous layer must be 4-dimensional (batch, H, W, channels)"

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):

            batch_size = int(self.get_output().get_shape()[0])
            H = int(self.get_output().get_shape()[1])
            W = int(self.get_output().get_shape()[2])
            channels = int(self.get_output().get_shape()[3])
            HalfChannels = int(channels/2)

            prev_units = self._get_num_inputs()
            
            initr   = self._glorot_initializer(H, W, stddev_factor=stddev_factor)
            initi   = self._glorot_initializer(H, W, stddev_factor=stddev_factor)
            weightr  = tf.get_variable('M2D_CRIR_weight', initializer=initr)
            weighti  = tf.get_variable('M2D_CRII_weight', initializer=initi)

            weightr = tf.reshape(weightr, [1,H,W,1])
            weighti = tf.reshape(weighti, [1,H,W,1])
            
            weightr = tf.tile(weightr,[batch_size,1,1,channels])
            weighti = tf.tile(weighti,[batch_size,1,1,channels])

            a = np.array([np.arange(2,channels+1,2),np.arange(1,channels,2)])
            Idxs=a.transpose().flatten()

            IdxsT = tf.constant(Idxs, shape=[channels])

            PlusMinus = tf.constant([1.0, -1.0], shape=[1,1,1,2])
            PlusMinus = tf.tile(PlusMinus,[batch_size,H,W,HalfChannels])

            A   = tf.multiply(self.get_output(), weightr)
            B   = tf.multiply(self.get_output(), weighti)

            B=tf.gather(B,IdxsT,validate_indices=None,name=None,axis=3)
            B=tf.multiply(B, PlusMinus)

            out= A + B

        self.outputs.append(out)
        return self

    # ggg
    def add_Mult3DComplexRI(self, stddev_factor=1.0):
        """Adds a dense linear layer to this model.

        Uses Glorot 2010 initialization assuming linear activation."""
        
        assert len(self.get_output().get_shape()) == 4, "Previous layer must be 4-dimensional (batch, H, W, channels)"

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):

            batch_size = int(self.get_output().get_shape()[0])
            H = int(self.get_output().get_shape()[1])
            W = int(self.get_output().get_shape()[2])
            channels = int(self.get_output().get_shape()[3])
            HalfChannels = int(channels/2)

            prev_units = self._get_num_inputs()
            
            initr   = self._glorot_initializer(H, W*HalfChannels, stddev_factor=stddev_factor)
            initi   = self._glorot_initializer(H, W*HalfChannels, stddev_factor=stddev_factor)
            weightr  = tf.get_variable('M3D_CRIR_weight', initializer=initr)
            weighti  = tf.get_variable('M3D_CRII_weight', initializer=initi)

            weightr = tf.reshape(weightr, [1,H,W,HalfChannels])
            weighti = tf.reshape(weighti, [1,H,W,HalfChannels])
            
            weightr = tf.tile(weightr,[batch_size,1,1,1])
            weighti = tf.tile(weighti,[batch_size,1,1,1])

            Idxs1122=np.int32(np.kron(np.arange(0,HalfChannels),np.ones(2)))
            Idxs1122T = tf.constant(Idxs1122, shape=[channels])

            weightr=tf.gather(weightr,Idxs1122T,validate_indices=None,name=None,axis=3)            
            weighti=tf.gather(weighti,Idxs1122T,validate_indices=None,name=None,axis=3)            
            
            A   = tf.multiply(self.get_output(), weightr)
            B   = tf.multiply(self.get_output(), weighti)

            PlusMinus = tf.constant([-1.0, 1.0], shape=[1,1,1,2])
            PlusMinus = tf.tile(PlusMinus,[batch_size,H,W,HalfChannels])

            Idxs = np.array([np.arange(1,channels,2),np.arange(0,channels-1,2)])
            Idxs = Idxs.transpose().flatten()
            IdxsT = tf.constant(Idxs, shape=[channels])
            
            B=tf.gather(B,IdxsT,validate_indices=None,name=None,axis=3)
            B=tf.multiply(B, PlusMinus)

            out= A + B

        self.outputs.append(out)
        return self

    # ggg
    #ef add_UNet(self, nunits1=1, nunits2=1, nunits3=1, nunits4=1, nunits5=1, nunits6=1, nunits7=1, nunits8=1, nunits9=1):

    def add_denseFromM(self, name):

        assert len(self.get_output().get_shape()) == 2, "Previous layer must be 2-dimensional (batch, channels)"

        ConstS=scipy.io.loadmat('/home/a/TF/Consts.mat')
        #CurM=ConstS[name]
        CurM=ConstS['piMDR']
        ConstMf=np.float32(CurM)

        prev_units = self._get_num_inputs()
        num_units=ConstMf.shape[1]
        
        ConstMtf = tf.constant(ConstMf, shape=[prev_units,num_units])
        
        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            
            
            # Weight term
            weight  = tf.get_variable('DM_weight', initializer=ConstMtf)

            # Bias term
            initb   = tf.constant(0.0, shape=[num_units])
            bias    = tf.get_variable('DM_bias', initializer=initb)

            # Output of this layer
            out     = tf.matmul(self.get_output(), weight) + bias

        self.outputs.append(out)
        return self

    def add_dense(self, num_units, stddev_factor=1.0):
        """Adds a dense linear layer to this model.

        Uses Glorot 2010 initialization assuming linear activation."""
        
        assert len(self.get_output().get_shape()) == 2, "Previous layer must be 2-dimensional (batch, channels)"

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            prev_units = self._get_num_inputs()
            
            # Weight term
            initw   = self._glorot_initializer(prev_units, num_units,
                                               stddev_factor=stddev_factor)
            weight  = tf.get_variable('Dweight', initializer=initw)

            # Bias term
            initb   = tf.constant(0.0, shape=[num_units])
            bias    = tf.get_variable('Dbias', initializer=initb)

            # Output of this layer
            out     = tf.matmul(self.get_output(), weight) + bias

        self.outputs.append(out)
        return self

    def add_sigmoid(self):
        """Adds a sigmoid (0,1) activation function layer to this model."""

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            prev_units = self._get_num_inputs()
            out = tf.nn.sigmoid(self.get_output())
        
        self.outputs.append(out)
        return self

    def add_softmax(self):
        """Adds a softmax operation to this model"""

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            this_input = tf.square(self.get_output())
            reduction_indices = list(range(1, len(this_input.get_shape())))
            acc = tf.reduce_sum(this_input, reduction_indices=reduction_indices, keep_dims=True)
            out = this_input / (acc+FLAGS.epsilon)
            #out = tf.verify_tensor_all_finite(out, "add_softmax failed; is sum equal to zero?")
        
        self.outputs.append(out)
        return self

    # ggg
    def add_tanh(self):
        """Adds a TanH activation function to this model"""

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            out = tf.nn.tanh(self.get_output())

        self.outputs.append(out)
        return self      

    def add_relu(self):
        """Adds a ReLU activation function to this model"""

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            out = tf.nn.relu(self.get_output())

        self.outputs.append(out)
        return self        

    def add_elu(self):
        """Adds a ELU activation function to this model"""

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            out = tf.nn.elu(self.get_output())

        self.outputs.append(out)
        return self

    def add_lrelu(self, leak=.2):
        """Adds a leaky ReLU (LReLU) activation function to this model"""

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            t1  = .5 * (1 + leak)
            t2  = .5 * (1 - leak)
            out = t1 * self.get_output() + \
                  t2 * tf.abs(self.get_output())
            
        self.outputs.append(out)
        return self

    def add_conv2d(self, num_units, mapsize=1, stride=1, stddev_factor=1.0):
        """Adds a 2D convolutional layer."""

        assert len(self.get_output().get_shape()) == 4 and "Previous layer must be 4-dimensional (batch, width, height, channels)"
        
        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            prev_units = self._get_num_inputs()
            
            # Weight term and convolution
            initw  = self._glorot_initializer_conv2d(prev_units, num_units,
                                                     mapsize,
                                                     stddev_factor=stddev_factor)
            weight = tf.get_variable('C2D_weight', initializer=initw)
            out    = tf.nn.conv2d(self.get_output(), weight,
                                  strides=[1, stride, stride, 1],
                                  padding='SAME')

            # Bias term
            initb  = tf.constant(0.0, shape=[num_units])
            bias   = tf.get_variable('C2D_bias', initializer=initb)
            out    = tf.nn.bias_add(out, bias)
            
        self.outputs.append(out)
        return self

    def add_conv2d_transpose(self, num_units, mapsize=1, stride=1, stddev_factor=1.0):
        """Adds a transposed 2D convolutional layer"""

        assert len(self.get_output().get_shape()) == 4 and "Previous layer must be 4-dimensional (batch, width, height, channels)"

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            prev_units = self._get_num_inputs()
            
            # Weight term and convolution
            initw  = self._glorot_initializer_conv2d(prev_units, num_units,
                                                     mapsize,
                                                     stddev_factor=stddev_factor)
            weight = tf.get_variable('C2DT_weight', initializer=initw)
            weight = tf.transpose(weight, perm=[0, 1, 3, 2])
            prev_output = self.get_output()
            output_shape = [FLAGS.batch_size,
                            int(prev_output.get_shape()[1]) * stride,
                            int(prev_output.get_shape()[2]) * stride,
                            num_units]
            out    = tf.nn.conv2d_transpose(self.get_output(), weight,
                                            output_shape=output_shape,
                                            strides=[1, stride, stride, 1],
                                            padding='SAME')

            # Bias term
            initb  = tf.constant(0.0, shape=[num_units])
            bias   = tf.get_variable('C2DT_bias', initializer=initb)
            out    = tf.nn.bias_add(out, bias)
            
        self.outputs.append(out)
        return self

    def add_residual_block(self, num_units, mapsize=3, num_layers=2, stddev_factor=1e-3):
        """Adds a residual block as per Arxiv 1512.03385, Figure 3"""

        assert len(self.get_output().get_shape()) == 4 and "Previous layer must be 4-dimensional (batch, width, height, channels)"

        # Add projection in series if needed prior to shortcut
        if num_units != int(self.get_output().get_shape()[3]):
            self.add_conv2d(num_units, mapsize=1, stride=1, stddev_factor=1.)

        bypass = self.get_output()

        # Residual block
        for _ in range(num_layers):
            self.add_batch_norm()
            self.add_relu()
            self.add_conv2d(num_units, mapsize=mapsize, stride=1, stddev_factor=stddev_factor)

        self.add_sum(bypass)

        return self

    def add_max_pool(self, pool_size=[2,2],strides=2):
        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            pool = tf.layers.max_pooling2d(inputs=self.get_output(), pool_size=pool_size, strides=strides)

        self.outputs.append(pool)
        return self

    # ggg
    def add_Unet1Step(self, num_units, mapsize=3, stride=2, num_layers=2, stddev_factor=1e-3):
        assert len(self.get_output().get_shape()) == 4 and "Previous layer must be 4-dimensional (batch, width, height, channels)"

        # Add projection in series if needed prior to shortcut
        if num_units != int(self.get_output().get_shape()[3]):
            self.add_conv2d(num_units, mapsize=1, stride=1, stddev_factor=1.)

        bypass = self.get_output()

        self.add_conv2d(num_units, mapsize=mapsize, stride=2, stddev_factor=stddev_factor)
        self.add_relu()

        self.add_conv2d_transpose(num_units, mapsize=mapsize, stride=stride, stddev_factor=1.)
        self.add_relu()


        # Residual block
        #for _ in range(num_layers):
        #   self.add_batch_norm()
        #  self.add_relu()
        
        #out = tf.concat( [bypass, self.get_output()], 3)
        #self.outputs.append(out)

        self.add_sum(bypass)

        return self

    def add_UnetKsteps(self, num_units, mapsize=3, stride=2, stddev_factor=1e-3):
        assert len(self.get_output().get_shape()) == 4 and "Previous layer must be 4-dimensional (batch, width, height, channels)"

        nLines=num_units.shape[0]
        CurLine=num_units[0]
        BeforeD=CurLine[0:3]
        AfterD=CurLine[5:8]


        for i in BeforeD:
            if i>0:
                print("add conv2D+relu %d" % (i))
                self.add_conv2d(i, mapsize=mapsize, stride=1, stddev_factor=stddev_factor)
                self.add_elu()

        # call recursive
        if nLines>1:
            bypass = self.get_output()
            
            # go down
            self.add_conv2d(CurLine[3], mapsize=mapsize, stride=stride, stddev_factor=stddev_factor)
            self.add_elu()
            print("add conv2D+relu %d stride %d" % (CurLine[3],stride))
            # or alternatively add_max_pool(self, pool_size=[2,2],strides=2):

            self.add_UnetKsteps(num_units=num_units[1:], mapsize=mapsize, stride=stride, stddev_factor=stddev_factor)

            # go up
            self.add_conv2d_transpose(CurLine[4], mapsize=mapsize, stride=stride, stddev_factor=stddev_factor)
            self.add_elu()
            print("add conv2DT+relu %d stride %d" % (CurLine[3],stride))

            # concat
            out = tf.concat( [bypass, self.get_output()], 3)
            self.outputs.append(out)

        for i in AfterD:
            if i>0:
                print("add conv2D+relu %d" % (i))
                self.add_conv2d(i, mapsize=mapsize, stride=1, stddev_factor=stddev_factor)
                self.add_elu()
        
        return self

    def add_bottleneck_residual_block(self, num_units, mapsize=3, stride=1, transpose=False):
        """Adds a bottleneck residual block as per Arxiv 1512.03385, Figure 3"""

        assert len(self.get_output().get_shape()) == 4 and "Previous layer must be 4-dimensional (batch, width, height, channels)"

        # Add projection in series if needed prior to shortcut
        if num_units != int(self.get_output().get_shape()[3]) or stride != 1:
            ms = 1 if stride == 1 else mapsize
            #bypass.add_batch_norm() # TBD: Needed?
            if transpose:
                self.add_conv2d_transpose(num_units, mapsize=ms, stride=stride, stddev_factor=1.)
            else:
                self.add_conv2d(num_units, mapsize=ms, stride=stride, stddev_factor=1.)

        bypass = self.get_output()

        # Bottleneck residual block
        self.add_batch_norm()
        self.add_relu()
        self.add_conv2d(num_units//4, mapsize=1,       stride=1,      stddev_factor=2.)

        self.add_batch_norm()
        self.add_relu()
        if transpose:
            self.add_conv2d_transpose(num_units//4,
                                      mapsize=mapsize,
                                      stride=1,
                                      stddev_factor=2.)
        else:
            self.add_conv2d(num_units//4,
                            mapsize=mapsize,
                            stride=1,
                            stddev_factor=2.)

        self.add_batch_norm()
        self.add_relu()
        self.add_conv2d(num_units,    mapsize=1,       stride=1,      stddev_factor=2.)

        self.add_sum(bypass)

        return self

    def add_sum(self, term):
        """Adds a layer that sums the top layer with the given term"""

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            prev_shape = self.get_output().get_shape()
            term_shape = term.get_shape()
            #print("%s %s" % (prev_shape, term_shape))
            assert prev_shape == term_shape and "Can't sum terms with a different size"
            out = tf.add(self.get_output(), term)
        
        self.outputs.append(out)
        return self

    def add_mean(self):
        """Adds a layer that averages the inputs from the previous layer"""

        with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
            prev_shape = self.get_output().get_shape()
            reduction_indices = list(range(len(prev_shape)))
            assert len(reduction_indices) > 2 and "Can't average a (batch, activation) tensor"
            reduction_indices = reduction_indices[1:-1]
            out = tf.reduce_mean(self.get_output(), reduction_indices=reduction_indices)
            #out = tf.reduce_mean(self.get_output(), axis=reduction_indices)
        
        self.outputs.append(out)
        return self

    def add_upscale(self):
        """Adds a layer that upscales the output by 2x through nearest neighbor interpolation"""

        prev_shape = self.get_output().get_shape()
        size = [2 * int(s) for s in prev_shape[1:3]]
        out  = tf.image.resize_nearest_neighbor(self.get_output(), size)

        self.outputs.append(out)
        return self        

    def get_output(self):
        """Returns the output from the topmost layer of the network"""
        return self.outputs[-1]

    def get_variable(self, layer, name):
        """Returns a variable given its layer and name.

        The variable must already exist."""

        scope      = self._get_layer_str(layer)
        collection = tf.get_collection(tf.GraphKeys.VARIABLES, scope=scope)

        # TBD: Ugly!
        for var in collection:
            if var.name[:-2] == scope+'/'+name:
                return var

        return None

    def get_all_layer_variables(self, layer):
        """Returns all variables in the given layer"""
        scope = self._get_layer_str(layer)
        return tf.get_collection(tf.GraphKeys.VARIABLES, scope=scope)

    # ggg
    def add_constMatMul(self):
        ConstS=scipy.io.loadmat('/home/a/TF/AdjMRB.mat')
        ConstM=ConstS['AdjMRB']
        nChosen=FLAGS.nChosen*2 # ConstS['nChosen']
        ConstMf=np.float32(ConstM)
        #ConstMf=np.repeat(ConstMf, 16, axis=0)
        #ConstM = np.array(np.random.choice([0, 1], size=(1,64,64,3)), dtype='float32')
        ConstMtf1 = tf.constant(ConstMf, shape=[1,1,nChosen,64*64*2])
        ConstMtf=   tf.tile(ConstMtf1,[16,1,1,1])
        #ConstMtf = tf.constant(ConstMf, shape=[16,1,64*64*2,64*64*2])

        #W_filtered = tf.mul(connectivity, W)
        #out     = tf.multiply(self.get_output(), connectivity) # + bias
        P=tf.transpose(self.get_output(), perm=[0, 1,3,2])
        #out = tf.reshape(P, [16,64,64,2])
        #out = tf.reshape(self.get_output(), [16,64,64,2])
        M=tf.matmul(P,ConstMtf) # now its [Nbatch 1 1 8192] real and then image
        out = tf.reshape(M, [16,2,64,64])
        out=tf.transpose(out, perm=[0, 3,2,1])
        #out=tf.transpose(x, perm=[1, 0])
        self.outputs.append(out)
        return self
        #"""Adds a 2D convolutional layer."""

        #assert len(self.get_output().get_shape()) == 4 and "Previous layer must be 4-dimensional (batch, width, height, channels)"       
        #with tf.variable_scope(self._get_layer_str(), reuse=tf.AUTO_REUSE):
        #    prev_units = self._get_num_inputs()
        #    
        #    # Weight term and convolution
        #    initw  = self._glorot_initializer_conv2d(prev_units, num_units,
        #                                             mapsize,
        #                                             stddev_factor=stddev_factor)
        #    weight = tf.get_variable('weight', initializer=initw)
        #    out    = tf.nn.conv2d(self.get_output(), weight,
        #                          strides=[1, stride, stride, 1],
        #                          padding='SAME')

        #    # Bias term
        #    initb  = tf.constant(0.0, shape=[num_units])
        #    bias   = tf.get_variable('bias', initializer=initb)
        #    out    = tf.nn.bias_add(out, bias)
            
        #self.outputs.append(out)
        #return self

def _discriminator_model(sess, features, disc_input):
    # Fully convolutional model
    mapsize = 3
    layers  = [64, 128, 256, 512]

    old_vars = tf.global_variables()

    model = Model('DIS', 2*disc_input - 1)

    #RunDiscriminator= FLAGS.gene_l1_factor < 0.999
    RunDiscriminator= FLAGS.gene_l1_factor < 2

    if RunDiscriminator:
        for layer in range(len(layers)):
            nunits = layers[layer]
            stddev_factor = 2.0

            model.add_conv2d(nunits, mapsize=mapsize, stride=2, stddev_factor=stddev_factor)
            model.add_batch_norm()
            model.add_relu()

        # Finalization a la "all convolutional net"
        model.add_conv2d(nunits, mapsize=mapsize, stride=1, stddev_factor=stddev_factor)
        model.add_batch_norm()
        model.add_relu()

        model.add_conv2d(nunits, mapsize=1, stride=1, stddev_factor=stddev_factor)
        model.add_batch_norm()
        model.add_relu()

        # Linearly map to real/fake and return average score
        # (softmax will be applied later)
        model.add_conv2d(1, mapsize=1, stride=1, stddev_factor=stddev_factor)
    else:
        stddev_factor = 2.0
        model.add_conv2d(1, mapsize=1, stride=1, stddev_factor=stddev_factor)
        #model.add_flatten()

    model.add_mean()

    new_vars  = tf.global_variables()
    disc_vars = list(set(new_vars) - set(old_vars))

    return model.get_output(), disc_vars


def _generator_model(sess, features, labels, channels):
    # Upside-down all-convolutional resnet

    mapsize = 3
    mapsize = myParams.myDict['MapSize']
    res_units  = [256, 128, 96]

    old_vars = tf.global_variables()

    # See Arxiv 1603.05027
    model = Model('GEN', features)

    # H=FLAGS.LabelsH;
    # W=FLAGS.LabelsW;
    H=myParams.myDict['LabelsH']
    W=myParams.myDict['LabelsW']

    print("_generator_model")
    print("%d %d" % (H, W))

    # AUTOMAP
    # AUTOMAP_units  = [64, 64, channels]
    # AUTOMAP_mapsize  = [5, 5, 7]

    model.add_flatten() # FC1
    
    model.add_dense(num_units=H*W*2)
    model.add_reshapeTo4D(H,W)

    #model.add_denseFromM('piMDR')
    #model.add_reshapeTo4D(FLAGS.LabelsH,FLAGS.LabelsW)
    # #model.add_tanh() # FC2

    #model.add_Unet1Step(128, mapsize=5, stride=2, num_layers=2, stddev_factor=1e-3)
    #model.add_conv2d(channels, mapsize=5, stride=1, stddev_factor=2.)

    b=np.array([[64,0,0,128,128,0,0,64],[128,0,0,256,256,0,0,128],[512,0,0,0,0,0,0,512]])
    #b=np.array([[64,0,0,128,128,0,0,64],[128,0,0,256,256,0,0,128]])
    #b=np.array([[64,0,0,0,0,0,0,64]])

    model.add_UnetKsteps(b, mapsize=mapsize, stride=2, stddev_factor=1e-3)
    model.add_conv2d(channels, mapsize=1, stride=1, stddev_factor=2.)


    # #model.add_flatten()
    # #model.add_dense(num_units=H*W*1)
    # model.add_reshapeTo4D(FLAGS.LabelsH,FLAGS.LabelsW)
    # #model.add_batch_norm()
    # #model.add_tanh() # TC3

    # # model.add_conv2d(AUTOMAP_units[0], mapsize=AUTOMAP_mapsize[0], stride=1, stddev_factor=2.)
    # # model.add_batch_norm()
    # #model.add_relu()

    # #model.add_conv2d(AUTOMAP_units[1], mapsize=AUTOMAP_mapsize[1], stride=1, stddev_factor=2.)
    # # model.add_batch_norm()
    # #model.add_relu()

    # #model.add_conv2d(AUTOMAP_units[2], mapsize=AUTOMAP_mapsize[2], stride=1, stddev_factor=2.)
    # # model.add_conv2d(AUTOMAP_units[2], mapsize=1, stride=1, stddev_factor=2.)
    # # model.add_relu()


    #model.add_constMatMul()
    #for ru in range(len(res_units)-1):
    #    nunits  = res_units[ru]

    #    for j in range(2):
    #        model.add_residual_block(nunits, mapsize=mapsize)

        # Spatial upscale (see http://distill.pub/2016/deconv-checkerboard/)
        # and transposed convolution
    #    model.add_upscale()
        
    #    model.add_batch_norm()
    #    model.add_relu()
    #    model.add_conv2d_transpose(nunits, mapsize=mapsize, stride=1, stddev_factor=1.)

    # model.add_flatten()
    # model.add_dense(num_units=H*W*4)
    # model.add_reshapeTo4D(FLAGS.LabelsH,FLAGS.LabelsW)

    # #model.add_Mult2D()
    # #model.add_Mult3DComplexRI()

    SrezOrigImagePartModel=False
    if SrezOrigImagePartModel:
        nunits  = res_units[0]
        for j in range(2):
            model.add_residual_block(nunits, mapsize=mapsize)
        #model.add_upscale()
        model.add_batch_norm()
        model.add_relu()
        model.add_conv2d_transpose(nunits, mapsize=mapsize, stride=1, stddev_factor=1.)

        nunits  = res_units[1]
        for j in range(2):
            model.add_residual_block(nunits, mapsize=mapsize)
        #model.add_upscale()
        model.add_batch_norm()
        model.add_relu()
        model.add_conv2d_transpose(nunits, mapsize=mapsize, stride=1, stddev_factor=1.)

        # Finalization a la "all convolutional net"
        nunits = res_units[-1]
        model.add_conv2d(nunits, mapsize=mapsize, stride=1, stddev_factor=2.)
        # Worse: model.add_batch_norm()
        model.add_relu()

        model.add_conv2d(nunits, mapsize=1, stride=1, stddev_factor=2.)
        # Worse: model.add_batch_norm()
        model.add_relu()

        # Last layer is sigmoid with no batch normalization
        model.add_conv2d(channels, mapsize=1, stride=1, stddev_factor=1.)
        model.add_sigmoid()
    
    new_vars  = tf.global_variables()
    gene_vars = list(set(new_vars) - set(old_vars))

    ggg = tf.identity(model.get_output(), name="ggg")

    return model.get_output(), gene_vars

def create_model(sess, features, labels):
    # Generator
    rows      = int(features.get_shape()[1])
    cols      = int(features.get_shape()[2])
    channelsIn  = int(features.get_shape()[3])
    channelsOut  = int(labels.get_shape()[3])

    gene_minput = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, rows, cols, channelsIn])

    # TBD: Is there a better way to instance the generator?
    with tf.variable_scope('gene') as scope:
        gene_output, gene_var_list = _generator_model(sess, features, labels, channelsOut)

        scope.reuse_variables()

        gene_moutput, _ = _generator_model(sess, gene_minput, labels, channelsOut)
    
    # Discriminator with real data
    disc_real_input = tf.identity(labels, name='disc_real_input')

    # TBD: Is there a better way to instance the discriminator?
    with tf.variable_scope('disc') as scope:
        disc_real_output, disc_var_list = _discriminator_model(sess, features, disc_real_input)

        scope.reuse_variables()
            
        disc_fake_output, _ = _discriminator_model(sess, features, gene_output)

    return [gene_minput,      gene_moutput,
            gene_output,      gene_var_list,
            disc_real_output, disc_fake_output, disc_var_list]

#def _downscale(images, K):
#    """Differentiable image downscaling by a factor of K"""
#    arr = np.zeros([K, K, 3, 3])
#    arr[:,:,0,0] = 1.0/(K*K)
#    arr[:,:,1,1] = 1.0/(K*K)
#    arr[:,:,2,2] = 1.0/(K*K)
#    dowscale_weight = tf.constant(arr, dtype=tf.float32)
#    
#    downscaled = tf.nn.conv2d(images, dowscale_weight,
#                              strides=[1, K, K, 1],
#                              padding='SAME')
#    return downscaled

def create_generator_loss(disc_output, gene_output, features, labels):
    # I.e. did we fool the discriminator?
    #cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=disc_output, logits=tf.ones_like(disc_output))
    #gene_ce_loss  = tf.reduce_mean(cross_entropy, name='gene_ce_loss')

    ls_loss_real = tf.square(disc_output - tf.ones_like(disc_output))
    gene_ce_loss  = tf.reduce_mean(ls_loss_real, name='gene_ce_loss')

    # I.e. does the result look like the feature?
    #K = int(gene_output.get_shape()[1])//int(features.get_shape()[1])
    #assert K == 2 or K == 4 or K == 8    
    #downscaled = _downscale(gene_output, K)
    #downscaled = gene_output
    
    #gene_l1_loss  = tf.reduce_mean(tf.abs(downscaled - features), name='gene_l1_loss')

    gene_l1_loss  = tf.reduce_mean(tf.abs(gene_output - labels), name='gene_l1_loss')

    gene_loss     = tf.add((1.0 - FLAGS.gene_l1_factor) * gene_ce_loss,
                           FLAGS.gene_l1_factor * gene_l1_loss, name='gene_loss')
    
    return gene_loss

def create_discriminator_loss(disc_real_output, disc_fake_output):
    # I.e. did we correctly identify the input as real or not?
    #cross_entropy_real = tf.nn.sigmoid_cross_entropy_with_logits(labels=disc_real_output, logits=tf.ones_like(disc_real_output))
    #cross_entropy_real = tf.nn.softmax_cross_entropy_with_logits_v2(labels=disc_real_output, logits=tf.ones_like(disc_real_output))
    #cross_entropy_real = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(disc_real_output), logits=disc_real_output)
    #disc_real_loss     = tf.reduce_mean(cross_entropy_real, name='disc_real_loss')
    
    #cross_entropy_fake = tf.nn.sigmoid_cross_entropy_with_logits(labels=disc_fake_output, logits=tf.zeros_like(disc_fake_output))
    #disc_fake_loss     = tf.reduce_mean(cross_entropy_fake, name='disc_fake_loss')

    ls_loss_real = tf.square(disc_real_output - tf.ones_like(disc_real_output))
    disc_real_loss = tf.reduce_mean(ls_loss_real, name='disc_real_loss')

    ls_loss_fake = tf.square(disc_fake_output)
    disc_fake_loss = tf.reduce_mean(ls_loss_fake, name='disc_fake_loss')

    return disc_real_loss, disc_fake_loss

def create_optimizers(gene_loss, gene_var_list,
                      disc_loss, disc_var_list):    
    # TBD: Does this global step variable need to be manually incremented? I think so.
    global_step    = tf.Variable(0, dtype=tf.int64,   trainable=False, name='global_step')
    learning_rate  = tf.placeholder(dtype=tf.float32, name='learning_rate')
    
    gene_opti = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       beta1=FLAGS.learning_beta1,
                                       name='gene_optimizer')
    # gene_opti = tf.train.AdamOptimizer(learning_rate=0.00001,
    #                                    beta1=0.9,
    #                                    beta2=0.999,
    #                                    name='gene_optimizer')
    
    disc_opti = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       beta1=FLAGS.learning_beta1,
                                       name='disc_optimizer')

    gene_minimize = gene_opti.minimize(gene_loss, var_list=gene_var_list, name='gene_loss_minimize', global_step=global_step)
    
    disc_minimize     = disc_opti.minimize(disc_loss, var_list=disc_var_list, name='disc_loss_minimize', global_step=global_step)
    
    return (global_step, learning_rate, gene_minimize, disc_minimize)
