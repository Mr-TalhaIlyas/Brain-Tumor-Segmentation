from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dropout
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate, add





def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),kernel_initializer = 'he_normal', padding = 'same')(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def U_net(input_img, n_filters = 32, dropout = 0.1, batchnorm = True):
    """Function to define the UNET Model"""
    # Contracting Path
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    
    # Expanding Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    
    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    
    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    outputs = Conv2D(6, (1, 1), activation='softmax')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model
#%%
def ES(input_tensor, n_filters):
    # Wide Conv
    x1 = Conv2D(n_filters, kernel_size = (15,1), kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    x1 = Conv2D(n_filters, kernel_size = (1,15), kernel_initializer = 'he_normal', padding = 'same')(x1)
 
    x2 = Conv2D(n_filters, kernel_size = (13,1), kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    x2 = Conv2D(n_filters, kernel_size = (1,13), kernel_initializer = 'he_normal', padding = 'same')(x2)
    
    x3 = Conv2D(n_filters, kernel_size = (11,1), kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    x3 = Conv2D(n_filters, kernel_size = (1,11), kernel_initializer = 'he_normal', padding = 'same')(x3)
    
    x4 = Conv2D(n_filters, kernel_size = (9,1), kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    x4 = Conv2D(n_filters, kernel_size = (1,9), kernel_initializer = 'he_normal', padding = 'same')(x4)
    
    xadd = add([x1,x2, x3, x4])
    xadd = Conv2D(n_filters, kernel_size = (3,3), kernel_initializer = 'he_normal', padding = 'same')(xadd)
    xadd = Conv2D(n_filters, kernel_size = (3,3), kernel_initializer = 'he_normal', padding = 'same')(xadd)

    xskip = Conv2D(n_filters, kernel_size = (1,1), kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    
    x_op = concatenate([xskip, xadd])
    return x_op   
def WC(input_tensor, n_filters, kernel_size):

    # Wide Conv
    x1 = Conv2D(n_filters, kernel_size = (kernel_size,1), kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    x1 = Conv2D(n_filters, kernel_size = (1,kernel_size), kernel_initializer = 'he_normal', padding = 'same')(x1)
    

    x2 = Conv2D(n_filters, kernel_size = (1,kernel_size), kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    x2 = Conv2D(n_filters, kernel_size = (kernel_size,1), kernel_initializer = 'he_normal', padding = 'same')(x2)    
    
    xa = add([x1,x2])
    return xa
def BU_net(input_img, n_filters, dropout, batchnorm = True):
    c0 = Conv2D(n_filters, kernel_size = (7, 7), kernel_initializer = 'he_normal', padding = 'same')(input_img)
    c0 = BatchNormalization()(c0)
    c0 = Activation('relu')(c0)
    # Contracting Path
    c1 = conv2d_block(c0, n_filters*1, kernel_size = 3, batchnorm = True)
    c1 = conv2d_block(c1, n_filters*1, kernel_size = 3, batchnorm = True)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = conv2d_block(p1, n_filters*2, kernel_size = 3, batchnorm = True)
    c2 = conv2d_block(c2, n_filters*2, kernel_size = 3, batchnorm = True)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = conv2d_block(p2, n_filters*2, kernel_size = 3, batchnorm = True)
    c3 = conv2d_block(c3, n_filters*2, kernel_size = 3, batchnorm = True)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = conv2d_block(p3, n_filters*2, kernel_size = 3, batchnorm = True)
    c4 = conv2d_block(c4, n_filters*2, kernel_size = 3, batchnorm = True)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters*2, kernel_size = 3, batchnorm = True)
    c5 = conv2d_block(c5, n_filters*2, kernel_size = 3, batchnorm = True)
    c5 = Dropout(dropout)(c5)
    c6 = conv2d_block(c5, n_filters*2, kernel_size = 3, batchnorm = True)
    c6 = conv2d_block(c6, n_filters*2, kernel_size = 3, batchnorm = True)
    
    #Transition
    ctr = WC(c6, n_filters*16, kernel_size=13)
    
    # Expanding Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(ctr)
    c4_es = ES(c4, n_filters * 8)
    u6 = concatenate([u6, c4_es])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters*8, kernel_size = 3, batchnorm = True)
    
    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    c3_es = ES(c3, n_filters * 4)
    u7 = concatenate([u7, c3_es])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters*4, kernel_size = 3, batchnorm = True)
    
    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    c2_es = ES(c2, n_filters * 2)
    u8 = concatenate([u8, c2_es])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters*2, kernel_size = 3, batchnorm = True)
    
    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    c1_es = ES(c1, n_filters * 1)
    u9 = concatenate([u9, c1_es])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters*1, kernel_size = 3, batchnorm = True)
    
    outputs = Conv2D(6, (1, 1), activation='softmax')(c9)#, activation='softmax
    model = Model(inputs=[input_img], outputs=[outputs])
    return model 