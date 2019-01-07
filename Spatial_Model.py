from keras.models import Model
from keras.layers import Conv2D, ReLU, MaxPooling2D, Lambda, Input, Layer
from keras.layers import concatenate, Dense, Activation, Dropout, Reshape, Flatten
from keras import backend as K

def create_model():
    input_y_h2 = Input(shape=(3,35))
    input_y_h1 = Input(shape=(3,35))
    input_x = Input(shape=(21,31,5))

    x_1 = Conv2D(64, (5, 5), padding='same', activation='relu',name='conv1_h1')(input_x)
    x_2 = Conv2D(64, (5, 5), padding='same', activation='relu',name='conv1_h2')(input_x)
    x_3 = Conv2D(64, (5, 5), padding='same', activation='relu',name='conv1_h3')(input_x)    
    x_1 = MaxPooling2D((2,2), name='MaxPool_h1_1')(x_1)
    x_2 = MaxPooling2D((2,2), name='MaxPool_h2_1')(x_2)
    x_3 = MaxPooling2D((2,2), name='MaxPool_h3_1')(x_3)    
    x_1 = Conv2D(32, (5, 5), activation='relu', name='conv2_h1')(x_1)
    x_2 = Conv2D(32, (5, 5), activation='relu', name='conv2_h2')(x_2)
    x_3 = Conv2D(32, (5, 5), activation='relu', name='conv2_h3')(x_3)      
    x_1 = MaxPooling2D((2,2), name='MaxPool_h1_2')(x_1)
    x_2 = MaxPooling2D((2,2), name='MaxPool_h2_2')(x_2)
    x_3 = MaxPooling2D((2,2), name='MaxPool_h3_2')(x_3)   
    x_1 = Conv2D(16, (5, 5), padding='same', name='conv3_h1')(x_1)
    x_2 = Conv2D(16, (5, 5), padding='same', name='conv3_h2')(x_2)
    x_3 = Conv2D(16, (5, 5), padding='same', name='conv3_h3')(x_3)     
    x_1 = Flatten()(x_1)
    x_2 = Flatten()(x_2)
    x_3 = Flatten()(x_3)    
    x_1 = Dense(35, name='dense_1')(x_1)
    x_2 = Dense(35, name='dense_2')(x_2)
    x_3 = Dense(35, name='dense_3')(x_3)
    x_1 = Lambda(lambda x: K.expand_dims(x, axis = 1), name='expand_dim_1')(x_1)
    x_2 = Lambda(lambda x: K.expand_dims(x, axis = 1), name='expand_dim_2')(x_2)
    x_3 = Lambda(lambda x: K.expand_dims(x, axis = 1), name='expand_dim_3')(x_3)
    
    y_1 = Dense(35, name='dense_h1')(input_y_h1)
    y_2 = Dense(35, name='dense_h2')(input_y_h2)

    x = Lambda(lambda x: K.concatenate([x[0],x[1],x[2]],axis=1), name='concatenate')([x_1,x_2,x_3])
    x = concatenate([x, y_1, y_2])
    output_layer = Dense(35, name='dense_output', activation='relu')(x)
    return Model(inputs=[input_y_h2, input_y_h1, input_x], outputs=output_layer)