"""
train your own dataset and predict
Written by LI Shuai
"""
import numpy as np
import  keras.backend as K
from keras.layers import Input,Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard,ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from yolo_v3.model.model import preprocess_true_boxes,\
    yolo_body,tiny_yolo_body,yolo_loss,get_random_data
def get_anchors(path):
    """
    todo
    :param path:
    :return:
    """
    with open(path) as f:
        anchors=f.readline()
    archors=[float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1,2)

def create_tiny_model(input_shape, anchors, num_classes,
                      load_pretrained=True, freeze_body=2,
            weights_path='yolo_v3/weight/tiny_yolo_weights.h5'):
    K.clear_session()
    image_input=Input(shape=(None,None,3))
    h,w=input_shape
    num_anchors=len(anchors)
    y_true = [Input(shape=(h//{0:32, 1:16}[l],
                           w//{0:32, 1:16}[l],num_anchors//2, num_classes+5)) for l in range(2)]
    model_body=tiny_yolo_body(image_input,num_anchors//2,num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))
    if load_pretrained:
        model_body.load_weights(weights_path,by_name=True,skip_mismatch=True)
        print("load weights {}",format(weights_path))
        if freeze_body in [1,2]:
            num=(20,len(model_body.layers)-2)[freeze_body-1]
            for i in range(num):
                model_body.layers[i].trainable=False
                print('Freeze the first {} layers of total {} layers'.format(num,len(model_body.layers)))
    model_loss=Lambda(yolo_loss,output_shape=(1,),name='yolo_loss',
                      arguments={'anchors':anchors,'num_classes':num_classes,'ignore_thresh':0.7})(
        [*model_body.output,*y_true]
    )
    model=Model([model_body.input,*y_true],model_loss)
    return model

def create_model(input_shape, anchors, num_classes,
                 load_pretrained=True, freeze_body=2,
            weights_path='yolo_v3/weight/yolo_weights.h5'):
    K.clear_session()
    image_input=Input(shape=(None,None,3))
    h,w=input_shape
    num_anchors=len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
            num_anchors//3, num_classes+5)) for l in range(3)]
    model_body=yolo_body(image_input,num_anchors//3,num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))
    if load_pretrained:
        model_body.load_weights(weights_path,by_name=True,skip_mismatch=True)
        print('load weights {}'.format(weights_path))
        if freeze_body in [1,2]:
            num=(185,len(model_body.layers)-3)[freeze_body-1]
            for i in range(num):
                model_body.layers[i].trainable=False
                print('Freeze the first {} layers of total {} layers.'
                      ''.format(num, len(model_body.layers)))
    model_loss=Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                       arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
                       [*model_body.output, *y_true])
    model=Model([model_body.input,*y_true],model_loss)
    return model

def data_generator(annotation_lines,batch_size,input_shape,anchors,num_classes):
    n=len(annotation_lines)
    i=0
    while True:
        image_data=[]
        box_data=[]
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image,box=get_random_data(annotation_lines[i],input_shape,random=True)
            image_data.append(image)
            box_data.append(box)
            i=(i+1)%n
        image_data=np.array(image_data)
        box_data=np.array(box_data)
        y_true=preprocess_true_boxes(box_data,input_shape,anchors,num_classes)
        yield [image_data,*y_true],np.zeros(batch_size)

def data_generator_wrapper(annotation_lines,batch_size,input_shape,anchors,num_classes):
    n=len(annotation_lines)
    if n==0 or batch_size<=0:return None
    return data_generator(annotation_lines,batch_size,input_shape,anchors,num_classes)





def train_yolo_v3(annotation_path,anchors_path):
    log_dir="logs/000/"

    #set according to your dataset"
    classes_names=None
    num_classes=len(classes_names)
    anchors=get_anchors(anchors_path)

    input_shape=(416,416)
    is_tiny_version=len(anchors)==6

    if is_tiny_version:
        model=create_tiny_model(
            input_shape,anchors,num_classes,freeze_body=2,
            weights_path='yolo_v3/weight/tiny_yolo_weights.h5'
        )
    else:
        model=create_model(input_shape,anchors,num_classes,freeze_body=2,
                           weights_path='yolo_v3/weight/yolo_weight.h5')
    logging=TensorBoard(log_dir=log_dir)
    checkpoint=ModelCheckpoint(log_dir+'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    val_split=0.1
    with open(annotation_path) as f:
        lines=f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val=int(len(lines)*val_split)
    num_train=len(lines)-num_val

    if True:
        model.compile(optimizer=Adam(lr=1e-3),loss={
            'yolo_loss':lambda y_ture,y_pred:y_pred
        })
        batch_size=16
        print('Train on {} samples, val on {} samples, with batch size {}.'
              ''.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train],batch_size,input_shape,
                                                   anchors,num_classes),
                            steps_per_epoch=max(1,num_train//batch_size),
                            validation_data=data_generator_wrapper(lines[num_train:],batch_size,input_shape,
                                                                   anchors,num_classes),
                            validation_steps=max(1,num_val//batch_size),
                            epochs=50,
                            initial_epoch=0,
                            callbacks=[logging,checkpoint]
                            )
        model.save_weights('yolo_v3/weight/yolo_v3_trained_weights_stage_1.h5')

        #unfreeze  and continue training, to fine tune
        if True:
            for i in range(len(model.layers)):
                model.layers[i].trainable=True
            model.compile(optimizer=Adam(1e-4),loss={'yolo_loss':lambda y_true,y_pred:y_pred})
            batch_size=16

            print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
            model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                        steps_per_epoch=max(1, num_train//batch_size),
                        validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                        validation_steps=max(1, num_val//batch_size),
                        epochs=100,
                        initial_epoch=50,
                        callbacks=[logging, checkpoint, reduce_lr, early_stopping])
            model.save_weights('yolo_v3/weight/yolo_v3_trained_weights_final.h5')
