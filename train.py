from tensorflow.keras import callbacks, optimizers
from core.yolov4.tf import SaveWeightsCallback, YOLOv4
import time


import cv2


yolo = YOLOv4(tiny=True)
yolo.classes = "/home/pinaq/Projects/yolov4/data/flir.name"
yolo.input_size = 256
yolo.batch_size = 32

yolo.make_model()
yolo.load_weights(
    "/home/pinaq/Projects/yolov4/data/yolov4-tiny.conv.29",
    weights_type="yolo"
)

train_data_set = yolo.load_dataset(
    "/home/pinaq/Projects/yolov4/data/trainFlirYolov4.txt",
    image_path_prefix="",
    label_smoothing=0.05
)
val_data_set = yolo.load_dataset(
    "/home/pinaq/Projects/yolov4/data/valFlirYolov4.txt",
    image_path_prefix="",
    training=False
)
#print(train_data_set.dataset_type)
#vount = 0
#vvount = 0
#for data in train_data_set:#
#	vount=0
#	print(data[0].shape)
#	for __data in data[0]:
#		print("TLOOP")
#		cv2.imwrite("/home/pinaq/Projects/yolov4/tmp/{img}--{num}-loop.jpg".format(num=vount,img=vvount),__data*255)
#		vount = vount + 1
#	print(data[0].batch_size)
#	vvount = vvount + 1
#exit()
epochs = 100
lr = 1e-5

optimizer = optimizers.Adam(learning_rate=lr)
yolo.compile(optimizer=optimizer, loss_iou_type="ciou")

def lr_scheduler(epoch):
    if epoch < int(epochs * 0.5):
        return lr
    if epoch < int(epochs * 0.8):
        return lr * 0.5
    if epoch < int(epochs * 0.9):
        return lr * 0.1
    return lr * 0.01

_callbacks = [
    callbacks.LearningRateScheduler(lr_scheduler),
    callbacks.TerminateOnNaN(),
    callbacks.TensorBoard(
        log_dir="/home/pinaq/Projects/yolov4/data/logs",
    ),
    SaveWeightsCallback(
        yolo=yolo, dir_path="/home/pinaq/Projects/yolov4/trained",
        weights_type="yolo", epoch_per_save=1
    ),
]

yolo.fit(
    train_data_set,
    epochs=epochs,
    callbacks=_callbacks,
    validation_data=val_data_set,
    validation_steps=30,
    validation_freq=1,
    steps_per_epoch=180,
    verbose=1
)
