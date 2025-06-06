from cuda import cudart
#from datetime import datetime as dt
import numpy as np
import onnx
import onnx_graphsurgeon as gs
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import config_pb2, meta_graph_pb2, rewriter_config_pb2
from tensorflow.python.framework import importer, ops
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.training import saver
import tensorrt as trt
import cv2

#dataPath = os.path.dirname(os.path.realpath(__file__)) + "/../../00-MNISTData/"
#sys.path.append(dataPath)
#import loadMnistData

tf2.compat.v1.disable_eager_execution()
np.random.seed(31193)
tf2.compat.v1.set_random_seed(97)
nTrainbatchSize = 256
ckptFile = "./model.ckpt"
pbFile = "model-V1.pb"
pb2File = "model-V2.pb"
onnxFile = "model-V1.onnx"
onnx2File = "model-V2.onnx"
trtFile = "model.plan"
#inferenceImage = dataPath + "8.png"
outputNodeName = "z"
isRemoveTransposeNode = False  # 变量说明见用到该变量的地方
isAddQDQForInput = False  # 变量说明见用到该变量的地方

#===============================================================================
from cuda import cudart
import cv2
from datetime import datetime as dt
from glob import glob
import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf2
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import tensorrt as trt

np.random.seed(31193)
tf2.random.set_seed(97)
nTrainBatchSize = 128
nHeight = 28
nWidth = 28
pbFilePath = "./model-NCHW/"
pbFile = "model-NCHW.pb"
onnxFile = "./model-NCHW.onnx"
trtFile = "./model-NCHW.plan"
dataPath = os.path.dirname(os.path.realpath(__file__)) + "/../../00-MNISTData/"
trainFileList = sorted(glob(dataPath + "train/*.jpg"))
testFileList = sorted(glob(dataPath + "test/*.jpg"))
inferenceImage = dataPath + "8.png"

# for FP16 mode
bUseFP16Mode = False
# for INT8 model
bUseINT8Mode = False
nCalibration = 1
cacheFile = "./int8.cache"
calibrationDataPath = dataPath + "test/"

#===============================================================================
checkpointPath = "./checkpoint/model"
checkpointSuffix = "-1"
#===============================================================================

#os.system("rm -rf %s ./*.plan ./*.cache" % pbFilePath)
np.set_printoptions(precision=3, linewidth=100, suppress=True)
tf2.config.experimental.set_memory_growth(tf2.config.list_physical_devices("GPU")[0], True)
cudart.cudaDeviceSynchronize()

def getData(fileList):
    nSize = len(fileList)
    xData = np.zeros([nSize, nHeight, nWidth, 1], dtype=np.float32)
    yData = np.zeros([nSize, 10], dtype=np.float32)
    for i in range(nSize):
        imageName = fileList[i]
        data = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)
        label = np.zeros(10, dtype=np.float32)
        label[int(imageName[-7])] = 1
        xData[i] = data.reshape(nHeight, nWidth, 1).astype(np.float32) / 255
        yData[i] = label
    return xData, yData

# TensorFlow 中创建网络并保存为 .pb 文件 -------------------------------------------
modelInput = tf2.keras.Input(shape=[nHeight, nWidth, 1], dtype=tf2.dtypes.float32)

layerConv1 = tf2.keras.layers.Conv2D(32, [5, 5], strides=[1, 1], padding="same", data_format=None, dilation_rate=[1, 1], groups=1, activation="relu", use_bias=True, kernel_initializer="glorot_uniform", bias_initializer="zeros", kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, name="conv1")
x = layerConv1(modelInput)

layerPool1 = tf2.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding="same", data_format=None, name="pool1")
x = layerPool1(x)

layerConv2 = tf2.keras.layers.Conv2D(64, [5, 5], strides=[1, 1], padding="same", data_format=None, dilation_rate=[1, 1], groups=1, activation="relu", use_bias=True, kernel_initializer="glorot_uniform", bias_initializer="zeros", kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, name="conv2")
x = layerConv2(x)

laerPool2 = tf2.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding="same", data_format=None, name="pool2")
x = laerPool2(x)

layerReshape = tf2.keras.layers.Reshape([-1], name="reshape")
x = layerReshape(x)

layerDense1 = tf2.keras.layers.Dense(1024, activation="relu", use_bias=True, kernel_initializer="glorot_uniform", bias_initializer="zeros", kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, name="dense1")
x = layerDense1(x)

layerDense2 = tf2.keras.layers.Dense(10, activation=None, use_bias=True, kernel_initializer="glorot_uniform", bias_initializer="zeros", kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, name="dense2")
x = layerDense2(x)

layerSoftmax = tf2.keras.layers.Softmax(axis=1, name="softmax")
z = layerSoftmax(x)

model = tf2.keras.Model(inputs=modelInput, outputs=z, name="MNISTExample")

model.summary()

model.compile(
    loss=tf2.keras.losses.CategoricalCrossentropy(from_logits=False),
    optimizer=tf2.keras.optimizers.Adam(),
    metrics=["accuracy"],
)

xTrain, yTrain = getData(trainFileList)
#history = model.fit(xTrain, yTrain, batch_size=128, epochs=10, validation_split=0.1)
history = model.fit(xTrain, yTrain, batch_size=128, epochs=1, validation_split=0.1)

xTest, yTest = getData(testFileList)
testScore = model.evaluate(xTest, yTest, verbose=2)
print("%s, loss = %f, accuracy = %f" % (dt.now(), testScore[0], testScore[1]))

#checkpoint = tf2.train.Checkpoint(model)
#checkpoint.save(checkpointPath)
#print("Succeeded saving .ckpt in TensorFlow!")

import tensorflow_model_optimization as tfmot

quantize_model = tfmot.quantization.keras.quantize_model

# q_aware stands for for quantization aware.
q_aware_model = quantize_model(model)

q_aware_model.summary()

# `quantize_model` requires a recompile.
q_aware_model.compile(
    loss=tf2.keras.losses.CategoricalCrossentropy(from_logits=False),
    optimizer=tf2.keras.optimizers.Adam(),
    metrics=["accuracy"],
)

train_images_subset = xTrain[:1000]  # out of 3000
train_labels_subset = yTrain[:1000]

q_aware_model.fit(train_images_subset, train_labels_subset, batch_size=100, epochs=1, validation_split=0.1)

_, baseline_model_accuracy = model.evaluate(xTest, yTest, verbose=0)

_, q_aware_model_accuracy = q_aware_model.evaluate(xTest, yTest, verbose=0)

print('Baseline test accuracy:', baseline_model_accuracy)
print('Quant test accuracy:', q_aware_model_accuracy)

#----------------
"""
# TensorFlow 中创建推理网络并保存为 .pb -----------------------------------------
modelInput = tf2.keras.Input(shape=[nHeight, nWidth, 1], dtype=tf2.dtypes.float32)

layerConv1 = tf2.keras.layers.Conv2D(32, [5, 5], strides=[1, 1], padding="same", data_format=None, dilation_rate=[1, 1], groups=1, activation="relu", use_bias=True, kernel_initializer="glorot_uniform", bias_initializer="zeros", kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, name="conv1")
x = layerConv1(modelInput)

layerPool1 = tf2.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding="same", data_format=None, name="pool1")
x = layerPool1(x)

layerConv2 = tf2.keras.layers.Conv2D(64, [5, 5], strides=[1, 1], padding="same", data_format=None, dilation_rate=[1, 1], groups=1, activation="relu", use_bias=True, kernel_initializer="glorot_uniform", bias_initializer="zeros", kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, name="conv2")
x = layerConv2(x)

laerPool2 = tf2.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding="same", data_format=None, name="pool2")
x = laerPool2(x)

layerReshape = tf2.keras.layers.Reshape([-1], name="reshape")
x = layerReshape(x)

layerDense1 = tf2.keras.layers.Dense(1024, activation="relu", use_bias=True, kernel_initializer="glorot_uniform", bias_initializer="zeros", kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, name="dense1")
x = layerDense1(x)

layerDense2 = tf2.keras.layers.Dense(10, activation=None, use_bias=True, kernel_initializer="glorot_uniform", bias_initializer="zeros", kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, name="dense2")
x = layerDense2(x)

layerSoftmax = tf2.keras.layers.Softmax(axis=1, name="softmax")
z = layerSoftmax(x)

w = tf2.math.argmax(z)

model = tf2.keras.Model(inputs=modelInput, outputs=w, name="MNISTExample2")

checkpoint = tf2.train.Checkpoint(model)
checkpoint.restore(checkpointPath + checkpointSuffix)

tf2.quantize.experimental_create_eval_graph(symmetric=True, use_qdq=True)

print("Finish!")
exit()
tf2.contrib.quantize.experimental_create_eval_graph(symmetric=True, use_qdq=True)

with tf2.Session(graph=g2) as sess:
    tf2.compat.v1.train.Saver().restore(sess, ckptFile)
    constantGraph = tf2.graph_util.convert_variables_to_constants(sess, g2.as_graph_def(), [outputNodeName])
with tf2.gfile.FastGFile(pbFile, mode="wb") as f:
    f.write(constantGraph.SerializeToString())
print("Succeeded saving .pb in TensorFlow!")

# 优化 .pb ---------------------------------------------------------------------
with open(pbFile, "rb") as f:
    graphdef = graph_pb2.GraphDef()
    graphdef.ParseFromString(f.read())

graph = ops.Graph()
with graph.as_default():
    outputCollection = meta_graph_pb2.CollectionDef()
    for output in outputNodeName.split(','):
        outputCollection.node_list.value.append(output)
    importer.import_graph_def(graphdef, name="")
    metagraph = saver.export_meta_graph(graph_def=graph.as_graph_def(add_shapes=True), graph=graph)
    metagraph.collection_def["train_op"].CopyFrom(outputCollection)

rewriter_config = rewriter_config_pb2.RewriterConfig()
rewriter_config.optimizers.extend(["constfold"])
rewriter_config.meta_optimizer_iterations = (rewriter_config_pb2.RewriterConfig.ONE)
session_config = config_pb2.ConfigProto()
session_config.graph_options.rewrite_options.CopyFrom(rewriter_config)

folded_graph = tf_optimizer.OptimizeGraph(session_config, metagraph)

with open(pb2File, "wb") as f:
    f.write(folded_graph.SerializeToString())
print("Succeeded optimizing .pb in TensorFlow!")

# 将 .pb 文件转换为 .onnx 文件 --------------------------------------------------
os.system("python3 -m tf2onnx.convert --opset 11 --input %s --output %s --inputs 'input_0:0' --outputs '%s:0' --inputs-as-nchw 'x:0'" % (pb2File, onnxFile, outputNodeName))
print("Succeeded converting model into ONNX!")

# 优化 .onnx 文件，去除 Conv 前的 Transpose 节点 --------------------------------
graph = gs.import_onnx(onnx.load(onnxFile))

# 原 repo 中解释，导出的计算图中 Conv 的 Weight 输入前会有一个 Transpose 节点，并且 TensorRT QAT 模式不支持这个节点，这里用于手工转置并去除该 Transpose 节点
# 但是在目前导出的计算图中已经没有了这个节点，不再需要这一步
if isRemoveTransposeNode:
    for node in [n for n in graph.nodes if n.op == "Conv"]:
        convKernelTensor = node.i(1).i().i().inputs[0]
        convKernelTensor.values = convKernelTensor.values.transpose(3, 2, 0, 1)
        node.inputs[1] = node.i(1).i(0).outputs[0]

onnx.save_model(gs.export_onnx(graph.cleanup().toposort()), onnx2File)
print("Succeeded optimizing .onnx in Onnx!")

# TensorRT 中加载 .onnx 创建 engine ---------------------------------------------
logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
networkFlag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) | (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION))
network = builder.create_network(networkFlag)
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.INT8)
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 3 << 30)
parser = trt.OnnxParser(network, logger)
if not os.path.exists(onnxFile):
    print("Failed finding ONNX file!")
    exit()
print("Succeeded finding ONNX file!")
with open(onnxFile, "rb") as model:
    if not parser.parse(model.read()):
        print("Failed parsing .onnx file!")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        exit()
    print("Succeeded parsing .onnx file!")

inputTensor = network.get_input(0)
inputTensor.shape = [-1, 1, 28, 28]
profile.set_shape(inputTensor.name, [1, 1, 28, 28], [4, 1, 28, 28], [16, 1, 28, 28])
config.add_optimization_profile(profile)

# 为所有输入张量添加 quantize 节点，这里使用经验值 127 <-> 1.0
# 理想状态下需要在 QAT 过程中获取这些取值并添加到输入节点上
if isAddQDQForInput:
    quantizeScale = np.array([1.0 / 127.0], dtype=np.float32)
    dequantizeScale = np.array([127.0 / 1.0], dtype=np.float32)
    one = np.array([1], dtype=np.float32)
    zero = np.array([0], dtype=np.float32)

    for i in range(network.num_inputs):
        inputTensor = network.get_input(i)

        for j in range(network.num_layers):
            layer = network.get_layer(j)

            for k in range(layer.num_inputs):
                if (layer.get_input(k) == inputTensor):
                    print(i, layer, k)
                    #quantizeLayer = network.add_scale(inputTensor, trt.ScaleMode.UNIFORM, zero, quantizeScale)
                    quantizeLayer = network.add_scale(inputTensor, trt.ScaleMode.UNIFORM, zero, one)
                    quantizeLayer.set_output_type(0, trt.int8)
                    quantizeLayer.name = "InputQuantizeNode"
                    quantizeLayer.get_output(0).name = "QuantizedInputTensor"
                    #dequantizeLayer = network.add_scale(quantizeLayer.get_output(0), trt.ScaleMode.UNIFORM, zero, dequantizeScale)
                    dequantizeLayer = network.add_scale(quantizeLayer.get_output(0), trt.ScaleMode.UNIFORM, zero, one)
                    dequantizeLayer.set_output_type(0, trt.float32)
                    dequantizeLayer.name = "InputDequantizeNode"
                    dequantizeLayer.get_output(0).name = "DequantizedInputTensor"
                    layer.set_input(k, dequantizeLayer.get_output(0))

engineString = builder.build_serialized_network(network, config)
if engineString == None:
    print("Failed building engine!")
    exit()
print("Succeeded building engine!")
with open(trtFile, "wb") as f:
    f.write(engineString)
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

context = engine.create_execution_context()
context.set_binding_shape(0, [1, 1, 28, 28])
_, stream = cudart.cudaStreamCreate()
print("Binding0->", engine.get_binding_shape(0), context.get_binding_shape(0), engine.get_binding_dtype(0))
print("Binding1->", engine.get_binding_shape(1), context.get_binding_shape(1), engine.get_binding_dtype(1))

data = cv2.imread(inferenceImage, cv2.IMREAD_GRAYSCALE).astype(np.float32)
inputH0 = np.ascontiguousarray(data.reshape(-1))
outputH0 = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))
_, inputD0 = cudart.cudaMallocAsync(inputH0.nbytes, stream)
_, outputD0 = cudart.cudaMallocAsync(outputH0.nbytes, stream)

cudart.cudaMemcpyAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
context.execute_async_v2([int(inputD0), int(outputD0)], stream)
cudart.cudaMemcpyAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
cudart.cudaStreamSynchronize(stream)

print("inputH0 :", data.shape)
#print(data)
print("outputH0:", outputH0.shape)
print(outputH0)

cudart.cudaStreamDestroy(stream)
cudart.cudaFree(inputD0)
cudart.cudaFree(outputD0)
print("Succeeded running model in TensorRT!")
"""