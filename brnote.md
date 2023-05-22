###
TRT python API 和 C++ API 是对应互通的，生成的engine是可以互通的

计算图优化
显存优化，显存池复用

基本流程：
1. 构建期：建立builder(引擎构建器)； 创建network(计算图内容)；生成serializedNetwork(网络的TRT内部表示)；
2. 建立engine(可执行程序的代码段)和context(进程)；
buffer准备(host端和device端，拷贝操作)
执行推理(execute)

网络建立使用parser，遇到不支持的OP需要(改网，改parser, 写plugin)

Logger
builder 常用设置逐渐转移到builder config中
Dynamic shape模式下必须使用builder config进行设置

使用Explicit Batch模式， 输入tensor显示包含了batch维，比implicit batch多一维， Explicit batch模式可以很好的解决LayerNorm
需要builder.create_network(1 << int(tensorrt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))


Dynamic shape模式：除了Batch维，其他维度也可以在推理时决定。适用于输入张量形状在推理是才决定的网络。需要Explicit_batch模式；
需要构造一个optimization profile帮助网络优化；
需要context.set_binding_shape绑定实际输入数据的形状；

profile.set_shape(tensorname, minshape,commonshape, maxshape)给定输入张量的最小，最常见，最大尺寸；

区分layer(计算节点)和tensor(数据节点)
onelayer(指向这个layer的指针) = network.add_xxx()
oneTensor = onelayer.get_output(0) # gettensor from the layer
nextlayer = network.add_xxx(oneTensor) # take the tensor into next layer

权重迁移：

使用训练框架pytroch等将原模型权重保存成npz。(见03-APIModel)

逐层搭建，逐层检验输出。
先保证FP32模式结果正确后再逐步尝试FP16和INT8模式。
常见Layer范例(02-API/Layer等)

FP16模式：

INT8：
1. PTQ  04-Parser
不需要对模型进行改动
2. QAT

### RUNTIME
engine
Binding： engine/context给所有输入输出张量安排了位置，总共engine.num_bindings个binding, 输入张量在前，输出张量在后。
运行期绑定张量形状时，要指定位置绑定；
context.set_binding_shape(0, [4, 1, 28, 28])
context可以根据输入张量形状自动计算出输出张量的形状。这样可以在推理时申请合适输出显存的大小。

context
Buffer 内存和显存的申请
inputHost = np.ascontiguousarray(inputData.reshape(-1))
outputHost = np.empty(context.get_binding_shape(1), trt.nptype(engine.get_binding_dtype(1)))


序列化和反序列化
serializedNetwork文件包含硬件的优化。
需要确保环境统一(硬件，CUDA/cuDNN/TRT)
不同的TRT版本的engine不能相互兼容

同平台同环境多次生成的engine可能不同(选择了不同的算子实现)
04-Parser/pytorch中包含int8的实现

onnx parser trtexec
TRT不支持的节点：
1. 修改源模型
2. 修改onnx计算图  onnx-surgeon
3. 在TRT中实现plugin
4. 修改parser (部分开源的)

07 Torch-TRT Frame

推荐nvidia-docker ??



