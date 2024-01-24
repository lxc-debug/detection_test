### 对于验证实验来说主要就是以下几个py文件
+ main.py是函数的主入口，里面是固定随机数种子以及整个函数运行的主逻辑
+ dataset.py完成了数据的读入，包括从模型转为onnx格式文件，以及onnx文件转为张量
+ .utils/from_onnx_to_dgl.py 这个文件中包含了model2onnx以及onnx2dgl两个函数，后面多出来的两个函数是主试验的部分，不包括在验证实验，就没有加注释
+ .model/simple_model.py 这个文件有两个模型，SimpleModel以及SimpleModelQ。SimpleModel是面对不引入结构信息，以及直接拼接结构信息的情况；SimpleModelQ面对的是使用结构信息作为query的情况
+ train.py是模型的训练文件
+ 所有的配置信息都在.config/conf.py文件中