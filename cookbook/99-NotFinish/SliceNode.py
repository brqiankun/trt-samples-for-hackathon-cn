#
# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from collections import OrderedDict
import numpy as np
import onnx
import onnx_graphsurgeon as gs

tensor0 = gs.Variable("tensor0", np.float32, ["B", 1, 1, 64])
tensor1 = gs.Variable("tensor1", np.float32, ["B", None, None, None])

constant0 = gs.Constant(name="constant0", values=np.array([0], dtype=np.int32))
constant2 = gs.Constant(name="constant2", values=np.array([2], dtype=np.int32))
constantM1 = gs.Constant(name="constantM1", values=np.array([-1], dtype=np.int32))
constant3 = gs.Constant(name="constant3", values=np.array([3], dtype=np.int32))

inputList = [
    tensor0,  # data
    constant0,  # start
    constantM1,  # end
    constant3,  # axes
    constant3  # step
]

node0 = gs.Node("Slice", "mySlice", inputs=inputList, outputs=[tensor1])

graph = gs.Graph(nodes=[node0], inputs=[tensor0], outputs=[tensor1])

graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "model-01.onnx")
