## WASM - running models for inference in your browser - overview

### Why do we try to do it?


Imagine that you have a decent model that could make using your application/website more interesting, but you:
- don't want to pay for hosting the model for inference and associated infrastructure
- make use of the end user hardware
- make use (if possible) of the end user GPU
- make it distributable to every architecture
- make the inference efficient (by using WASM)

WASM itself already partly answers this problem:
WebAssembly (abbreviated _Wasm_) is a binary instruction format for a stack-based virtual machine. Wasm is designed as a portable compilation target for programming languages, enabling deployment on the web for client and server applications.

***WebAssembly (Wasm) is generally faster than JavaScript for certain types of tasks**. WebAssembly is a binary instruction format that allows developers to run code at near-native speed in modern web browsers. JavaScript, on the other hand, is a high-level scripting language that is often used for web development.*
https://graffersid.com/webassembly-vs-javascript/

So, it would be nice to have the inference part of your code written in WASM, to make it more optimized.

### What is WebGPU and what are the alternatives?

https://tvm.apache.org/2020/05/14/compiling-machine-learning-to-webassembly-and-webgpu
The introduction of the GPU to accelerate deep learning workloads has increased the rate of progress dramatically. Given the growing requirement to deploy machine learning everywhere, the browser becomes a natural place to deploy intelligent applications.

While TensorFlow.js and ONNX.js are existing efforts to bring machine learning to the browser, there still exist gaps in performance between the web versions and native ones. One of the many reasons is the lack of standard and performant access to the GPU on the web. WebGL lacks important features such as compute shaders and generic storage buffers that are necessary for high performance deep learning.



Interesting implementation which is in line with HuggingFace transformers library is Transformers.js. 
*By default, Transformers.js uses hosted pretrained models and precompiled WASM binaries, which should work out-of-the-box.* https://huggingface.co/docs/transformers.js/custom_usage

Despite not taking advantage of available WebGPU I would recommend it for the sake of simplicity, impressive results and wide range of models supported. Check out some of their numerous examples, such as Whisper in the browser: https://huggingface.co/spaces/Xenova/whisper-web
or translation model based on distilled gpt2 architecture (only 85 MB!): https://xenova.github.io/transformers.js/
This small size of the model was obtained with quantization of the ONNX model.



*Computer shaders, often referred to as just "shaders," are programs designed to run on a computer's graphics processing unit (GPU). They are a crucial component of modern graphics rendering pipelines and play a significant role in generating visual effects in video games, simulations, and other graphics-intensive applications.*
WebGPU is the upcoming standard for next generation web graphics which has the possibility to dramatically change this situation. Like the latest generation graphics APIs such as Vulkan and Metal, WebGPU offers first-class compute shader support.
What to do with WebGPU for our models?

**Tensorflow.js and ONNX.js approach:**
- write shaders for primitive operators in Neural Networks and directly optimize their performance

**TVM Apache approach:**
- compilation-based, 
- TVM automatically ingests models from high-level frameworks such as TensorFlow, Keras, PyTorch, MXNet and ONNX and uses a machine learning driven approach to automatically generate low level code, in this case compute shaders in SPIR-V format. The generated code can then be packaged as a deployable module.
- we get one app for different infrastructures
Problems they faced:
- The runtime needs to call into system library calls (malloc, stderr) - it is done with WASI (WASI-like library generated with emscripten for the web)
- The wasm runtime needs to interact with the WebGPU driver (in javascript where the WebGPU API is the first-class citizen) -> *We solve the second problem by building a WebGPU runtime inside TVM’s JS runtime, and calling back to these functions from the WASM module when invoking GPU code. Using the [PackedFunc](https://tvm.apache.org/docs/dev/runtime.html#packedfunc) mechanism in TVM’s runtime system, we can directly expose high-level runtime primitives by passing JavaScript closures to the WASM interface. This approach keeps most of the runtime code in JavaScript, we could bring more JS code into the WASM runtime as WASI and WASM support matures.*


### Web GPU availability - TODO

-------------------------------------------------------------------------------------
### Interesting projects using WebGPU
Created with TVM approach:
- https://github.com/mlc-ai/web-stable-diffusion
  *This project brings stable diffusion models onto web browsers. **Everything runs inside the browser with no server support.*** 

  What problems I encountered:
  - you have to have a lot of memory, as it downloads models which are not lightweight (but at least they are cached and you get a progress update while they download).
  - downloading the model for the first time is slow
- https://github.com/mlc-ai/web-llm
 *WebLLM is a modular, customizable javascript package that directly brings language model chats directly onto web browsers with hardware acceleration. **Everything runs inside the browser with no server support and is accelerated with WebGPU.** We can bring a lot of fun opportunities to build AI assistants for everyone and enable privacy while enjoying GPU acceleration.*

  What problems I encountered:
  again, it is slow and downloading TinyLLama for example, takes 2 GB of your memory.
  I experimented with disabling GPU support - then the chat does not work at all. This undermines the idea that we can have one app to run on all hardwares.
Anyway, this project seems to have a lot of potential and well documented workflow on how to add your own model library. I wanted to add a small BERT model, to make it decently good for conversations or question  but at the same time keep it as small as possible.  Despite my efforts, I wasn't successful in this attempt. [link to .ipynb notebook]

- WONNX

