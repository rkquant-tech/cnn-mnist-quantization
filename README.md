# CNN Model Quantization & Optimization for Edge AI (MNIST)

This repository demonstrates high-efficiency Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT) techniques applied to a Convolutional Neural Network (CNN) trained on the MNIST dataset. The project focuses on extreme model size reduction tailored for deployment on resource-constrained peripheral hardware (Edge AI devices / microcontrollers like ESP32).

## 🚀 Key Achievements
* **11.4x Size Reduction** using INT8 Post-Training Quantization (PTQ).
* **11.5x Size Reduction** using Quantization-Aware Training (QAT).
* **High Accuracy Retention**: Over 98.7% accuracy for PTQ and 98.9% for QAT maintained post-compression.

---

## 📊 Performance & Compression Metrics

| Model Variant | Model Size (KB) | Compression Ratio | Accuracy (%) | Target Deployment |
| :--- | :---: | :---: | :---: | :--- |
| **Baseline FP32 Model** | *Original* | 1.0x (Baseline) | **98.93%** | High-performance PC / Server |
| **INT8 PTQ Model** | *Compressed* | **11.4x smaller** | **98.7%** | Standard Edge Devices |
| **INT8 QAT Model** | *Compressed* | **11.5x smaller** | **98.9%** | Low-power Microcontrollers (ESP32) |

---

## 🛠️ Tech Stack & Methods
* **Frameworks:** TensorFlow / Keras, TensorFlow Lite (TFLite Converter)
* **Optimization Techniques:** 
  * Post-Training Integer Quantization (PTQ)
  * Quantization-Aware Training (QAT)
* **Target Hardware Architecture:** Harvard Architecture Microcontrollers (such as ESP32), low-power Edge AI platforms.

---

## 📂 Project Structure & Usage

*   `quantization_stage1.py`: Data loading and initial MobileNetV2 exploration.
*   `quantization_stage2.py`: CNN training on MNIST and **PTQ** (Post-Training Quantization) pipeline.
*   `quantization_stage3.py`: Fine-tuning with **QAT** (Quantization-Aware Training) and INT8 conversion.
*   `test_int8.py`: Benchmarking script to evaluate the accuracy of the quantized `.tflite` model.
*   `mnist_model.h5`: The baseline FP32 trained Keras model.
*   `mnist_model_int8.tflite`: The optimized PTQ model (128 KB).
*   `mnist_model_qat_int8.tflite`: The optimized QAT model (126 KB).

### 🚀 How to Run
1. Install dependencies: `pip install tensorflow tensorflow-model-optimization numpy`
2. Train and quantize the model (PTQ): `python quantization_stage2.py`
3. Apply QAT and compare results: `python quantization_stage3.py`

---
*Developed for professional Edge AI model optimization and embedded deployment pipelines.*
