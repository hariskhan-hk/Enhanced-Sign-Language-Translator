# Enhanced Sign Language Translator
**Integrating Behavioral Signature for Improved Contextual Accuracy**

## Executive Summary
Individuals with hearing impairments face significant communication barriers, especially in real-time settings. This project presents an advanced sign-to-text translation system incorporating **behavioral signatures** for improved contextual accuracy and inclusivity. It is optimized for real-time performance on mobile and embedded devices.

## Problem Statement
Sign language users face limitations in educational, professional, and social environments due to the lack of robust, real-time sign-to-text systems. Variations in gesture styles further hinder the performance of conventional recognition models.

## Background
We developed a deep learning-based sign recognition system capable of:
- Handling diverse gesture styles
- Maintaining accuracy with imperfect or varied signs
- Running efficiently on low-resource devices

The system integrates behavioral signature analysis to enhance model adaptability and user inclusivity.

## Methodology

- **Data Collection**: Captured 38 classes of gestures from 24 signers using a DSLR camera.
- **Preprocessing**: Performed annotation and data augmentation to enrich the dataset.
- **Model Evaluation**:
  - Explored YOLOv8, YOLOv11, and YOLO NAS.
  - Compared models on accuracy, inference time, and computational efficiency.
- **Behavioral Signature Integration**: Incorporated signer behavior patterns to handle gesture variations more effectively.

## Results & Findings

- **YOLOv8** and **YOLO NAS** demonstrated strong performance in early tests.
- Models are:
  - **Highly accurate**
  - **Efficient for real-time applications**
  - **Robust against inconsistent gestures**

## Conclusion
This project provides a scalable, inclusive, and efficient sign language translation system that can transform communication for the deaf and mute community. The behavioral signature approach ensures real-world usability and increased accessibility.

## Acknowledgments
Special thanks to:
- **Ida Rieu Welfare Association** – for their support and collaboration

## Dedication
This work is dedicated to Ida Rieu and the **deaf and mute community in Pakistan**

## Authors
- [Muhammad Haris Khan](https://github.com/hariskhan-hk)
- [Shoaib Ul Haq](https://github.com/shoaibulhaque)
- [Zargul](https://github.com/zar373)
- [Aleezah Aatif](https://github.com/Aleezahshaikh)
