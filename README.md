# Klasifikasi Makanan Tradisional Indonesia dengan Semi-Supervised Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Deskripsi

Proyek ini mengimplementasikan klasifikasi makanan tradisional Indonesia dalam perlombaan Action Unesa 2025 menggunakan teknik **Semi-Supervised Learning** dengan kombinasi arsitektur **Convolutional Neural Network (CNN)** dan **Vision Transformer (ViT)**. Penelitian ini memanfaatkan metode **pseudo-labeling** untuk meningkatkan akurasi klasifikasi pada dataset makanan tradisional Indonesia.

## Tujuan

- Mengklasifikasikan makanan tradisional Indonesia dengan akurasi tinggi
- Mengevaluasi efektivitas pseudo-labeling dalam identifikasi makanan tradisional
- Menentukan arsitektur pretrained yang optimal antara CNN (ResNet) dan ViT (DINO)
- Mengembangkan sistem klasifikasi kuliner nasional yang robust

## Metodologi

### Arsitektur Model
- **CNN**: ResNet (pretrained)
- **Vision Transformer**: DINO v2 (pretrained)

### Teknik Semi-Supervised Learning
1. **Pseudo-labeling** dengan confidence threshold yang ketat
2. **Fine-tuning** iteratif menggunakan pretrained models
3. **Layer-wise Learning Rate Decay (LLRD)** untuk optimasi DINOv2

### Pipeline Training
1. Ekstraksi fitur menggunakan DINOv2
2. Linear Probing
3. Fine-Tuning dengan LLRD

## Hasil Eksperimen

### Model Terbaik: DINO v2
- **Training Accuracy**: 100.00%
- **Validation Accuracy**: 93.45%
- **Test Accuracy**: 93.67%


## Fitur Utama

- **Pseudo-labeling** dengan confidence threshold adaptif
- **Multi-architecture support** (CNN & Vision Transformer)
- **Layer-wise Learning Rate Decay** untuk ViT
- **Iterative fine-tuning** untuk optimasi performa
- **Comprehensive evaluation** dengan multiple metrics

