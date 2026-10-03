# Synthetic Manuscript Generator

An automated Python pipeline for generating synthetic historical manuscript folios with synchronized ground-truth annotations for OCR dataset creation.

## Overview

This project generates synthetic manuscript images using historical Indic script corpora and configurable visual variations.

The pipeline currently supports:

- Devanagari
- Modi
- Sharada

For every generated manuscript image, a corresponding Markdown (`.md`) annotation file is created containing the source text, script information, image dimensions, and text bounding boxes.

## Features

- Aged manuscript-style backgrounds
- Fine and large-scale paper texture
- Random aging and stains
- Edge darkening
- Fold and crease effects
- Multiple text blocks
- Random text selection from script-specific corpora
- Text wrapping within manuscript boundaries
- Position and alignment variation
- Ink color variation
- Slight text rotation
- Faded and blurred ink effects
- Smudging and ink imperfections
- Synchronized Markdown annotations
- Train, validation and test dataset splits
- Support for Devanagari, Modi and Sharada scripts

## Project Structure

```text
synthetic-manuscript-generator/
│
├── data/
│   ├── devanagari_md.md
│   ├── Modi_md.md
│   └── sharada_md.md
│
├── fonts/
│   ├── NotoSansDevanagari-Regular.ttf
│   ├── NotoSansModi-Regular.ttf
│   └── Sharada.ttf
│
├── output/
│   └── Generated dataset
│
├── generate.py
└── README.md
