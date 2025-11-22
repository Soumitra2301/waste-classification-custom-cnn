# Binary Waste Classification: Custom CNN vs Transfer Learning

**Best Model**: Custom CNN — **92% Test Accuracy** (outperforms MobileNetV2, ResNet50V2, EfficientNetB3)

## Dataset
- Source: [Mendeley Data](https://data.mendeley.com/datasets/n3gtgm9jxj/2) (DOI: 10.17632/n3gtgm9jxj.2)
- Classes: Organic (O) vs Recyclable (R)
- Total: ~25,077 images

## Quick Start (Reproduce Results)

```bash
# 1. Clone repo
git clone [https://github.com/Soumitra2301/waste-classification-custom-cnn.git]
cd waste-classification-custom-cnn

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run inference demo
python inference.py
