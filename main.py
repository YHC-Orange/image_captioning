import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, GPT2Tokenizer
from PIL import Image
import matplotlib.pyplot as plt

# 加载模型和相关工具
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = GPT2Tokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# 图像预处理函数
def preprocess(image_path):
    image = Image.open(image_path).convert("RGB")
    pixel_values = feature_extractor(images=[image], return_tensors="pt", padding=True).pixel_values
    return pixel_values

# 生成图像描述
def generate_caption(image_path):
    pixel_values = preprocess(image_path)
    output_ids = model.generate(pixel_values, max_length=16, num_beams=4, repetition_penalty=2.0)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

# 加载并显示图像及其描述
image_path = "DJI.JPG"  # 替换为你的图像路径
caption = generate_caption(image_path)
image = Image.open(image_path)

plt.imshow(image)
plt.title(caption)
plt.axis('off')  # 隐藏坐标轴
plt.show()

print("Generated Caption:", caption)
