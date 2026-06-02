#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LocateAnything-3B 汽车检测测试
用于检测图片中的汽车
"""

import os
import torch
import requests
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor

# 配置
MODEL_PATH = "/home/yuan/.cache/modelscope/hub/models/nv-community/LocateAnything-3B"
TEST_IMG_PATH = "/home/yuan/文档/git_project/LocateAnything-3B/cars.jpg"

def download_test_image():
    """下载测试图片（汽车图片）"""
    url = "https://images.unsplash.com/photo-1541443131-c3254ac7444e?w=800&q=80"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(TEST_IMG_PATH, 'wb') as f:
            f.write(response.content)
        print(f"✓ 测试图片已下载到 {TEST_IMG_PATH}")
        return True
    except Exception as e:
        print(f"✗ 下载图片失败: {e}")
        return False

class LocateAnythingWorker:
    def __init__(self, model_path: str, device: str = "cuda", dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            fix_mistral_regex=True
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
            low_cpu_mem_usage=True,
        ).to(device)

        self.model.eval()

    def predict(
        self,
        image: Image.Image,
        question: str,
        generation_mode: str = "hybrid",
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        verbose: bool = False,
    ) -> dict:

        max_size = 1024
        w, h = image.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ]
            }
        ]

        text = self.processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=images, videos=videos, return_tensors="pt"
        ).to(self.device)

        pixel_values = inputs["pixel_values"].to(self.dtype)

        with torch.no_grad():
            response = self.model.generate(
                pixel_values=pixel_values,
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_grid_hws=inputs.get("image_grid_hws", None),
                tokenizer=self.tokenizer,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                generation_mode=generation_mode,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                verbose=verbose,
            )

        result = {"answer": response[0] if isinstance(response, tuple) else response}
        return result

    def detect(self, image: Image.Image, categories: list[str], **kwargs) -> dict:
        cats = "</c>".join(categories)
        prompt = f"Locate all the instances that matches the description: {cats}."
        return self.predict(image, prompt, **kwargs)

    @staticmethod
    def parse_boxes(answer: str, w: int, h: int):
        import re
        boxes = []
        for m in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer):
            x1, y1, x2, y2 = map(int, m.groups())
            boxes.append({
                "x1": x1 / 1000 * w, "y1": y1 / 1000 * h,
                "x2": x2 / 1000 * w, "y2": y2 / 1000 * h
            })
        return boxes


def main():
    print("=" * 60)
    print("    LocateAnything-3B 汽车检测测试")
    print("=" * 60)
    
    # 检查并下载测试图片
    if not os.path.exists(TEST_IMG_PATH):
        print(f"\n测试图片 {TEST_IMG_PATH} 不存在，正在下载...")
        if not download_test_image():
            print("无法下载测试图片，请手动放置 cars.jpg 到项目目录")
            return
    
    # 加载图片
    img = Image.open(TEST_IMG_PATH).convert("RGB")
    print(f"\n图片尺寸: {img.size}")
    
    # 加载模型
    print("\n=== 加载模型 ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    worker = LocateAnythingWorker(MODEL_PATH, device=device, dtype=dtype)
    print("✅ 模型加载成功")
    
    # 测试汽车检测
    print("\n=== 检测汽车 ===")
    res = worker.detect(img, ["car"])
    print(f"原始输出: {res['answer']}")
    
    # 解析检测结果
    boxes = worker.parse_boxes(res["answer"], img.width, img.height)
    print(f"\n检测到 {len(boxes)} 辆车:")
    for i, box in enumerate(boxes):
        print(f"  汽车 {i+1}: x1={box['x1']:.1f}, y1={box['y1']:.1f}, x2={box['x2']:.1f}, y2={box['y2']:.1f}")
    
    print("\n" + "=" * 60)
    print("    检测完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()