#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LocateAnything-3B 简化测试脚本
用于验证模型是否能正常加载和运行
"""

import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModel

MODEL_PATH = "/home/yuan/.cache/modelscope/hub/models/nv-community/LocateAnything-3B"

def main():
    print("=" * 60)
    print("    LocateAnything-3B 模型测试")
    print("=" * 60)
    
    # 环境检查
    print("\n=== 环境检查 ===")
    print(f"Python版本: {torch.__version__}")
    print(f"PyTorch版本: {torch.__version__}")
    
    # 检查设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 加载模型
    print("\n=== 加载模型 ===")
    try:
        print("加载 Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        print("✓ Tokenizer 加载成功")
        
        print("加载 Processor...")
        processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
        print("✓ Processor 加载成功")
        
        print("加载 Model...")
        model = AutoModel.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
            device_map=device,
        ).eval()
        print("✓ Model 加载成功")
        
        # 打印模型信息
        print(f"\n=== 模型信息 ===")
        print(f"模型类型: {type(model).__name__}")
        print(f"模型设备: {next(model.parameters()).device}")
        
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"总参数量: {total_params / 10**9:.2f} B")
        
        print("\n" + "=" * 60)
        print("    测试完成！模型加载成功！")
        print("=" * 60)
        print("\n提示: 在有GPU的环境中运行完整测试")
        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()