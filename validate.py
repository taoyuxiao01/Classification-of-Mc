from ultralytics import YOLO
import os
import glob
import shutil

# ================= 配置 =================
# 指向你的数据集 yaml
DATA_YAML = 'mcdonalds.yaml' 
# 验证集图片路径 (必须与 yaml 中的 val 路径一致)
VAL_IMAGES_DIR = 'dynamic_datasets/images/val'
# 想要测试的模型权重 (通常选最后一次训练的 output)
# 这里假设你训练了50轮，想看第50轮的
LAST_EPOCH_WEIGHT = 'runs/custom_loop/epoch_50/weights/last.pt' 
# =======================================

def main():
    if not os.path.exists(LAST_EPOCH_WEIGHT):
        print(f"找不到权重文件: {LAST_EPOCH_WEIGHT}")
        print("请修改代码中的 LAST_EPOCH_WEIGHT 变量指向正确的 .pt 文件")
        return

    print(f"正在加载模型: {LAST_EPOCH_WEIGHT} ...")
    model = YOLO(LAST_EPOCH_WEIGHT)

    # --- 1. 数值评估 (Metrics) ---
    print("\n====== 步骤 1: 计算验证集指标 (mAP) ======")
    # 这里的 split='val' 对应 yaml 文件里的 val: images/val
    metrics = model.val(data=DATA_YAML, split='val', plots=True)
    
    print("\n结果摘要:")
    # map50 是最直观的“识别对了没有”的指标
    print(f"mAP@0.5 (识别准确率): {metrics.box.map50:.4f}")
    print(f"mAP@0.5-0.95 (框的位置精准度): {metrics.box.map:.4f}")
    
    # --- 2. 视觉推理 (Visual Inference) ---
    print("\n====== 步骤 2: 生成可视化结果图片 ======")
    output_dir = 'inference_results'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # 获取验证集里的图片
    jpg_files = glob.glob(os.path.join(VAL_IMAGES_DIR, '*.jpg'))
    
    # 运行预测并保存图片
    # conf=0.25 表示置信度大于 0.25 的才画出来
    results = model.predict(source=jpg_files, conf=0.25, save=True, project=output_dir, name='vis')

    print(f"\n可视化图片已保存在: {output_dir}/vis/")
    print("请打开该文件夹查看模型在真实汉堡可乐上的表现！")

if __name__ == "__main__":
    main()