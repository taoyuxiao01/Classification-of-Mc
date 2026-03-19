import random
import os
import shutil
from ultralytics import YOLO
from pathlib import Path

#==================配置区域====================
#路径配置
BASE_DIR = os.getcwd()
REAL_DATA_DIR = os.path.join(BASE_DIR, 'datasets', 'real')
SYN_DATA_DIR = os.path.join(BASE_DIR, 'datasets', 'synthetic')
TARGET_DIR = os.path.join(BASE_DIR, 'dynamic_datasets')

#训练配置
TOTAL_EPOCHS = 50
IMG_SIZE = 640
BATCH_SIZE = 16
MODEL_NAME = 'yolo11n.pt'

#====================工具函数=======================
def setup_directories():
    for split in ['train', 'val']:
        for dtype in ['images', 'labels']:
            path = os.path.join(TARGET_DIR, dtype, split)
            os.makedirs(path, exist_ok=True)

def copy_files(file_list, source_dir, split_type):
    """将图片和对应的标签复制到目标目录"""
    for filename in file_list:
        #复制图片
        src_img = os.path.join(source_dir, 'images', filename)
        dst_img = os.path.join(TARGET_DIR, 'images', split_type, filename)
        shutil.copy(src_img, dst_img)

        #复制标签
        label_name = os.path.splitext(filename)[0] + '.txt'
        src_label = os.path.join(source_dir, 'labels', label_name)
        dst_label = os.path.join(TARGET_DIR, 'labels', split_type, label_name)

        if os.path.exists(src_label):
            shutil.copy(src_label, dst_label)

def get_all_images(directory):
    """获取目录下所有图片文件"""
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    return [f for f in os.listdir(os.path.join(directory, 'images'))
            if os.path.splitext(f)[1].lower() in valid_exts]

#======================主逻辑=====================
def main():
    # 1. 准备工作
    print("正在初始化项目结构...")
    setup_directories()

    # 获取所有文件名
    real_images = get_all_images(REAL_DATA_DIR)
    syn_images = get_all_images(SYN_DATA_DIR)

    if len(real_images) < 50:
        print(f"警告：真实数据只有{len(real_images)}张，不足50张")
    
    #2. 锁定验证集
    # 随机打乱真实数据
    random.shuffle(real_images)

    # 切分：前十张做验证，后40张做训练
    val_files = real_images[:10]
    train_real_pool = real_images[10:]

    print(f"验证集已锁定：{len(val_files)}张真实图片")
    #将验证集一次性复制进去，之后不再变动
    copy_files(val_files, REAL_DATA_DIR, 'val')

    #3. 初始化模型
    model = YOLO(MODEL_NAME)

    #4. 开始循环训练
    # 手动控制每一轮的数据构成
    current_weights = MODEL_NAME
    PROJECT_PATH = os.path.join(os.getcwd(), 'runs', 'custom_loop')

    for epoch in range(1, TOTAL_EPOCHS+1):
        print(f"\n======== 开始第{epoch}/{TOTAL_EPOCHS}轮训练=========")

        # --- A. 动态构建训练集 ---
        # 1. 清空训练集目录 (images/train 和 labels/train)
        # 注意：这里我们通过重新创建目录来清空
        shutil.rmtree(os.path.join(TARGET_DIR, 'images', 'train'), ignore_errors=True)
        shutil.rmtree(os.path.join(TARGET_DIR, 'labels', 'train'), ignore_errors=True)
        os.makedirs(os.path.join(TARGET_DIR, 'images', 'train'), exist_ok=True)
        os.makedirs(os.path.join(TARGET_DIR, 'labels', 'train'), exist_ok=True)

        # 2. 选取数据
        # 真实数据
        current_real = train_real_pool
        # 虚拟数据
        current_syn = random.sample(syn_images, 40)

        # 3. 复制文件
        copy_files(current_real, REAL_DATA_DIR, 'train')
        copy_files(current_syn, SYN_DATA_DIR, 'train')

        print(f"当前数据集构成 -> 真实：{len(current_real)} + 虚拟：{len(current_syn)}")

        # ---B. 执行训练 ---
        # 关键点：epoch=1, 并且家在上一次的权重
        # project 和name 用于管理输出， 防止生成 runs/detect/train1...train100
        results = model.train(
            data='mcdonalds.yaml',
            epochs=5,               # 每次只练x轮
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            project=PROJECT_PATH,
            name=f'epoch_{epoch}',  # 或者保持固定名字并覆盖
            exist_ok=True,          # 允许覆盖
            verbose=False,          # 减少日志输出
            plots=False,            # 减少画图时间
            device=0
        )

        # --- C. 更新权重 ---
        # 训练结束后，YOLO会自动保存 best.pt 和 last.pt
        # 我们将模型更新为刚训练完的 last.pt， 以便进行下一轮即成
        ckpt_path = os.path.join(PROJECT_PATH, f'epoch_{epoch}', 'weights', 'last.pt')
        model = YOLO(ckpt_path)

    print("\n全部训练完成!")
    print(f"最终模型保存在：{ckpt_path}")

if __name__ == "__main__":
    main()