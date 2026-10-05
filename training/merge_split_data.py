import os
import sys
import shutil
import random
from collections import defaultdict
from tqdm import tqdm

# ════════════════════════════════════════════════════════════
# ⚙️ 划分配置 — 修改这里即可调整参数
# ════════════════════════════════════════════════════════════
CONFIG = {
    # 数据集根目录列表（按顺序排列，遇到冲突时，文件数相同的情况保留下标靠前的）
    "source_roots": [
        "./dataset_total/dataset1",
        "./dataset_total/dataset2",
        "./dataset_total/dataset3",
        "./dataset4", 
        "./dataset5",
    ],
    
    # 输出目录（划分后的 train/ 和 test/ 会放在这里）
    "target_root": "./dataset",
    
    # 训练集比例（85%训练，15%验证/测试）
    "train_ratio": 0.85,
    
    # 随机种子
    "seed": 42,
    
    # 是否复制文件（True=复制，False=移动，建议先复制测试）
    "copy_files": True,
    
    # 支持的图片格式
    "image_extensions": (".jpg", ".jpeg", ".png", ".bmp", ".gif"),

    # 新增限制：每个类别（书法家-字体）最多取多少张图片
    "max_images_per_class": 200,
    
    # 书法家ID映射（为了兼容后续的训练，统一用拼音缩写）
    "author_map": {
        "wxz":  "王羲之", "yzq":  "颜真卿", "lgq":  "柳公权", "sgt":  "孙过庭",
        "smh":  "沙孟海", "mf":   "米芾",   "htj":  "黄庭坚", "oyx":  "欧阳询",
        "zmf":  "赵孟頫", "csl":  "褚遂良", "wzm":  "文徵明", "yyr":  "于右任",
        "hy":   "弘一",   "bdsr": "八大山人","shz":  "宋徽宗", "mzd":  "毛泽东",
        "lx":   "鲁迅",   "cx":   "蔡襄",   "dqc":  "董其昌", "ly":   "刘墉",
        "wd":   "王铎",   "wcs":  "吴昌硕", "ybs":  "伊秉绶", "zx":   "朱熹",
        "cdf":  "陈道复", "fs":   "傅山",   "jn":   "金农",   "qg":   "启功",
        "ss":   "苏轼",   "xys":  "鲜于枢", "zrt":  "张瑞图", "zzq":  "赵之谦",
        "zbq":  "郑板桥", "zym":  "祝允明", "hsj":  "何绍基", "klnn": "康里巎巎",
        "sz":   "沈周",   "ty":   "唐寅",   "wc":   "王宠",   "xw":   "徐渭",
        "nyl":  "倪元璐", "wfa":  "王福庵", "cqw":  "成亲王", "lyb":  "李阳冰",
        "txz":  "唐玄宗", "dsr":  "邓石如", "ysn":  "虞世南", "sgz":  "宋高宗",
        "hs":   "怀素",   "lsz":  "林散之",
        # 冲突名字处理
        "wxz2": "王献之", "zxu":  "张旭",  "zyao": "钟繇", "zy": "智永",
    },

    # 别名映射（用于解决异体字、不同写法的问题，统一映射到标准名）
    "author_alias": {
        "文征明": "文徵明",
        "赵孟俯": "赵孟頫",
    }
}

def get_folder_size(folder_path):
    """计算文件夹内支持格式的图片总数量"""
    count = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(CONFIG["image_extensions"]):
                count += 1
    return count

def main():
    random.seed(CONFIG["seed"])
    target_root = os.path.abspath(CONFIG["target_root"])
    
    # 安全检查：清理目标目录
    if os.path.exists(target_root) and len(os.listdir(target_root)) > 0:
        print(f"\n[警告] 目标目录 {target_root} 已存在且不为空！")
        response = input("继续运行将删除该目录下的所有文件，是否继续？(y/N): ")
        if response.lower() != 'y':
            sys.exit(0)
        print(f"正在清空目标目录 {target_root}...")
        shutil.rmtree(target_root)

    name_to_id = {v: k for k, v in CONFIG["author_map"].items()}

    # ════════════════════════════════════════════════════════════
    # 第一步：扫描数据集，收集所有 "书法家-字体" 文件夹
    # ════════════════════════════════════════════════════════════
    print(f"开始扫描 {len(CONFIG['source_roots'])} 个数据集...")
    author_style_folders = defaultdict(list)
    
    for ds_idx, source_root in enumerate(CONFIG["source_roots"]):
        source_root = os.path.abspath(source_root)
        if not os.path.exists(source_root):
            print(f"[警告] 数据集路径不存在，跳过: {source_root}")
            continue
            
        print(f"\n正在扫描数据集 {ds_idx + 1}: {source_root}")
        
        for root, dirs, _ in os.walk(source_root):
            dir_name = os.path.basename(root)
            
            # 检查是否是 "书法家-字体" 格式的底层文件夹
            if "-" in dir_name:
                parts = dir_name.split("-")
                if len(parts) >= 2:
                    author_name = parts[0].strip()
                    style_name = parts[1].strip().replace("书", "").strip()
                    
                    # 处理别名（异体字归一化）
                    if author_name in CONFIG["author_alias"]:
                        author_name = CONFIG["author_alias"][author_name]
                    
                    if author_name in name_to_id:
                        author_id = name_to_id[author_name]
                        folder_path = root
                        file_count = get_folder_size(folder_path)
                        
                        if file_count > 0:
                            author_style_folders[(author_id, style_name)].append(
                                (folder_path, file_count, ds_idx)
                            )
                        
                        # 关键修复：找到了文件夹，立刻清空 dirs，不再往内部继续扫描
                        dirs[:] = [] 
                    else:
                        print(f"  [警告] 未识别的书法家: {author_name}，跳过文件夹: {root}")

    # ════════════════════════════════════════════════════════════
    # 第二步：冲突解决 —— 比较文件夹大小，保留最大的
    # ════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("开始进行文件夹冲突检测与合并（保留最大文件夹）...")
    print("="*60)
    
    final_folders = {}
    
    for (author_id, style), folders in author_style_folders.items():
        # 按文件数量降序排序
        folders.sort(key=lambda x: x[1], reverse=True)
        
        best_folder = folders[0]
        final_folders[(author_id, style)] = best_folder[0]
        
        if len(folders) > 1:
            author_name = CONFIG["author_map"][author_id]
            print(f"\n[冲突] {author_name}-{style}:")
            for f_path, f_count, ds_idx in folders:
                status = " 保留" if f_path == best_folder[0] else "❌ 丢弃"
                print(f"  {status} | 数据集{ds_idx+1} | 数量: {f_count:4d} | 路径: {f_path}")
            print(f"  -> 最终保留: {best_folder[0]} ({best_folder[1]}张)")

    # ════════════════════════════════════════════════════════════
    # 第三步：按书法家归类，并进行数量限制 (最多200张)
    # ════════════════════════════════════════════════════════════
    print(f"\n开始提取图片路径，并对每个 '书法家-字体' 类别限制最多 {CONFIG['max_images_per_class']} 张...")
    
    # 临时结构：收集所有图片，按 (author_id, style) 分组
    temp_style_images = defaultdict(list) # (author_id, style) -> [img_path]
    
    for (author_id, style), folder_path in final_folders.items():
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(CONFIG["image_extensions"]):
                    full_path = os.path.join(root, file)
                    temp_style_images[(author_id, style)].append(full_path)

    # 进行随机截断
    author_images = defaultdict(list)
    author_stats = defaultdict(lambda: defaultdict(int))
    
    for (author_id, style), paths in temp_style_images.items():
        random.shuffle(paths)
        limit = CONFIG["max_images_per_class"]
        selected_paths = paths[:limit]
        
        for p in selected_paths:
            author_images[author_id].append((p, style))
            author_stats[author_id][style] += 1

    # 打印截断后的统计信息
    print("\n" + "="*60)
    print("数量限制完成！各书法家-字体最终数量统计：")
    print("="*60)
    
    total_images = 0
    for author_id in sorted(CONFIG["author_map"].keys()):
        if author_id not in author_stats:
            continue
        author_name = CONFIG["author_map"][author_id]
        styles = author_stats[author_id]
        style_str = " | ".join([f"{style}: {count}张" for style, count in styles.items()])
        author_total = sum(styles.values())
        print(f"  {author_name} ({author_id}): {style_str} -> 总计: {author_total}张")
        total_images += author_total
    print(f"\n限制后总计: {total_images} 张图片")
    print("="*60 + "\n")

    # ════════════════════════════════════════════════════════════
    # 第四步：按书法家划分训练集/测试集 (85% : 15%)
    # ════════════════════════════════════════════════════════════
    train_dir = os.path.join(target_root, "train")
    test_dir = os.path.join(target_root, "test")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    for author_id in CONFIG["author_map"].keys():
        os.makedirs(os.path.join(train_dir, author_id), exist_ok=True)
        os.makedirs(os.path.join(test_dir, author_id), exist_ok=True)

    print("开始复制并划分数据集...")
    train_count = defaultdict(int)
    test_count = defaultdict(int)
    
    for author_id, images in tqdm(author_images.items(), desc="按书法家划分"):
        random.shuffle(images)
        split_idx = int(len(images) * CONFIG["train_ratio"])
        
        # 复制训练集
        for img_path, style in images[:split_idx]:
            target_path = os.path.join(train_dir, author_id, os.path.basename(img_path))
            counter = 1
            while os.path.exists(target_path):
                name, ext = os.path.splitext(os.path.basename(img_path))
                target_path = os.path.join(train_dir, author_id, f"{name}_{counter}{ext}")
                counter += 1
            if CONFIG["copy_files"]:
                shutil.copy2(img_path, target_path)
            else:
                shutil.move(img_path, target_path)
            train_count[author_id] += 1
            
        # 复制测试集
        for img_path, style in images[split_idx:]:
            target_path = os.path.join(test_dir, author_id, os.path.basename(img_path))
            counter = 1
            while os.path.exists(target_path):
                name, ext = os.path.splitext(os.path.basename(img_path))
                target_path = os.path.join(test_dir, author_id, f"{name}_{counter}{ext}")
                counter += 1
            if CONFIG["copy_files"]:
                shutil.copy2(img_path, target_path)
            else:
                shutil.move(img_path, target_path)
            test_count[author_id] += 1

    # ════════════════════════════════════════════════════════════
    # 第五步：打印最终划分结果
    # ════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("数据集划分完成！")
    print("="*60)
    print(f"训练集目录: {train_dir}")
    print(f"测试集目录: {test_dir}")
    print(f"训练集比例: {CONFIG['train_ratio']*100:.1f}%")
    print(f"测试集比例: {(1-CONFIG['train_ratio'])*100:.1f}%")
    print("\n各书法家最终划分结果：")
    print("-"*60)
    print(f"{'书法家':<10} {'训练集':<8} {'测试集':<8} {'总计':<8}")
    print("-"*60)
    
    total_train, total_test = 0, 0
    for author_id in sorted(CONFIG["author_map"].keys()):
        author_name = CONFIG["author_map"][author_id]
        tc = train_count.get(author_id, 0)
        vc = test_count.get(author_id, 0)
        total = tc + vc
        if total > 0:
            print(f"{author_name:<10} {tc:<8} {vc:<8} {total:<8}")
        total_train += tc
        total_test += vc
    
    print("-"*60)
    print(f"{'总计':<10} {total_train:<8} {total_test:<8} {total_train+total_test:<8}")
    print("="*60)
    
    if total_train + total_test != total_images:
        print(f"\n[警告] 文件数量不匹配！限制后统计到 {total_images} 张，实际复制了 {total_train+total_test} 张")
    else:
        print(f"\n 文件数量验证通过：{total_images} 张全部成功复制")

if __name__ == "__main__":
    main()