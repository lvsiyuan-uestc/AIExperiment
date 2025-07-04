
import math
from collections import Counter


# --------------------------
# 数据加载函数
# --------------------------
def load_data(filename):
    """
    从文件中加载数据，忽略文件头尾的标识行。
    文件中每行格式为：数值[空格或制表符]... 数值 标签
    """
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("traindata") or line.startswith("testdata") or line in ["[", "]"]:
                continue
            parts = line.split()
            try:
                features = [float(x) for x in parts[:-1]]
                label = int(float(parts[-1]))
            except:
                continue
            data.append((features, label))
    return data


# --------------------------
# 熵计算函数
# --------------------------
def calc_entropy(data):
    """
    计算数据集熵：data 为 (features, label) 的列表
    """
    if not data:
        return 0.0
    label_counts = Counter([label for (_, label) in data])
    total = len(data)
    entropy = 0.0
    for count in label_counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


# --------------------------
# 数据划分函数
# --------------------------
def split_data(data, feature_index, threshold):
    """
    根据指定特征和阈值划分数据：
      - 左子集：样本在 feature_index 上的值 <= threshold
      - 右子集：样本在 feature_index 上的值 > threshold
    """
    left = []
    right = []
    for sample in data:
        features, label = sample
        if features[feature_index] <= threshold:
            left.append(sample)
        else:
            right.append(sample)
    return left, right


# --------------------------
# 选择最佳分裂点函数
# --------------------------
def choose_best_split(data, criterion="info_gain"):
    """
    对每个特征，尝试所有可能的分裂阈值（连续属性），选择能获得最大分裂指标的(feature_index, threshold)：
      - criterion="info_gain" 时，使用信息增益
      - criterion="gain_ratio" 时，使用信息增益率
    返回： best_feature, best_threshold, best_gain
    """
    base_entropy = calc_entropy(data)
    best_gain = -1
    best_feature = None
    best_threshold = None
    n_features = len(data[0][0])

    for feature_index in range(n_features):
        # 按当前特征排序
        sorted_data = sorted(data, key=lambda x: x[0][feature_index])
        # 遍历相邻样本，候选分裂点取相邻数值的均值
        for i in range(1, len(sorted_data)):
            f1 = sorted_data[i - 1][0][feature_index]
            f2 = sorted_data[i][0][feature_index]
            # 如果两个值相等，则无候选意义
            if f1 == f2:
                continue
            threshold = (f1 + f2) / 2.0
            left, right = split_data(data, feature_index, threshold)
            if not left or not right:
                continue
            weighted_entropy = (len(left) / len(data)) * calc_entropy(left) + (len(right) / len(data)) * calc_entropy(
                right)
            gain = base_entropy - weighted_entropy

            if criterion == "gain_ratio":
                # 计算划分信息（SplitInfo）
                p_left = len(left) / len(data)
                p_right = len(right) / len(data)
                split_info = 0.0
                if p_left > 0:
                    split_info -= p_left * math.log2(p_left)
                if p_right > 0:
                    split_info -= p_right * math.log2(p_right)
                if split_info != 0:
                    gain_ratio = gain / split_info
                else:
                    gain_ratio = 0
                if gain_ratio > best_gain:
                    best_gain = gain_ratio
                    best_feature = feature_index
                    best_threshold = threshold
            else:  # 默认 "info_gain"
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold
    return best_feature, best_threshold, best_gain


# --------------------------
# 求多数类标签
# --------------------------
def majority_label(data):
    labels = [label for (_, label) in data]
    return Counter(labels).most_common(1)[0][0]


# --------------------------
# 决策树节点定义
# --------------------------
class TreeNode:
    def __init__(self):
        self.is_leaf = False
        self.predicted_label = None

        self.feature_index = None  # 当前分裂特征
        self.threshold = None  # 分裂阈值（连续属性）
        self.left = None  # 树左子树（<= threshold）
        self.right = None  # 树右子树（> threshold)


# --------------------------
# 递归构建决策树
# --------------------------
def build_tree(data, depth=0, criterion="info_gain", max_depth=10):
    node = TreeNode()
    # 终止条件：纯节点或达到最大树深度
    if calc_entropy(data) < 1e-6 or depth >= max_depth:
        node.is_leaf = True
        node.predicted_label = majority_label(data)
        return node

    best_feature, best_threshold, best_gain = choose_best_split(data, criterion)
    if best_feature is None:
        node.is_leaf = True
        node.predicted_label = majority_label(data)
        return node

    node.feature_index = best_feature
    node.threshold = best_threshold
    left, right = split_data(data, best_feature, best_threshold)
    node.left = build_tree(left, depth + 1, criterion, max_depth)
    node.right = build_tree(right, depth + 1, criterion, max_depth)
    return node


# --------------------------
# 树的预测函数
# --------------------------
def classify(node, sample):
    if node.is_leaf:
        return node.predicted_label
    if sample[node.feature_index] <= node.threshold:
        return classify(node.left, sample)
    else:
        return classify(node.right, sample)


# --------------------------
# 打印决策树结构
# --------------------------
def print_tree(node, depth=0):
    indent = "  " * depth
    if node.is_leaf:
        print(f"{indent}Predict: {node.predicted_label}")
    else:
        print(f"{indent}Feature[{node.feature_index}] <= {node.threshold:.2f}?")
        print_tree(node.left, depth + 1)
        print(f"{indent}else:")
        print_tree(node.right, depth + 1)


# --------------------------
# 主函数
# --------------------------
def main():
    train_data = load_data("traindata.txt")
    test_data = load_data("testdata.txt")

    # 选择指标 "info_gain" 或 "gain_ratio"
    criterion = "info_gain"  # 默认使用信息增益 (ID3)
    # criterion = "gain_ratio"   # 使用信息增益率 (C4.5)

    max_depth = 10  # 可调节最大树深限制（剪枝效果）

    print("Building decision tree using {}...".format(criterion))
    tree = build_tree(train_data, depth=0, criterion=criterion, max_depth=max_depth)

    print("\n=== Decision Tree ===")
    print_tree(tree) # 可以选择是否打印整棵树结构

    # 在测试集上计算分类准确率并输出错误样本
    correct = 0
    print("\n=== Testing on Test Set ===")
    for features, label in test_data:
        pred = classify(tree, features)
        if pred == label:
            correct += 1
        else:
            # 输出错误分类的样本信息，格式类似图片所示
            print("错误数据:")
            print(f"种类: {label}")
            print(f"数据: {features}")
            print(f"误分类为: {pred}")
            print("-" * 20)  # 分隔线，可选

    accuracy = correct / len(test_data)
    print("\n准确率: {:.1f}%".format(accuracy * 100))  # 保持和图片类似的精度


if __name__ == "__main__":
    main()
input("按回车键退出...")