import pandas as pd
from apyori import apriori
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

# ========== 第一部分：你的原Apriori代码 ==========
# 导入数据
data = pd.read_csv(r"E:\电商RFM模型\goods\GoodsOrder.csv")

# 处理缺失值
data.drop(index=data[data['Goods'].isnull()].index, inplace=True)

# 处理异常值
data.drop(index=data[data['id']<0].index, inplace=True)

# 计算关联规则
length = data['id'].value_counts().count()
data_list = []

for i in range(1, length):
    item = data[data['id']==i]
    item_list = list(item['Goods'])
    data_list.append(item_list)

# 计算关联规则
rules = apriori(data_list, min_support=0.01, min_confidence=0.5)

relationship_list = []
for rule in rules:
    support = round(rule.support, 3)
    for i in rule.ordered_statistics:
        if i.lift > 2:
            head_set = list(i.items_base)
            head_tail = list(i.items_add)
            related_category = str(head_set) + '->' + str(head_tail)
            confidence = round(i.confidence, 3)
            lift = round(i.lift, 3)
    relationship_list.append([related_category, support, confidence, lift])

df_rules = pd.DataFrame(relationship_list, columns=['关联规则', '支持度', '置信度', '提升度'])

print("=== Apriori关联规则结果 ===")
print(df_rules.head())

# ========== 第二部分：结合决策树 ==========
print("\n" + "="*60)
print("开始结合决策树进行用户购买预测")
print("="*60)

# 1. 选取最重要的5条关联规则作为特征
top_rules = df_rules.nlargest(5, '提升度')
print(f"\n选取的5条高提升度规则：")
for i, (_, rule) in enumerate(top_rules.iterrows()):
    print(f"  规则{i+1}: {rule['关联规则']} (提升度: {rule['提升度']:.2f})")

# 2. 为每个用户创建特征向量
print("\n创建机器学习特征...")
user_features = []
all_user_ids = data['id'].unique()

for user_id in all_user_ids:
    # 获取该用户购买的所有商品
    user_items = set(data[data['id'] == user_id]['Goods'])
    
    features = []
    # 对每条重要规则，检查用户是否满足前件
    for _, rule_row in top_rules.iterrows():
        rule_str = rule_row['关联规则']
        antecedent = eval(rule_str.split('->')[0])  # 获取前件商品列表
        
        # 特征：用户是否购买了前件商品（0/1）
        has_antecedent = 1 if set(antecedent).issubset(user_items) else 0
        features.append(has_antecedent)
    
    user_features.append([user_id] + features)

# 创建特征DataFrame
feature_columns = ['用户ID'] + [f'规则{i+1}_前件' for i in range(len(top_rules))]
features_df = pd.DataFrame(user_features, columns=feature_columns)

# 3. 创建目标变量：用户是否购买了高提升度的后件商品
print("创建目标变量...")
targets = []

for idx, user_id in enumerate(features_df['用户ID']):
    user_items = set(data[data['id'] == user_id]['Goods'])
    
    # 检查用户是否购买了任意一条高提升度规则的后件
    bought_recommended = 0
    for _, rule_row in top_rules.iterrows():
        rule_str = rule_row['关联规则']
        consequent = eval(rule_str.split('->')[1])  # 获取后件商品
        
        if set(consequent).issubset(user_items):
            bought_recommended = 1
            break
    
    targets.append(bought_recommended)

# 准备机器学习数据
X = features_df.drop('用户ID', axis=1)
y = np.array(targets)

print(f"\n数据集信息：")
print(f"  样本数: {X.shape[0]}个用户")
print(f"  特征数: {X.shape[1]}个关联规则特征")
print(f"  正样本(购买推荐商品): {sum(y)}人, 负样本: {len(y)-sum(y)}人")

# 4. 训练决策树模型
print("\n训练决策树模型...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

tree_model = DecisionTreeClassifier(
    max_depth=3,  # 限制深度便于解释
    min_samples_split=20,
    random_state=42
)

tree_model.fit(X_train, y_train)

# 5. 评估模型
from sklearn.metrics import accuracy_score, classification_report

y_pred = tree_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n模型评估结果：")
print(f"  测试集准确率: {accuracy:.3f}")
print(f"\n详细分类报告：")
print(classification_report(y_test, y_pred, target_names=['未购买', '购买']))

# 6. 可视化决策树
print("\n生成决策树可视化...")
plt.figure(figsize=(15, 8))
plot_tree(tree_model, 
          feature_names=[f'规则{i+1}' for i in range(X.shape[1])],
          class_names=['不购买', '购买'],
          filled=True,
          rounded=True,
          fontsize=10)
plt.title('关联规则购买预测决策树', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# 7. 特征重要性分析
print("\n" + "="*60)
print("特征重要性分析")
print("="*60)

feature_importance = pd.DataFrame({
    '特征': [f'规则{i+1}' for i in range(X.shape[1])],
    '重要性': tree_model.feature_importances_
}).sort_values('重要性', ascending=False)

print(feature_importance)

# 8. 业务解读
print("\n" + "="*60)
print("业务洞察与应用建议")
print("="*60)

for idx, row in feature_importance.iterrows():
    if row['重要性'] > 0:
        rule_idx = int(row['特征'].replace('规则', '')) - 1
        rule = top_rules.iloc[rule_idx]
        
        antecedent = rule['关联规则'].split('->')[0]
        consequent = rule['关联规则'].split('->')[1]
        
        print(f"\n📌 关键规则 {rule_idx+1}:")
        print(f"   关联关系: {antecedent} → {consequent}")
        print(f"   特征重要性: {row['重要性']:.3f}")
        print(f"   推荐策略: 对购买了{antecedent}的用户，重点推荐{consequent}")
        print(f"   预期效果: 提升度 {rule['提升度']:.1f}倍")

# 9. 预测示例
print("\n" + "="*60)
print("预测示例")
print("="*60)

print("示例用户特征向量（0=未购买前件，1=购买了前件）：")
sample_features = pd.DataFrame([X.iloc[0]], columns=X.columns)
print(sample_features)

sample_pred = tree_model.predict(sample_features)[0]
pred_proba = tree_model.predict_proba(sample_features)[0]

print(f"\n预测结果：")
print(f"  是否购买推荐商品: {'是' if sample_pred == 1 else '否'}")
print(f"  购买概率: {pred_proba[1]:.1%}")

# ========== 原可视化部分保持 ==========
print("\n" + "="*60)
print("关联规则提升度排序可视化")
print("="*60)

plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

df_sort = df_rules.sort_values(by='提升度', ascending=False).head(10)  # 只显示前10
y_values = df_sort['提升度']
x_pos = range(len(df_sort))
x_labels = df_sort['关联规则']

plt.figure(figsize=(12, 6))
bars = plt.bar(x=x_pos, height=y_values, width=0.6, color='skyblue')

plt.xticks(x_pos, x_labels, rotation=45, ha='right', fontsize=9)
plt.xlabel('商品关联规则', fontsize=14, fontweight='bold')
plt.ylabel('提升度', fontsize=14, fontweight='bold')
plt.title('TOP 10 商品关联规则提升度排序', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

for i, bar in enumerate(bars):
    height = bar.get_height()
    bar_x = bar.get_x() + bar.get_width() / 2
    bar_y = height + 0.05
    plt.text(bar_x, bar_y, f'{height:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

print("\n✅ 分析完成！Apriori + 决策树结合分析已完成。")
print(f"   发现有效关联规则: {len(df_rules)} 条")
print(f"   构建预测模型准确率: {accuracy:.3f}")
print(f"   识别关键业务规则: {len(feature_importance[feature_importance['重要性']>0])} 条")
