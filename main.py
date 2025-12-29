import pandas as pd
from apyori import apriori
import matplotlib.pyplot as plt

#导入数据
data = pd.read_csv(r"E:\电商RFM模型\goods\GoodsOrder.csv")

'''数据预处理'''

#处理缺失值
data.drop(index=data[data['Goods'].isnull()].index, inplace=True)

#处理异常值
data.drop(index=data[data['id']<0].index, inplace=True)


'''计算关联规则'''

#计算不重复id的总个数
length = data['id'].value_counts().count()

#创建空列表data_list，用于存放不同id的所有商品
data_list = []

#循环遍历每个id
for i in range(1,length):
    #找到第i列的商品
    item = data[data['id']==i]
    #转为商品列表并存储到新列表中
    item_list = list(item['Goods'])
    #新列表，每一个元素也是列表，对应一个id的商品
    data_list.append(item_list)

#计算关联规则，设定最小支持度为0.01，最小置信度为0.5
rules = apriori(data_list, min_support=0.01, min_confidence=0.5)

# 创建空列表relationship_lift，用于存储关联规则和提升度数据
relationship_list = []

#遍历关联规则
for rule in rules:
    #计算支持度，保留3位小数
    support = round(rule.support,3)
    #遍历存放关联规则的ordered_statistics对象
    for i in rule.ordered_statistics:

        #筛选提升度大于2的关联规则前后件
        if i.lift > 2:
            head_set = list(i.items_base)
            head_tail = list(i.items_add)

            #将前件、后件用str()转化为字符串，拼接成关联规则的形式,并且重新赋值
            related_category = str(head_set) + '->' + str(head_tail)

            #计算置信度
            confidence = round(i.confidence, 3)
            #计算提升度,保留三位小数
            lift = round(i.lift, 3)

    relationship_list.append([related_category, support, confidence, lift])

#将列表转为DataFrame类型，方便观察
df = pd.DataFrame(relationship_list, columns=['关联规则', '支持度', '置信度', '提升度'])

print(df)

'''提升度排序可视化'''

#设置字体为黑体
plt.rcParams['font.sans-serif'] = 'SimHei'
#防止负号显示方框
plt.rcParams['axes.unicode_minus'] = False

#对"提升度"列进行降序排序
df_sort = df.sort_values(by='提升度', ascending=False)

#x轴标签：提升度
y_values = df_sort['提升度']

#x轴位置
x_pos = range(len(df_sort))
#y轴：关联规则
x_labels = df_sort['关联规则']

#绘制柱状图
bars = plt.bar(
    x=x_pos, 
    height=y_values,
    width=0.6   #设置宽度
)

# 设置x轴：替换为关联规则标签，旋转45°避免遮挡,选择右对齐， 字体大小为9
plt.xticks(x_pos, x_labels, rotation=45, ha='right', fontsize=9)

# 设置坐标轴标签和标题
plt.xlabel('商品关联规则' ,fontsize=14, fontweight='bold')
plt.ylabel('提升度' ,fontsize=14, fontweight='bold')
plt.title('商品关联规则提升度排序', fontsize=14, fontweight='bold')

# 给每个柱状图添加数值标签（直观看到提升度具体值）
for i, bar in enumerate(bars):  #获取bars的行索引,i表示索引，bar表示柱子本身

    # 获取 “要贴的数值”：柱子的高度（对应提升度）
    height = bar.get_height() 
    #获取 “标签的水平位置”：柱子的正中间
    bar_x = bar.get_x() + bar.get_width() / 2
    #获取 “标签的垂直位置”：柱子顶部上方一点(留0.05的空隙)
    bar_y = height + 0.05
    #把数值贴到对应位置
    plt.text(bar_x, bar_y, f'{height:.3f}', ha='center', va='bottom') #ha为水平对齐方式，va为垂直对齐方式

plt.tight_layout()
plt.show()