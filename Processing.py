import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

# 设置 Seaborn 样式
sns.set_style('whitegrid')

# 清除警告
import warnings
warnings.filterwarnings('ignore')

# 数据读取
data = pd.read_csv("D:/archive/Data_Entry_2017.csv")

# 移除年龄大于100的行
data = data[['Image Index', 'Finding Labels', 'Follow-up #', 'Patient ID', 'Patient Age', 'Patient Gender']]
data = data[data['Patient Age'] < 100]

# 计算每条记录的标签数量
data['Labels_Count'] = data['Finding Labels'].apply(lambda text: len(text.split('|')) if text != 'No Finding' else 0)

# 结果可视化
# 结果1：疾病标签的分布
label_counts = data['Finding Labels'].value_counts()[:15]
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.bar(np.arange(len(label_counts)) + 0.5, label_counts)
ax1.set_xticks(np.arange(len(label_counts)) + 0.5)
ax1.set_xticklabels(label_counts.index, rotation=45, ha='right')
ax1.set_title('Top 15 Disease Labels')
ax1.set_ylabel('Count')
plt.tight_layout()
plt.show()

# 结果2：按性别划分的年龄分布
sns.FacetGrid(data, hue='Patient Gender', height=5).map(sns.histplot, 'Patient Age').add_legend()
plt.title('Age Distribution by Gender')
plt.tight_layout()
plt.show()

# 结果3：按病理分布绘制患者年龄和性别分布
f, axarr = plt.subplots(7, 2, sharex=True, figsize=(15, 20))
pathology_list = ['Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Nodule', 'Pneumothorax', 'Atelectasis',
                  'Pleural_Thickening', 'Mass', 'Edema', 'Consolidation', 'Infiltration', 'Fibrosis', 'Pneumonia']
df = data[data['Finding Labels'] != 'No Finding']
x = np.arange(0, 100, 10)

for i, pathology in enumerate(pathology_list):
    row, col = divmod(i, 2)
    index = df[df['Finding Labels'].str.contains(pathology)].index
    g = sns.countplot(x='Patient Age', hue='Patient Gender', data=df.loc[index], ax=axarr[row, col])
    axarr[row, col].set_title(pathology)
    g.set_xlim(0, 90)
    g.set_xticks(x)
    g.set_xticklabels(x, rotation=45, ha='right')

f.subplots_adjust(hspace=0.5)
plt.tight_layout()
plt.show()

# 为每个病理创建单独的列
for pathology in pathology_list:
    data[pathology] = data['Finding Labels'].apply(lambda x: 1 if pathology in x else 0)

# 结果4：可视化每个病理的性别分布
plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(8, 1)
ax1 = plt.subplot(gs[:7, :])
ax2 = plt.subplot(gs[7, :])

data1 = pd.melt(data, id_vars=['Patient Gender'], value_vars=pathology_list, var_name='Category', value_name='Count')
data1 = data1[data1['Count'] > 0]
sns.countplot(y='Category', hue='Patient Gender', data=data1, ax=ax1, order=data1['Category'].value_counts().index)
ax1.set(ylabel="", xlabel="")
ax1.legend(fontsize=10)
ax1.set_title('X Ray partition (total number = {})'.format(len(data)))

# 显示没有病的分类
data['Nothing'] = data['Finding Labels'].apply(lambda x: 1 if 'No Finding' in x else 0)
data2 = pd.melt(data, id_vars=['Patient Gender'], value_vars=['Nothing'], var_name='Category', value_name='Count')
data2 = data2[data2['Count'] > 0]
sns.countplot(y='Category', hue='Patient Gender', data=data2, ax=ax2)
ax2.set(ylabel="", xlabel="Number of disease")
plt.subplots_adjust(hspace=0.5)
plt.tight_layout()
plt.show()