import os
import warnings
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob
from itertools import chain
from sklearn.model_selection import train_test_split
# 设置绘图风格
sns.set_style('whitegrid')

# 忽略警告
warnings.filterwarnings('ignore')

# 打印可用GPU和使用的模型
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))



# ======================================================================Part 1 command-line parameter======================================================================
# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='Choose model: densenet121, vgg16, mobilenetV3Large, or efficientnetB0')
parser.add_argument('--learning_rate', type=float, help='Learning rate for model training')
parser.add_argument('--batch_size', type=int, help='Batch size for model training')
parser.add_argument('--epochs', type=int, help='Number of epochs for model training')
parser.add_argument('--stop_patience', type=int, help='Early stopping patience. Number of epochs with no improvement before stopping training.')
parser.add_argument('--add_dense_layers', type=str, help="Whether to add extra dense layers. Use 'true' or 'false'. Default is 'false'.")
args, unknown = parser.parse_known_args()
if args.add_dense_layers.lower() not in ['true', 'false']:
    print("Invalid value for --add_dense_layers. Please use 'true' or 'false'.")
    exit(1)
args.add_dense_layers = args.add_dense_layers.lower() == 'true'

# 确保用户指定了所有必要的参数
missing_args = []
if not args.model:
    missing_args.append("--model")
if args.learning_rate is None:
    missing_args.append("--learning_rate")
if args.batch_size is None:
    missing_args.append("--batch_size")
if args.epochs is None:
    missing_args.append("--epochs")
if args.stop_patience is None:
    missing_args.append("--stop_patience")
if args.add_dense_layers is None:
    missing_args.append("--add_dense_layers")
if missing_args:
    parser.error(f"The following arguments are required: {', '.join(missing_args)}")

# 打印model epoch batch size learning rate
print(f"Using model: {args.model}")
print(f"Learning Rate: {args.learning_rate}")
print(f"Batch Size: {args.batch_size}")
print(f"Epochs: {args.epochs}")
print(f"Stop Patience: {args.stop_patience}")
print(f"Add Dense Layers: {'True' if args.add_dense_layers else 'False'}")



# ======================================================================Part 2: The data Preprocess======================================================================
# 读取 CSV 文件，确保路径正确
data = pd.read_csv('D:/Datasets/Data_Entry_2017.csv')

# 去除年龄大于100的异常数据
data = data[data['Patient Age'] < 100]

# 将年龄数据转换为整数
data['Patient Age'] = data['Patient Age'].map(lambda x: int(x))

# 获取图像路径，并存入字典
data_image_paths = {
    os.path.basename(x): x for x in glob(os.path.join('D:/Datasets', 'images*', '*', '*.png'))
}

# 将图像路径添加到数据框中
data['path'] = data['Image Index'].map(data_image_paths.get)

# 打印数据概况，随机查看3条样本数据
print('Scans found:', len(data_image_paths), ', Total Headers', data.shape[0])
print(data.sample(3))

# 提取所有独特的标签
all_labels = np.unique(list(chain(*data['Finding Labels'].map(lambda x: x.split('|')).tolist())))
all_labels = [x for x in all_labels if len(x) > 0]

# 打印移除No Finding之后的全部标签
print('All Labels ({}): {}'.format(len(all_labels), all_labels))

# 为每个病症标签创建独立的列
for c_label in all_labels:
    if len(c_label) > 1:  # 移除空标签
        data[c_label] = data['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)

# 随机查3条数据样本
print(data.sample(3))

# 可视化重采样前的数据不平衡情况
original_label_counts = data['Finding Labels'].value_counts()[:15]
# 可视化前15个标签组合的数据不平衡情况
fig, ax = plt.subplots(figsize=(15, 8))
bar_width = 0.6
bar_positions = np.arange(len(original_label_counts))
ax.bar(bar_positions, original_label_counts, width=bar_width, color='lightcoral', edgecolor='black')
ax.set_xticks(bar_positions)
ax.set_xticklabels(original_label_counts.index, rotation=45, ha='right', fontsize=12)
ax.set_xlabel('Finding Labels', fontsize=14)
ax.set_ylabel('Count', fontsize=14)
ax.set_title('Top 15 Label Distribution Before Resampling', fontsize=16)
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 加载测试集的图像索引
test_list_path = 'D:/Datasets/test_list.txt'  # 替换为实际路径
if not os.path.exists(test_list_path):
    raise FileNotFoundError(f"Test list file not found at {test_list_path}")
with open(test_list_path, 'r') as f:
    test_image_names = f.read().splitlines()

# 确保测试集从原数据集中移除，避免后续采样影响
train_data = data[~data['Image Index'].isin(test_image_names)]
print(f"Data size after removing test images: {len(train_data)}")

# 确保测试集提取正确且未受影响
test_df = data[data['Image Index'].isin(test_image_names)].dropna(subset=['path'])
print(f"Final test set size: {len(test_df)}")

# 确保测试集包含 'disease_vec' 列（test）
test_df['disease_vec'] = test_df[all_labels].values.tolist()

# 检查是否遗漏任何测试图片
missing_test_images = set(test_image_names) - set(test_df['Image Index'])
if len(missing_test_images) > 0:
    print(f"Missing test images: {len(missing_test_images)}")
    print("Sample missing test images:", list(missing_test_images)[:5])
else:
    print("All test images are included in the test set.")

# 设置权重使数据尽可能平衡
label_counts = train_data['Finding Labels'].value_counts()
max_count = label_counts.max()
def calculate_weights(finding_labels):
    count = label_counts[finding_labels]
    # 计算每个标签的权重，使得数据集更平衡
    weight = max_count / count if count > 0 else 0
    return weight
sample_weights = train_data['Finding Labels'].map(calculate_weights).values

# 归一化样本权重
sample_weights /= sample_weights.sum()

# 重采样训练数据，保持测试集独立
train_data = train_data.sample(50000, weights=sample_weights, random_state=42)
print(f"Resampled train data size: {len(train_data)}")

# 创建 'disease_vec' 列（train）
train_data['disease_vec'] = train_data[all_labels].values.tolist()

# 仅对重采样后的训练数据进行训练/验证分割
train_df, valid_df = train_test_split(train_data, test_size=0.10, random_state=2018)
print(f"Train size: {len(train_df)}, Validation size: {len(valid_df)}")

# 确保训练和验证集包含 'disease_vec'
train_df['disease_vec'] = train_df[all_labels].values.tolist()
valid_df['disease_vec'] = valid_df[all_labels].values.tolist()

# 打印最终数据分布
print(f"Final dataset sizes - Train: {len(train_df)}, Validation: {len(valid_df)}, Test: {len(test_df)}")

# 统计最常见的 15 个标签的组合出现次数，可视化采样后的标签分布
resampled_label_counts = train_data['Finding Labels'].value_counts()[:15]
fig, ax = plt.subplots(figsize=(15, 8))
bar_width = 0.6
bar_positions = np.arange(len(resampled_label_counts))
ax.bar(bar_positions, resampled_label_counts, width=bar_width, color='skyblue', edgecolor='black')
ax.set_xticks(bar_positions)
ax.set_xticklabels(resampled_label_counts.index, rotation=45, ha='right', fontsize=12)
ax.set_xlabel('Finding Labels', fontsize=14)
ax.set_ylabel('Count', fontsize=14)
ax.set_title('Resampled Train Data Label Distribution', fontsize=16)
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()



# ============================================================Part 3 Data Load and preprocessing with different models============================================================
from keras.applications.densenet import preprocess_input as densenet_preprocess_input
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from keras.applications.mobilenet_v3 import preprocess_input as mobilenet_v3_preprocess_input
from keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input
from keras.preprocessing.image import ImageDataGenerator
# 正确创建 'disease_vec' 列，确保它包含列表形式的标签
data['disease_vec'] = data[all_labels].values.tolist()

# 数据增强
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# 定义从 DataFrame 到带有模型特定预处理的 tf.data.Dataset 的转换函数
def dataframe_to_dataset(df, model_name, img_size=(224, 224)):
    paths = df['path'].values
    labels = np.array(df['disease_vec'].tolist(), dtype='float32')
    # 定义加载和预处理图像的函数
    def load_and_preprocess_image(path, label, model_name=model_name):
        # 加载并处理图像
        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(image, img_size)
        # 应用模型特定的预处理
        model_name = model_name
        if model_name == 'densenet121':
            image = densenet_preprocess_input(image)
        elif model_name == 'vgg16':
            image = vgg16_preprocess_input(image)
        elif model_name == 'mobilenetV3Large':
            image = mobilenet_v3_preprocess_input(image)
        elif model_name == 'efficientnetB0':
            image = efficientnet_preprocess_input(image)
        else:
            raise ValueError("Invalid model name. Choose 'densenet121', 'vgg16', 'mobilenetV3Large', or 'efficientnetB0'.")
        return image, label
    # 构建数据集
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(lambda x, y: load_and_preprocess_image(x, y), num_parallel_calls=tf.data.AUTOTUNE)
    return dataset



# ======================================================================Part 4 Models======================================================================
from keras.applications.densenet import DenseNet121
from keras.applications.vgg16 import VGG16
from keras.applications.mobilenet_v3 import MobileNetV3Large
from keras.applications.efficientnet import EfficientNetB0
from keras.layers import Input, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc as auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# 将数据分为训练、验证和测试集（change using offical test）
IMG_SIZE = (224, 224, 3)
img_in = Input(shape=IMG_SIZE)
if args.model == 'densenet121':
    base_model = DenseNet121(include_top=False, 
                             weights='imagenet', 
                             input_tensor=img_in, 
                             pooling='avg')
elif args.model == 'vgg16':
    base_model = VGG16(include_top=False, 
                              weights='imagenet', 
                              input_tensor=img_in, 
                              pooling='avg')
elif args.model == 'mobilenetV3Large':
    base_model = MobileNetV3Large(include_top=False, 
                                  weights='imagenet', 
                                  input_tensor=img_in, 
                                  pooling='avg')
elif args.model == 'efficientnetB0':
    base_model = EfficientNetB0(include_top=False, 
                                weights='imagenet', 
                                input_tensor=img_in, 
                                pooling='avg')
else:
    raise ValueError("Invalid model name. Choose 'densenet121', 'vgg16', 'mobilenetV3Large', or 'efficientnetB0'.")

# 添加额外的层VGG16 dense layers
x = base_model.output
if args.add_dense_layers:
    x = Flatten()(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)

# 最后的全连接层用于多标签分类任务
predictions = Dense(len(all_labels), activation="sigmoid", name="predictions")(x)
model = Model(inputs=img_in, outputs=predictions)

# 编译模型
optimizer = Adam(learning_rate=args.learning_rate)
model.compile(optimizer=optimizer, 
              loss='binary_crossentropy', 
              metrics=['binary_accuracy'])

# 创建训练、验证和测试集的数据集
batch_size = args.batch_size
train_ds = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=None,
    x_col='path',
    y_col=all_labels,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='raw'
)
valid_ds = dataframe_to_dataset(valid_df, model_name=args.model).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = dataframe_to_dataset(test_df, model_name=args.model).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

# 计算 steps_per_epoch 和 validation_steps
steps_per_epoch = np.ceil(len(train_df) / batch_size).astype(int)
validation_steps = np.ceil(len(valid_df) / batch_size).astype(int)
test_steps = np.ceil(len(test_df) / batch_size).astype(int)

# 设置模型检查点，保存最优模型
save_dir = 'D:/NIH-Chest-X-ray-Dataset/models'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
checkpoint_filepath = os.path.join('D:/NIH-Chest-X-ray-Dataset/models', f'best_model_{args.model}.h5')
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    mode='max',
    save_best_only=True, 
    save_weights_only=True
)

# 设置早停机制，避免过拟合
early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=args.stop_patience,
    restore_best_weights=True
)

# 训练模型
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=args.epochs,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[model_checkpoint_callback, early_stopping_callback]
)

# 可视化训练过程中的损失和准确率
fig, axes = plt.subplots(2, 1, figsize=(18, 12))
# 绘制训练和验证损失
axes[0].plot(history.history['loss'], label='Training Loss', color='b', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Validation Loss', color='r', linewidth=2)
axes[0].set_xlabel('Epochs', fontsize=14)
axes[0].set_ylabel('Loss', fontsize=14)
axes[0].set_title('Training and Validation Loss', fontsize=16)
axes[0].legend(fontsize=12)
axes[0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
# 绘制训练和验证准确率
axes[1].plot(history.history['binary_accuracy'], label='Training Accuracy', color='b', linewidth=2)
axes[1].plot(history.history['val_binary_accuracy'], label='Validation Accuracy', color='r', linewidth=2)
axes[1].set_xlabel('Epochs', fontsize=14)
axes[1].set_ylabel('Accuracy', fontsize=14)
axes[1].set_title('Training and Validation Accuracy', fontsize=16)
axes[1].legend(fontsize=12)
axes[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.tight_layout(pad=3.0)
plt.show()

# 加载并测试最佳模型
model.load_weights(checkpoint_filepath)  # 只加载最优权重
loss, accuracy = model.evaluate(test_ds, steps=test_steps)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# 计算并打印 AUC
predictions = model.predict(test_ds, steps=test_steps)
true_labels = np.concatenate([y for x, y in test_ds], axis=0)
auc = roc_auc_score(true_labels, predictions, average='macro')
print(f'Test AUC: {auc}')

# 画confusion matrix
# 假设你有 n 个标签
num_labels = len(all_labels)
# 创建子图，分为三行
rows = 3
cols = (num_labels + 2) // rows  # 计算列数以确保每一行的子图数目尽量均匀
fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 6 * rows))  # 调整图像大小
fig.suptitle('Confusion Matrices for All Labels', fontsize=16)
axes = axes.flatten()  # 展平以方便索引
for i, label in enumerate(all_labels):
    # 将多标签问题分成每个单独的标签
    true_binary = (true_labels[:, i] > 0.5).astype(int)  # 真实标签
    pred_binary = (predictions[:, i] > 0.5).astype(int)  # 模型预测
    # 生成混淆矩阵
    cm = confusion_matrix(true_binary, pred_binary)
    # 绘制混淆矩阵到对应的子图中
    ax = axes[i]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
    disp.plot(cmap="Blues", values_format="d", ax=ax)
    ax.set_title(f'Confusion Matrix for {label}')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# 绘制ROC曲线
plt.figure(figsize=(12, 10))
for i, label in enumerate(all_labels):
    true_labels = test_df[label].values
    pred_probs = predictions[:, i]
    true_labels = np.array(true_labels, dtype=np.int32)
    fpr, tpr, _ = roc_curve(true_labels, pred_probs)
    roc_auc = auc_score(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC: {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=0.8)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves for Each Disease Label')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
