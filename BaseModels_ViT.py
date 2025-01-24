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
parser.add_argument('--model', type=str, help='Choose model: densenet121, vgg16, mobilenetV3Large, efficientnetB0, or vit')
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
data = pd.read_csv('/home2/grkp39/My_project/Datasets/Data_Entry_2017.csv')   # 替换为实际路径

# 去除年龄大于100的异常数据并且将年龄数据转换为整数
data = data[data['Patient Age'] < 100]
data['Patient Age'] = data['Patient Age'].map(lambda x: int(x))

# 获取图像路径，并存入字典
data_image_paths = {
    os.path.basename(x): x for x in glob(os.path.join('/home2/grkp39/My_project/Datasets', 'images*', '*', '*.png'))  # 替换为实际路径
}

# 将图像路径添加到数据框中同时确保路径为字符串并删除缺失路径的行
data['path'] = data['Image Index'].map(data_image_paths.get)
data['path'] = data['path'].astype(str)
data = data.dropna(subset=['path'])

# 打印数据概况，随机查看3条样本数据
print('Scans found:', len(data_image_paths), ', Total Headers:', data.shape[0])
print(data.sample(3))

# 提取所有独特的标签并且打印全部标签
all_labels = np.unique(list(chain(*data['Finding Labels'].map(lambda x: x.split('|')).tolist())))
all_labels = [x for x in all_labels if len(x) > 0]
print('All Labels ({}): {}'.format(len(all_labels), all_labels))

# 为每个病症标签创建独立的列
for c_label in all_labels:
    if len(c_label) > 1:  # 移除空标签
        data[c_label] = data['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)

# 首先创建所有样本的 'disease_vec' 列
data['disease_vec'] = data[all_labels].values.tolist()

# 加载图像索引
test_list_path = '/home2/grkp39/My_project/Datasets/test_list.txt'  # 替换为实际路径
if not os.path.exists(test_list_path):
    raise FileNotFoundError(f"Test list file not found at {test_list_path}")
with open(test_list_path, 'r') as f:
    test_image_names = f.read().splitlines()

# 分离训练数据和测试数据
train_data = data[~data['Image Index'].isin(test_image_names)]
test_data = data[data['Image Index'].isin(test_image_names)].dropna(subset=['path'])

# 从训练数据中分离验证集
train_data, valid_data = train_test_split(train_data, test_size=0.1, random_state=42)
print(f"Data size for training: {len(train_data)}")
print(f"Data size for validation: {len(valid_data)}")
print(f"Data size for testing: {len(test_data)}")

# 创建可视化保存目录
vis_dir = './visualizations'
if not os.path.exists(vis_dir):
    os.makedirs(vis_dir)

# 数据分布可视化
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
data[all_labels].sum().sort_values(ascending=False).plot(kind='bar')
plt.title('Disease Distribution in Dataset')
plt.xlabel('Disease')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.subplot(1, 2, 2)
sns.histplot(data=data, x='Patient Age', bins=30)
plt.title('Age Distribution in Dataset')
plt.xlabel('Age')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, 'data_distribution.png'), dpi=300, bbox_inches='tight')
plt.show()



# ============================================================Part 3 Data Load and preprocessing with different models============================================================
from keras.applications.densenet import preprocess_input as densenet_preprocess_input
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from keras.applications.mobilenet_v3 import preprocess_input as mobilenet_v3_preprocess_input
from keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input
from transformers import ViTImageProcessor
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
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
        if model_name == 'vit':
            # ViT特定的预处理
            processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
            image = Image.fromarray(tf.keras.preprocessing.image.array_to_img(image).numpy())
            processed = processor(images=image, return_tensors="tf")
            return processed['pixel_values'][0], label
        else:
            # 其他模型的预处理保持不变
            if model_name == 'densenet121':
                image = densenet_preprocess_input(image)
            elif model_name == 'vgg16':
                image = vgg16_preprocess_input(image)
            elif model_name == 'mobilenetV3Large':
                image = mobilenet_v3_preprocess_input(image)
            elif model_name == 'efficientnetB0':
                image = efficientnet_preprocess_input(image)
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
from transformers import ViTForImageClassification
from keras.layers import Input, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model
from sklearn.metrics import roc_curve, auc as auc_score
# 将数据分为训练、验证和测试集（change using offical test）
IMG_SIZE = (224, 224, 3)
img_in = Input(shape=IMG_SIZE)
if args.model == 'vit':
    # 加载预训练的ViT模型
    vit_model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=len(all_labels),
        ignore_mismatched_sizes=True
    )
    # 创建自定义的Keras模型包装ViT
    class ViTKerasModel(tf.keras.Model):
        def __init__(self, vit_model):
            super().__init__()
            self.vit = vit_model
            
        def call(self, inputs):
            outputs = self.vit(inputs)
            return tf.nn.sigmoid(outputs.logits)
    model = ViTKerasModel(vit_model)

elif args.model == 'densenet121':
    base_model = DenseNet121(
        include_top=False, 
        weights='imagenet', 
        input_tensor=img_in, 
        pooling='avg'
    )

elif args.model == 'vgg16':
    base_model = VGG16(
        include_top=False, 
        weights='imagenet', 
        input_tensor=img_in, 
        pooling='avg'
    )

elif args.model == 'mobilenetV3Large':
    base_model = MobileNetV3Large(
        include_top=False, 
        weights='imagenet', 
        input_tensor=img_in, 
        pooling='avg'
    )

elif args.model == 'efficientnetB0':
    base_model = EfficientNetB0(
        include_top=False, 
        weights='imagenet', 
        input_tensor=img_in, 
        pooling='avg'
    )

else:
    raise ValueError("Invalid model name. Choose 'densenet121', 'vgg16', 'mobilenetV3Large', 'efficientnetB0', or 'vit'.")

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

# 优化器
if args.model == 'vit':
    optimizer = Adam(
        learning_rate=args.learning_rate,
        weight_decay=0.0001  # 添加权重衰减
    )
else:
    optimizer = Adam(learning_rate=args.learning_rate)
# 编译模型
model.compile(
    optimizer=optimizer, 
    loss='binary_crossentropy', 
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.AUC(name='auc', multi_label=True)  # 确保使用多标签AUC
    ]
)



# ======================================================================Part 5 Training and Testing======================================================================
# 数据增强
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# 创建训练、验证和测试集的数据集
batch_size = args.batch_size
train_ds = datagen.flow_from_dataframe(
    dataframe=train_data, 
    directory=None, x_col='path', 
    y_col=all_labels, target_size=(224, 224), 
    batch_size=batch_size, class_mode='raw'
)
valid_ds = dataframe_to_dataset(valid_data, model_name=args.model).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = dataframe_to_dataset(test_data, model_name=args.model).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

# 计算 steps_per_epoch 和 validation_steps
steps_per_epoch = np.ceil(len(train_data) / batch_size).astype(int)
validation_steps = np.ceil(len(valid_data) / batch_size).astype(int)
test_steps = np.ceil(len(test_data) / batch_size).astype(int)

# 设置模型检查点，保存最优模型
save_dir = './models'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
checkpoint_filepath = os.path.join(save_dir, f'best_model_{args.model}.h5')
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    mode='min',
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

# 训练过程可视化
fig, axes = plt.subplots(3, 1, figsize=(18, 16))  # 改为3行以容纳AUC图

# 绘制训练和验证损失
axes[0].plot(history.history['loss'], label='Training Loss', color='b', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Validation Loss', color='r', linewidth=2)
axes[0].set_xlabel('Epochs', fontsize=14)
axes[0].set_ylabel('Loss', fontsize=14)
axes[0].set_title('Training and Validation Loss', fontsize=16)
axes[0].legend(fontsize=12)
axes[0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

# 绘制训练和验证准确率
axes[1].plot(history.history['accuracy'], label='Training Accuracy', color='b', linewidth=2)
axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', color='r', linewidth=2)
axes[1].set_xlabel('Epochs', fontsize=14)
axes[1].set_ylabel('Accuracy', fontsize=14)
axes[1].set_title('Training and Validation Accuracy', fontsize=16)
axes[1].legend(fontsize=12)
axes[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

# 绘制训练和验证AUC
axes[2].plot(history.history['auc'], label='Training AUC', color='b', linewidth=2)
axes[2].plot(history.history['val_auc'], label='Validation AUC', color='r', linewidth=2)
axes[2].set_xlabel('Epochs', fontsize=14)
axes[2].set_ylabel('AUC', fontsize=14)
axes[2].set_title('Training and Validation AUC', fontsize=16)
axes[2].legend(fontsize=12)
axes[2].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

plt.tight_layout(pad=3.0)
plt.savefig(os.path.join(vis_dir, f'training_history_{args.model}.png'), dpi=300, bbox_inches='tight')
plt.show()

# 加载并测试最佳模型
model.load_weights(checkpoint_filepath)  # 只加载最优权重
test_results = model.evaluate(test_ds, steps=test_steps)
print(f'Test Loss: {test_results[0]:.4f}')
print(f'Test Accuracy: {test_results[1]:.4f}')
print(f'Test AUC: {test_results[2]:.4f}')

# 预测并绘制每个类别的ROC曲线
predictions = model.predict(test_ds, steps=test_steps)
plt.figure(figsize=(12, 10))
aucs = []  # 存储每个类别的AUC值

for i, label in enumerate(all_labels):
    true_labels = test_data[label].values
    pred_probs = predictions[:, i]
    fpr, tpr, _ = roc_curve(true_labels, pred_probs)
    roc_auc = auc_score(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC: {roc_auc:.2f})')

# 计算并打印平均AUC
mean_auc = np.mean(aucs)
print(f'Mean AUC across all classes: {mean_auc:.4f}')

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=0.8)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curves for Each Disease Label (Mean AUC: {mean_auc:.2f})')
plt.legend(loc='lower right', bbox_to_anchor=(1.7, 0))
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, f'roc_curves_{args.model}.png'), dpi=300, bbox_inches='tight')
plt.show()
