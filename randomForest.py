from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加载鸢尾花数据集。经典的分类任务的数据集，
# 每一个数据点包含四个花的特征，以及对应的品种，目的是根据俄四个特征去对一朵新的鸢尾花花进行分类
iris = load_iris()
# X代表数据的四个特征，y代表该数据所属的鸢尾花种类
X, y = iris.data, iris.target

# 划分训练集和测试集，按照8:2分配。8成的数据集用于训练，两成数据集用于测试
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建随机森林分类器，
# n_estimators=随机森林中决策数的数量，越多的数量越大越好，但占用的内存与训练和预测的时间也会相应增长，且边际效益是递减的。
# random_state 随机种子
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练模型。利用训练集合测试数据
clf.fit(X_train, y_train)

# 预测，对测试集预测鸢尾花的种类
y_pred = clf.predict(X_test)

# 评估模型，根据预测的鸢尾花种类和实际的数据记录种类对比
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
