from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

digits = load_digits()
digits
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
Accuracy: 0.8518518518518519
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

digits = datasets.load_digits()
X, y = digits.data, digits.target
images = digits.images 

X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(
    X, y, images, test_size=0.25, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = metrics.accuracy_score(y_test, y_pred) * 100
print(f"Accuracy: {accuracy:.2f}%")

n_images = 10
fig, axes = plt.subplots(2, n_images, figsize=(12, 4))

for i in range(n_images):
    axes[0, i].imshow(images_train[i], cmap='gray')
    axes[0, i].set_title(f"Train: {y_train[i]}")
    axes[0, i].axis('off')
for i in range(n_images):
    axes[1, i].imshow(images_test[i], cmap='gray')
    axes[1, i].set_title(f"P:{y_pred[i]} / T:{y_test[i]}")
    axes[1, i].axis('off')

plt.suptitle("Top: Original Training Images | Bottom: Test Images with Predictions", fontsize=14)
plt.tight_layout()
plt.show()
Accuracy: 85.56%
