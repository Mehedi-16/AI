
```

# ==== Simple Iris Classification using TensorFlow (CSV version) ====

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import models, layers

# 2️⃣ ডেটা লোড (CSV থেকে)
df = pd.read_csv("iris.csv")
X = df.drop("target", axis=1).values
y = df["target"].values

# 3️⃣ ট্রেন-টেস্ট ভাগ করা
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4️⃣ ডেটা স্কেল করা
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc  = sc.transform(X_test)

# 5️⃣ মডেল তৈরি করা
model = models.Sequential([
    layers.Dense(32, activation='relu', input_shape=(X_train_sc.shape[1],)),
    layers.Dense(16, activation='relu'),
    layers.Dense(3, activation='softmax')
])

# 6️⃣ মডেল কম্পাইল করা
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 7️⃣ মডেল ট্রেন করা
model.fit(X_train_sc, y_train, epochs=20, batch_size=16, verbose=0)

# 8️⃣ মডেল টেস্ট করা
pred = model.predict(X_test_sc, verbose=0).argmax(axis=1)
print("✅ Test Accuracy:", accuracy_score(y_test, pred))

```





```

# ==== Simple Iris Classification using TensorFlow ====

# 1️⃣ প্রয়োজনীয় লাইব্রেরি ইমপোর্ট
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import models, layers

# 2️⃣ ডেটা লোড
data = load_iris()
X, y = data.data, data.target

# 3️⃣ ট্রেন-টেস্ট ভাগ করা
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4️⃣ ডেটা স্কেল করা
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc  = sc.transform(X_test)

# 5️⃣ মডেল তৈরি করা
model = models.Sequential([
    layers.Dense(32, activation='relu', input_shape=(X_train_sc.shape[1],)),
    layers.Dense(16, activation='relu'),
    layers.Dense(3, activation='softmax')   # 3 ক্লাস → setosa, versicolor, virginica
])

# 6️⃣ মডেল কম্পাইল করা
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 7️⃣ মডেল ট্রেন করা
model.fit(X_train_sc, y_train, epochs=20, batch_size=16, verbose=0)

# 8️⃣ মডেল টেস্ট করা
pred = model.predict(X_test_sc, verbose=0).argmax(axis=1)
print("✅ Test Accuracy:", accuracy_score(y_test, pred))

# 9️⃣ নির্দিষ্ট একটি ফুল চেক করা (উদাহরণ: নম্বর 10)
num = 10
sample_sc = sc.transform(X_test[[num]])
pred = model.predict(sample_sc, verbose=0).argmax(axis=1)[0]

print(f"\nNumber: {num}")
print("Predicted:", data.target_names[pred])
print("Actual:", data.target_names[y_test[num]])


```
