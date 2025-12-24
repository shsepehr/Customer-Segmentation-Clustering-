import pandas as pd
from sklearn.cluster import KMeans

data = pd.DataFrame({
    "age": [25, 45, 30, 35, 50],
    "income": [50000, 80000, 60000, 70000, 90000]
})

kmeans = KMeans(n_clusters=2, random_state=42)
data["segment"] = kmeans.fit_predict(data)
print(data)
