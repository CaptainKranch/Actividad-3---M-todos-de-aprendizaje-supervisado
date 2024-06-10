import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Crear un dataset de muestra
data = {
    'Feature1': [1, 2, 2.5, 5, 4, 6, 7, 8],
    'Feature2': [1, 2, 3, 4, 5, 6, 7, 8]
}

df = pd.DataFrame(data)

# Crear y entrenar el modelo de K-Means
kmeans = KMeans(n_clusters=2)
kmeans.fit(df)

# Obtener las etiquetas de los clusters asignados a cada instancia
labels = kmeans.labels_

# Visualizar los clusters y guardar la figura en un archivo
plt.scatter(df['Feature1'], df['Feature2'], c=labels, cmap='viridis')
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.title('Clustering con K-Means')
plt.savefig('clustering_result.png')
