import cv2
import numpy as np
from sklearn.neural_network import MLPClassifier
from minisom import MiniSom
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import KMeans

def menu():
    print("Seleccione el tipo de red neuronal:")
    print("1. Perceptrón Multicapa (MLP)")
    print("2. Red Neuronal de Hopfield")
    print("3. Mapas Autoorganizados de Kohonen (SOM)")
    print("4. Redes de Base Radial (RBF)")
    opcion = input("Ingrese el número de la opción: ")
    return opcion

def iniciar_camara(procesar_frame=None):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return None

    print("Cámara iniciada correctamente.")
    cv2.namedWindow('Cámara', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Cámara', 800, 600)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo capturar el frame.")
            break

        if procesar_frame is not None:
            frame = procesar_frame(frame)

        cv2.imshow('Cámara', frame)
        key = cv2.waitKey(30)
        if key == 27:  # ESC para salir
            print("Cámara cerrada.")
            break
    cap.release()
    cv2.destroyAllWindows()

def normalizar_rgb(rgb):
    return [v / 255.0 for v in rgb]

def extraer_colores_dominantes(frame, n_colores=5):
    """Extrae los colores más dominantes del frame usando K-means"""
    # Redimensionar para procesar más rápido
    resized = cv2.resize(frame, (100, 100))
    # Convertir BGR a RGB
    rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # Reshape para K-means
    pixels = rgb_frame.reshape(-1, 3)
    
    # Aplicar K-means para encontrar colores dominantes
    kmeans = KMeans(n_clusters=n_colores, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    # Obtener colores y sus frecuencias
    colores = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    # Contar frecuencias
    frecuencias = np.bincount(labels)
    
    # Ordenar por frecuencia
    indices_ordenados = np.argsort(frecuencias)[::-1]
    
    return colores[indices_ordenados], frecuencias[indices_ordenados]

def filtrar_color_semaforo(colores, frecuencias):
    """Filtra y selecciona el color más probable del semáforo"""
    colores_norm = [normalizar_rgb(color) for color in colores]
    scores = []
    
    for i, color in enumerate(colores_norm):
        r, g, b = color
        freq_peso = frecuencias[i] / np.sum(frecuencias)
        
        # Scores para cada tipo de color de semáforo
        # VERDE: alto en G, bajo en R y B
        score_verde = g * (1 - r) * (1 - b) * (1 + freq_peso)
        
        # AMARILLO: alto en R y G, bajo en B
        score_amarillo = (r + g) * 0.5 * (1 - b) * (1 + freq_peso)
        # Bonus si R y G están balanceados para amarillo
        if abs(r - g) < 0.3:
            score_amarillo *= 1.5
        
        # ROJO: alto en R, bajo en G y B
        score_rojo = r * (1 - g) * (1 - b) * (1 + freq_peso)
        
        max_score = max(score_verde, score_amarillo, score_rojo)
        scores.append(max_score)
    
    # Seleccionar el color con mayor score
    mejor_idx = np.argmax(scores)
    return colores_norm[mejor_idx]

def obtener_color_semaforo(frame):
    """Obtiene el color más probable del semáforo del frame"""
    colores, frecuencias = extraer_colores_dominantes(frame, n_colores=5)
    return filtrar_color_semaforo(colores, frecuencias)

# Datos expandidos con más variaciones, especialmente para amarillo
# VERDE - Diferentes tonalidades
colores_verde = [
    [0, 255, 0],      # Verde puro
    [0, 200, 0],      # Verde medio
    [0, 150, 0],      # Verde oscuro
    [50, 255, 50],    # Verde claro
    [34, 139, 34],    # Verde bosque
    [0, 128, 0],      # Verde estándar
    [124, 252, 0],    # Verde césped
    [50, 205, 50],    # Verde lima
    [0, 100, 0],      # Verde muy oscuro
    [144, 238, 144],  # Verde claro pastel
    [32, 178, 32],    # Verde intermedio
    [0, 180, 0],      # Verde brillante medio
]

# AMARILLO - MÁS variaciones con énfasis en detección
colores_amarillo = [
    [255, 255, 0],    # Amarillo puro
    [255, 215, 0],    # Amarillo dorado
    [255, 255, 224],  # Amarillo muy claro
    [255, 250, 205],  # Amarillo crema
    [255, 228, 181],  # Amarillo suave
    [255, 235, 0],    # Amarillo brillante
    [255, 223, 0],    # Amarillo medio
    [255, 204, 0],    # Amarillo naranja claro
    [255, 191, 0],    # Amarillo dorado medio
    [255, 165, 0],    # Amarillo naranja
    [255, 140, 0],    # Amarillo oscuro
    [255, 255, 102],  # Amarillo claro verdoso
    [255, 255, 51],   # Amarillo claro
    [255, 246, 0],    # Amarillo cadmio
    [255, 215, 100],  # Amarillo dorado claro
    [230, 230, 0],    # Amarillo opaco
    [200, 200, 0],    # Amarillo oscuro opaco
    [255, 200, 50],   # Amarillo anaranjado
]

# ROJO - Diferentes tonalidades
colores_rojo = [
    [255, 0, 0],      # Rojo puro
    [220, 20, 60],    # Rojo carmesí
    [178, 34, 34],    # Rojo ladrillo
    [139, 0, 0],      # Rojo oscuro
    [255, 99, 71],    # Rojo tomate
    [255, 69, 0],     # Rojo naranja
    [200, 0, 0],      # Rojo medio
    [128, 0, 0],      # Rojo granate
    [255, 160, 122],  # Rojo claro
    [205, 92, 92],    # Rojo indio
    [255, 20, 20],    # Rojo brillante
    [180, 0, 0],      # Rojo oscuro medio
]

# Normalizar todos los colores
entradas_verde = [normalizar_rgb(color) for color in colores_verde]
entradas_amarillo = [normalizar_rgb(color) for color in colores_amarillo]
entradas_rojo = [normalizar_rgb(color) for color in colores_rojo]

# Crear arrays de entrenamiento
X_entrenamiento = np.array(entradas_verde + entradas_amarillo + entradas_rojo)
y_entrenamiento = np.array([0] * len(entradas_verde) + 
                          [1] * len(entradas_amarillo) + 
                          [2] * len(entradas_rojo))

etiquetas = ['Verde', 'Amarillo', 'Rojo']

# Mapeo de color a nivel de respuesta
niveles_respuesta = {
    'Verde': 'SEGURIDAD - AVANZAR',
    'Amarillo': 'PRECAUCIoN - PREPARARSE', 
    'Rojo': 'ALTO - DETENERSE'
}

def clasificar_color_forzado(color_actual):
    """Clasifica SIEMPRE en uno de los 3 colores usando múltiples métricas"""
    # Asegurar que color_actual sea un array de numpy
    color_actual = np.array(color_actual)
    r, g, b = color_actual
    
    # Múltiples métricas de clasificación
    scores = np.zeros(3)  # [verde, amarillo, rojo]
    
    # Convertir listas a arrays para cálculos
    entradas_verde_array = np.array(entradas_verde)
    entradas_amarillo_array = np.array(entradas_amarillo)
    entradas_rojo_array = np.array(entradas_rojo)
    
    # Métrica 1: Distancia euclidiana ponderada
    peso_distancia = 0.3
    dist_verde = min([np.sqrt(np.sum((color_actual - color_ref)**2)) for color_ref in entradas_verde_array])
    dist_amarillo = min([np.sqrt(np.sum((color_actual - color_ref)**2)) for color_ref in entradas_amarillo_array])
    dist_rojo = min([np.sqrt(np.sum((color_actual - color_ref)**2)) for color_ref in entradas_rojo_array])
    
    distancias = np.array([dist_verde, dist_amarillo, dist_rojo])
    scores += peso_distancia * (1 / (distancias + 0.01))  # Invertir distancias
    
    # Métrica 2: Análisis de componentes de color
    peso_componentes = 0.4
    
    # Verde: G dominante
    score_verde_comp = g * (1 - abs(r - 0.2)) * (1 - abs(b - 0.2))
    
    # Amarillo: R y G altos, B bajo - MÁS SENSIBLE
    score_amarillo_comp = (r + g) * 0.5 * (1 - b)
    # Bonus especial para amarillo si R y G están balanceados
    if abs(r - g) < 0.2 and r > 0.4 and g > 0.4 and b < 0.5:
        score_amarillo_comp *= 2.0
    # Bonus adicional si parece amarillo
    if r > 0.6 and g > 0.6 and b < 0.3:
        score_amarillo_comp *= 2.5
    
    # Rojo: R dominante
    score_rojo_comp = r * (1 - g) * (1 - b)
    
    scores += peso_componentes * np.array([score_verde_comp, score_amarillo_comp, score_rojo_comp])
    
    # Métrica 3: Similitud coseno
    peso_coseno = 0.3
    
    # Calcular vectores promedio para cada color
    verde_prom = np.mean(entradas_verde_array, axis=0)
    amarillo_prom = np.mean(entradas_amarillo_array, axis=0)
    rojo_prom = np.mean(entradas_rojo_array, axis=0)
    
    # Similitud coseno
    def coseno_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    
    sim_verde = coseno_sim(color_actual, verde_prom)
    sim_amarillo = coseno_sim(color_actual, amarillo_prom)
    sim_rojo = coseno_sim(color_actual, rojo_prom)
    
    scores += peso_coseno * np.array([sim_verde, sim_amarillo, sim_rojo])
    
    # Retornar el color con mayor score
    return etiquetas[np.argmax(scores)]

def mlp_colores():
    # MLP más robusto
    mlp = MLPClassifier(
        hidden_layer_sizes=(20, 15, 10), 
        max_iter=3000, 
        random_state=42,
        activation='relu', 
        solver='adam', 
        alpha=0.0001,
        learning_rate='adaptive',
        early_stopping=True,
        validation_fraction=0.1
    )
    mlp.fit(X_entrenamiento, y_entrenamiento)
    print("MLP entrenado para reconocer colores de semáforo.")
    print(f"Precisión del modelo: {mlp.score(X_entrenamiento, y_entrenamiento):.3f}")

    def procesar(frame):
        color_norm = np.array(obtener_color_semaforo(frame)).reshape(1, -1)
        
        # Predicción del MLP
        pred_prob = mlp.predict_proba(color_norm)[0]
        pred = mlp.predict(color_norm)[0]
        confianza = max(pred_prob)
        
        # Clasificación forzada como respaldo
        color_forzado = clasificar_color_forzado(color_norm[0])
        
        # Decidir qué clasificación usar
        if confianza > 0.5:
            color_final = etiquetas[pred]
        else:
            color_final = color_forzado
        
        nivel = niveles_respuesta[color_final]
        
        # Información adicional para debug
        r, g, b = color_norm[0]
        
        # Mostrar información
        cv2.putText(frame, f"Color: {color_final}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.putText(frame, f"Estado: {nivel}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.putText(frame, f"Confianza: {confianza:.2f}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        cv2.putText(frame, f"RGB: ({r:.2f}, {g:.2f}, {b:.2f})", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        
        return frame

    iniciar_camara(procesar)

class HopfieldNetwork:
    def __init__(self):
        self.weights = None

    def train(self, patterns):
        n = patterns.shape[1]
        self.weights = np.zeros((n, n))
        for p in patterns:
            self.weights += np.outer(p, p)
        np.fill_diagonal(self.weights, 0)
        self.weights /= patterns.shape[0]

    def predict(self, pattern, steps=15):
        s = pattern.copy()
        for _ in range(steps):
            s = np.sign(self.weights @ s)
        return s

def binarizar_color(color):
    return np.where(np.array(color) > 0.5, 1, -1)

def hopfield_colores():
    # Usar colores más representativos
    patrones_principales = np.array([
        binarizar_color(np.mean(entradas_verde, axis=0)),
        binarizar_color(np.mean(entradas_amarillo, axis=0)),
        binarizar_color(np.mean(entradas_rojo, axis=0))
    ])
    
    hopfield = HopfieldNetwork()
    hopfield.train(patrones_principales)
    print("Red de Hopfield entrenada para colores de semáforo.")

    def procesar(frame):
        color_norm = obtener_color_semaforo(frame)
        color_bin = binarizar_color(color_norm)
        salida = hopfield.predict(color_bin)
        
        distancias = [np.sum(salida != p) for p in patrones_principales]
        idx_hopfield = np.argmin(distancias)
        
        # Siempre usar clasificación forzada como respaldo
        color_forzado = clasificar_color_forzado(color_norm)
        
        # Decidir entre Hopfield y clasificación forzada
        if distancias[idx_hopfield] <= 3:
            color_final = etiquetas[idx_hopfield]
        else:
            color_final = color_forzado
            
        nivel = niveles_respuesta[color_final]
        
        cv2.putText(frame, f"Color: {color_final}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.putText(frame, f"Estado: {nivel}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        
        return frame

    iniciar_camara(procesar)

def som_colores():
    som = MiniSom(4, 4, 3, sigma=1.5, learning_rate=0.3, random_seed=42)
    som.train(X_entrenamiento, 3000, verbose=False)
    print("SOM entrenado para colores de semáforo.")
    
    # Mapeo mejorado
    mapa_etiquetas = {}
    for i, x in enumerate(X_entrenamiento):
        ganador = som.winner(x)
        etiqueta = etiquetas[y_entrenamiento[i]]
        if ganador not in mapa_etiquetas:
            mapa_etiquetas[ganador] = {}
        if etiqueta not in mapa_etiquetas[ganador]:
            mapa_etiquetas[ganador][etiqueta] = 0
        mapa_etiquetas[ganador][etiqueta] += 1
    
    # Asignar etiqueta más frecuente
    mapa_final = {}
    for pos, counts in mapa_etiquetas.items():
        mapa_final[pos] = max(counts, key=counts.get)

    def procesar(frame):
        color_norm = np.array(obtener_color_semaforo(frame))
        ganador = som.winner(color_norm)
        
        # Siempre tener un resultado
        if ganador in mapa_final:
            color_final = mapa_final[ganador]
        else:
            color_final = clasificar_color_forzado(color_norm)
        
        nivel = niveles_respuesta[color_final]
        
        cv2.putText(frame, f"Color: {color_final}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.putText(frame, f"Estado: {nivel}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        
        return frame

    iniciar_camara(procesar)

class RBFClassifier:
    def __init__(self, gamma=1.0):
        self.gamma = gamma
        self.centros = None
        self.labels = None

    def fit(self, X, y):
        self.centros = X
        self.labels = y

    def predict(self, X):
        K = rbf_kernel(X, self.centros, gamma=self.gamma)
        idx = np.argmax(K, axis=1)
        return self.labels[idx]

def rbf_colores():
    rbf = RBFClassifier(gamma=20)
    rbf.fit(X_entrenamiento, y_entrenamiento)
    print("RBF entrenado para colores de semáforo.")

    def procesar(frame):
        color_norm = np.array(obtener_color_semaforo(frame)).reshape(1, -1)
        pred = rbf.predict(color_norm)[0]
        
        # Asegurar resultado válido
        if 0 <= pred < len(etiquetas):
            color_final = etiquetas[pred]
        else:
            color_final = clasificar_color_forzado(color_norm[0])
            
        nivel = niveles_respuesta[color_final]
        
        cv2.putText(frame, f"Color: {color_final}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.putText(frame, f"Estado: {nivel}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        
        return frame

    iniciar_camara(procesar)

if __name__ == "__main__":
    print("=== SISTEMA AVANZADO DE RECONOCIMIENTO DE SEMÁFORO ===")
    print("🟢 VERDE: Seguridad - Avanzar")
    print("🟡 AMARILLO: Precaución - Prepararse") 
    print("🔴 ROJO: Alto - Detenerse")
    print("Sistema optimizado para filtrar ruido de fondo")
    print()
    
    opcion = menu()
    if opcion == "1":
        print("Seleccionaste Perceptrón Multicapa (MLP)")
        mlp_colores()
    elif opcion == "2":
        print("Seleccionaste Red Neuronal de Hopfield")
        hopfield_colores()
    elif opcion == "3":
        print("Seleccionaste Mapas Autoorganizados de Kohonen (SOM)")
        som_colores()
    elif opcion == "4":
        print("Seleccionaste Redes de Base Radial (RBF)")
        rbf_colores()
    else:
        print("Opción no válida.")