#%% Régression linéaire
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def load_data(filename):
    """
    Charge les données 2D depuis un fichier texte (2 colonnes x et y).
    
    Parameters
    ----------
    filename : str
    Nom du fichier texte à charger.
    
    Returns
    -------
    data : np.ndarray or None
    Tableau (n, 2) des données chargées, ou None si échec.
    """
    try:
        data = np.loadtxt(filename)
        print("Données chargées depuis", 'filename')
        return data
    except Exception as exc:
        print(f"Erreur lors du chargement de '{filename}': {exc}")
        return None

def regression_line(x, y):
    """
    Calcule la droite de régression linéaire pour les données fournies.
    
    Parameters
    ----------
    x : np.ndarray
    Tableau des valeurs de l'axe des abscisses (indépendantes).
    y : np.ndarray
    Tableau des valeurs de l'axe des ordonnées (dépendantes).
    
    Returns
    -------
    x_sorted : np.ndarray
    Valeurs de x triées par ordre croissant.
    y_sorted : np.ndarray
    Valeurs de y correspondantes à x_sorted.
    slope : float
    Pente (coefficient directeur) de la droite de régression.
    intercept : float
    Ordonnée à l'origine de la droite de régression.
    r_value : float
    Coefficient de corrélation.
    p_value : float
    p-valeur de la régression (test statistique).
    std_err : float
    Erreur standard de l'estimation de la pente.
    """
    # Tri des données selon x
    indices = np.argsort(x)
    x_sorted = x[indices]
    y_sorted = y[indices]
    
    # Régression linéaire
    slope, intercept, r_value, p_value, std_err = linregress(x_sorted, y_sorted)
    return x_sorted, y_sorted, slope, intercept, r_value, p_value, std_err


def plot_regression(datasets):
    """
    Trace la régression linéaire pour plusieurs ensembles de données
    sur 4 subplots (2x2).
    
    Parameters
    ----------
    datasets : list of tuples
    Liste contenant des tuples (x, y, label) pour chaque dataset.
    - x : np.ndarray
    - y : np.ndarray
    - label : str, label du set de données (ex: "I", "II", etc.)
    """
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.ravel() # Facilite l'itération
    
    for i, (x, y, set_label) in enumerate(datasets):
    # Calcul de la régression linéaire
        (x_sorted, y_sorted,
        slope, intercept,
        r_value, p_value,
        std_err) = regression_line(x, y)
    
        # Droite de régression
        y_reg = slope * x_sorted + intercept
        
        # Tracé du nuage de points
        axs[i].scatter(x, y, label="Données originales")
        # Tracé de la droite de régression
        axs[i].plot(x_sorted, y_reg, 'r',
        label=f"y = {slope:.3f}x + {intercept:.3f}")
        axs[i].set_title(f"Set {set_label}")
        axs[i].set_xlabel("x")
        axs[i].set_ylabel("y")
        axs[i].legend()
        axs[i].grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Données du tableau (Sets I, II, III, IV)
    
    data = load_data('Documents/regression_lineaire.txt')
    x1 = data[:,0]
    y1 = data[:,1]
    
    x2 = data[:,2]
    y2 = data[:,3]
    
    x3 = data[:,4]
    y3 = data[:,5]
    
    x4 = data[:,6]
    y4 = data[:,7]
    
    # Préparation des sets
    datasets_regression = [
    (x1, y1, "I"),
    (x2, y2, "II"),
    (x3, y3, "III"),
    (x4, y4, "IV")
    ]
    
    # Tracé de la régression pour chaque set
    plot_regression(datasets_regression)
    

#%% Régression Numpy
import numpy as np
import matplotlib.pyplot as plt


def rotate_points(points, angle_deg):
    """
    Fait tourner un ensemble de points 2D d'un angle donné (degrés).
    
    Parameters
    ----------
    points : np.ndarray
    Tableau de forme (n, 2) contenant les coordonnées (x, y).
    angle_deg : float
    Angle de rotation en degrés.
    
    Returns
    -------
    rotated : np.ndarray
    Tableau (n, 2) des points après rotation.
    """
    # Convertir l'angle en radians
    angle_rad = np.deg2rad(angle_deg)
    
    # Matrice de rotation
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
    [np.sin(angle_rad), np.cos(angle_rad)]])
    # Produit matriciel pour appliquer la rotation
    rotated = points.dot(rotation_matrix.T)
    return rotated


def load_data(filename):
    """
    Charge les données 2D depuis un fichier texte (2 colonnes x et y).
    
    Parameters
    ----------
    filename : str
    Nom du fichier texte à charger.
    
    Returns
    -------
    data : np.ndarray or None
    Tableau (n, 2) des données chargées, ou None si échec.
    """
    try:
        data = np.loadtxt(filename)
        print("Données chargées depuis", 'filename')
        return data
    except Exception as exc:
        print(f"Erreur lors du chargement de '{filename}': {exc}")
        return None


def plot_rotations_cumulative(data):
    """
    Trace sur un même graphique la rotation d'un ensemble de points
    pour chaque angle de 0° à 350° (par pas de 10°).
    
    Parameters
    ----------
    data : np.ndarray
    Tableau (n, 2) représentant les points à faire tourner.
    """
    # Liste d'angles
    angles = np.arange(0, 360, 10)
    
    # Préparation de la figure
    plt.figure(figsize=(6, 6))
    # Palette de couleurs
    cmap = plt.get_cmap("hsv")

    for i, angle in enumerate(angles):
        rotated = rotate_points(data, angle)
        color = cmap(i / len(angles))
        plt.plot(rotated[:, 0], rotated[:, 1], color=color, lw=1.5)
    
    plt.title("Rotations cumulées (0° à 350° par pas de 10°)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    # Lecture des données depuis un fichier externe ou usage de données simulées
    filename = "Documents/fichier_resultat.txt"
    data_points = np.loadtxt(filename)

    if data_points is None:
        print("Utilisation de données simulées pour la rotation.")
        # Exemple : motif ondulé basé sur un rayon variable
        theta = np.linspace(0, 2 * np.pi, 50)
        r = 1 + 0.5 * np.sin(3 * theta)
        x_sim = r * np.cos(theta)
        y_sim = r * np.sin(theta)
        data_points = np.column_stack((x_sim, y_sim))   
    
        # Tracé cumulatif des rotations de 0° à 350°
    plot_rotations_cumulative(data_points)

