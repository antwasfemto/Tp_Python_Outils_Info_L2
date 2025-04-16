#%% EX 1
import matplotlib.pyplot as plt

w_px = 500
h_px = 300
dpi = 100

# convesion en pouces
w_inch = w_px / dpi
h_inch = h_px / dpi

# création de la figure
fig = plt.figure(figsize=(w_inch, h_inch), dpi=dpi, facecolor='blue')

plt.axis('off')
plt.show()

#%% EX 2
import matplotlib.pyplot as plt
w_px = 500
h_px = 300
dpi = 100

# convesion en pouces
w_inch = w_px / dpi
h_inch = h_px / dpi

# création de la figure
plt.figure(figsize=(w_inch, h_inch), dpi=dpi, facecolor='blue')

plt.axes([0.25, 0.25, 0.5, 0.5])
plt.show()

#%% EX 3
import matplotlib.pyplot as plt
w_px = 500
h_px = 300
dpi = 100

# convesion en pouces
w_inch = w_px / dpi
h_inch = h_px / dpi

# création de la figure
fig = plt.figure(figsize=(w_inch, h_inch), dpi=dpi, facecolor='blue')

plt.subplot(2, 2, 1)
plt.subplot(2, 2, 2)
plt.subplot(212)
plt.show()

#%% EX 4
import numpy as np
v = np.array([1,2,3.4])*np.pi/4
w = np.sin(v)
print(w)

#%% EX 5
# np.zeros() Renvoie un nouveau tableau de forme et de type donnés, rempli de zéros.

# np.ones() Renvoie un nouveau tableau de forme et de type donnés, rempli de un.

# np.linspace() Renvoie le nombre d'échantillons régulièrement espacés, calculé 
# sur l'intervalle [début, fin].La fin de l'intervalle peut éventuellement être exclue.

# np.random.random() Renvoie un nombre aléatoirement choisi

# np.arange() Renvoie des valeurs régulièrement espacées dans un intervalle donné,
# et pouvant être appelé avec un nombre variable d'arguments positionnels
"""
Rapel tp 1 : 
3.3 Exercice Récap

from math import pi, sin

x = 100 * [0]  # création d'une liste x formée de 100 zéros  
y = 100 * [0]  # création d'une liste y formée de 100 zéros  

for i in range(100):  
    x[i] = 4 * pi * i / 100  # x s'incrémente de 0 à 4π sur 100 valeurs
    y[i] = sin(x[i])        # y est égal au sinus de x
    
    if x[i] == 2 * pi:
        print("La première période est passée")  
"""

import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

# Générer les x entre 0 et 2π avec linspace (100 points)
x = np.linspace(0, 2 * np.pi, 100)

# Générer les valeurs sinus correspondantes
y = np.sin(x)

# Ajouter un peu de bruit aléatoire (optionnel, pour voir l'effet de random)
bruit = 0.1 * np.random.random(100)  # Valeurs aléatoires entre 0 et 0.1
y_bruite = y + bruit

# Exemple d'utilisation de zeros et ones juste pour montrer
zero_line = np.zeros_like(x)
one_line = np.ones_like(x)
minus_one_line = -np.ones_like(x)
# ajout d'une ligne verticale 
axv = np.array([[np.pi,np.pi],[-1,1]]) 

plt.plot(x, y, label='sin(x)', color='blue')
plt.plot(x, y_bruite, label='sin(x) + bruit', color='orange', linestyle='--')
plt.plot(x, zero_line, label='zeros', linestyle='-', color='gray')
plt.plot(x, one_line, label='ones', linestyle='-', color='green')
plt.plot(x, minus_one_line, label='minus ones', linestyle='-', color='green')
plt.plot(axv[0], axv[1], label='vertical', linestyle='-', color='gray')

plt.title('Courbe de sin(x) avec Numpy')
plt.xlabel('x')
plt.ylabel('y')
#plt.legend()
plt.grid(False)
plt.show()

#%% EX 6
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

# Générer les x entre 0 et10π avec linspace
x = np.linspace(0, 10 * np.pi, 500)

# Générer les valeurs sinus correspondantes
y1 = np.cos(x)
y2 = np.exp(-x/10)*np.cos(x)

plt.plot(x,y1,"g-",label="cos")
plt.plot(x,y2,"r-",label="exp*cos")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Courbe de cos(x) amorti avec Numpy")
plt.grid(True)
plt.legend()
plt.show()

#%% EX 7
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

t1 = np.linspace(0, 2 * np.pi, 1000)
x1 = np.sin(t1) / (1 + np.cos(t1)**2)
y1 = (np.sin(t1) * np.cos(t1)) / (1 + np.cos(t1)**2)

plt.figure()
plt.plot(x1,y1)
plt.title("La Lemniscate de Bernoulli")
plt.show()

t2 = np.linspace(0, 10 * np.pi, 1000)
x2 = t2 * np.cos(t2)
y2 = t2 * np.sin(t2)

plt.figure()
plt.plot(x2,y2)
plt.title("La spirale d’Archimède")
plt.show()

t3 = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
x3 = 16 * np.sin(t3)**3
y3 = 13 * np.cos(t3) - 5 * np.cos(2 * t3) - 2 * np.cos(3 * t3) - np.cos(4 * t3)

plt.figure()
plt.plot(x3,y3)
plt.title("La courbe du cœur")
plt.show()

q4 = 8
p4 = 2
t4 = np.linspace(0, 2 * q4 * np.pi, 1000)
x4 = (1+np.cos(p4*t4/q4))*np.cos(t4)
y4 = (1+np.cos(p4*t4/q4))*np.sin(t4)

plt.figure()
plt.plot(x4,y4)
plt.title("Les cyclo-harmoniques")

plt.show()
#%% EX 8

import numpy as np
import matplotlib.pyplot as plt

# Création de l'axe des abscisses
x = np.linspace(0, 2, 100)

# Demande du nombre de fonctions à tracer
try:
    n = int(input("Entrez la valeur de n (nombre de fonctions x^n à tracer) : "))
    
    for i in range(1, n + 1):
        y = x**i
        plt.plot(x, y, label=f"$x^{i}$")  # Ajout de la légende dynamiquement

    plt.title(f"Fonctions de xⁿ pour n de 1 à {n}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc='upper left')  # Légende en haut à gauche
    plt.grid(True)
    plt.show()

except ValueError:
    print("Veuillez entrer un nombre entier valide.")

#%% EX 9
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('Documents/fichier_resultat.txt')  # Assurez-vous que le fichier est dans le même dossier que le script et que le répertoir de travail
x = data[:, 0]  # Première colonne
y = data[:, 1]  # Deuxième colonne

# Tracer les données
plt.plot(x, y, marker='o',linestyle='', color='b', label='Données chargées')
plt.title("Tracé des données depuis fichier_resultat.txt")
plt.xlabel("axe des x")
plt.ylabel("axe des y")
plt.legend()
plt.grid(True)
plt.show()

#%% EX 10
import numpy as np
import matplotlib.pyplot as plt

def f(t):
    return np.exp(-t) * np.cos(2 * np.pi * t)

# Génération des données
t1 = np.arange(0, 5, 0.1)
t2 = np.arange(0, 5, 0.02)

# Figure 1 - Tracé de la fonction f(t)
plt.figure(1)
plt.subplot(221)
plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k-')   # points bleus
plt.grid(True)
plt.margins(0.1)

# Figure 2 - Tracé de cos(2πt)
plt.subplot(222)
plt.plot(t2, np.cos(2 * np.pi * t2), 'r--')
plt.grid(True)
plt.margins(0)

# Figure 3 - Tracé de sin(2πt)
plt.subplot(212)
plt.plot(t2, np.sin(2 * np.pi * t2), 'b-')
plt.grid(True)
plt.margins(x=0, y=0.001)

plt.show()

#%% EX 11
import matplotlib.pyplot as plt
import numpy as np
plt.close('all')

# Données
groupes = ['Gr. 1', 'Gr. 2', 'Gr. 3', 'Gr. 4', 'Gr. 5']
hommes = [20, 35, 30, 35, 27]
femmes = [25, 32, 34, 20, 25]

# Position des barres
x = np.arange(len(groupes))  # [0, 1, 2, 3, 4]
width = 0.35  # Largeur des barres

# Création de la figure
plt.figure(figsize=(8, 5))

# Barres pour les hommes et les femmes
plt.bar(x - width/2, hommes, width, label='Hommes', color='blue')
plt.bar(x + width/2, femmes, width, label='Femmes', color='magenta')

# Étiquettes des axes et titre
plt.ylabel('Scores')
plt.title('Scores par groupe et par genre')
plt.xticks(x, groupes)  # Remplace les indices par les noms des groupes
plt.legend()

# Affichage
plt.grid(False)
plt.tight_layout()


#%% EX 12
import matplotlib.pyplot as plt
plt.close('all')

# Données
labels = ['Science et technologie', 'Science du sport', 'Droit, économie, gestion', 'Arts lettres et langues', 'Science humaines et sociales']
sizes = [27.9, 6.3, 25.3, 20.8, 19.7]  # en pourcentage
explode = (0.1, 0.0, 0, 0, 0)  # Mettre en valeur "Physique" (2e part)

# Création du camembert
plt.figure(figsize=(6, 6))
plt.pie(
    sizes,
    explode=explode,
    labels=labels,
    autopct='%1.1f%%',  # Affiche 1 chiffre après la virgule
    shadow=True,
    startangle=0
)


#%% EX 13

import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

# Domaine de θ
theta = np.linspace(0, 2 * np.pi, 1000)

# Équation paramétrique
r = np.abs(np.sqrt(5 / (16 * np.pi)) * 3 * np.cos(theta)**2 - 1)
r2 = np.abs(np.sqrt(5 / (16 * np.pi)) * 3 * np.sin(theta)**2 - 1)

# Tracé en coordonnées polaires
plt.figure(figsize=(6,6))
plt.subplot(polar=True)
plt.plot(theta, r, color='blue')
plt.plot(theta, r2, color='orange')  # Courbe complémentaire pour esthétique

#%% EX 14
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

# Création des données
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = 5 * np.sqrt(X**2 + Y**2) + np.sin(X**2 + Y**2)

# Contour
plt.figure()
contour = plt.contourf(X, Y, Z, cmap='viridis', levels=100)
contour_lines = plt.contour(X, Y, Z, colors='black')
plt.clabel(contour_lines, inline=True, fontsize=12)
plt.colorbar(contour)


# Surface 3D
plt.figure()
ax = plt.subplot(projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.view_init(20,-35)


# Affichage
plt.tight_layout()
plt.show()
