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
