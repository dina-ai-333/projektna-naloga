import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
import sys

# Povečamo limit rekurzije za Flood Fill
sys.setrecursionlimit(100000)


# ROČNI UVOZ IN VIZUALIZACIJA ASCII STL
def nalozi_ascii_stl(pot_do_datoteke):
    trikotniki = []
    normale = []
    trenutni_trikotnik = []
    
    try:
        with open(pot_do_datoteke, 'r', encoding='utf-8', errors='ignore') as f:
            for vrstica in f:
                vrstica = vrstica.strip()
                if vrstica.startswith("facet normal"):
                    # Preberemo normalo
                    deli = vrstica.split()
                    n = [float(deli[2]), float(deli[3]), float(deli[4])]
                    normale.append(n)
                elif vrstica.startswith("vertex"):
                    # Preberemo oglišče trikotnika
                    deli = vrstica.split()
                    v = [float(deli[1]), float(deli[2]), float(deli[3])]
                    trenutni_trikotnik.append(v)
                    if len(trenutni_trikotnik) == 3:
                        trikotniki.append(trenutni_trikotnik)
                        trenutni_trikotnik = []
    except Exception as e:
        print(f"Napaka pri branju datoteke: {e}")
        return None

    if len(trikotniki) == 0:
        print("POZOR: Datoteka je prazna ali pa je v BINARNEM formatu! Izvozi jo kot ASCII STL.")
        return None

    return np.array(trikotniki), np.array(normale)

def vizualiziraj_stl(trikotniki):
    print(f"-> Uspešno uvoženih {len(trikotniki)} trikotnikov.")
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    poly_collection = Poly3DCollection(trikotniki, facecolors='cyan', linewidths=0.5, edgecolors='black', alpha=0.6)
    ax.add_collection3d(poly_collection)
    
    vse_tocke = trikotniki.reshape(-1, 3)
    min_osi = vse_tocke.min(axis=0)
    max_osi = vse_tocke.max(axis=0)
    ax.set_xlim(min_osi[0], max_osi[0])
    ax.set_ylim(min_osi[1], max_osi[1])
    ax.set_zlim(min_osi[2], max_osi[2])
    ax.set_title("3D Trikotniški model ohišja")
    plt.show()

# PREVERJANJE VODOTESNOSTI
def preveri_vodotesnost(trikotniki):
    robovi_stevec = defaultdict(int)
    for t in trikotniki:
        for i in range(3):
            p1 = tuple(np.round(t[i], 4))
            p2 = tuple(np.round(t[(i + 1) % 3], 4))
            rob = tuple(sorted([p1, p2]))
            robovi_stevec[rob] += 1
            
    vodotesen = True
    napacni_robovi = 0
    for rob, stevec in robovi_stevec.items():
        if stevec != 2:
            vodotesen = False
            napacni_robovi += 1
            
    if vodotesen:
        print("-> Rezultat: Površje modela JE VODOTESNO (vsi robovi pripadajo natanko 2 trikotnikoma).")
    else:
        print(f"-> Rezultat: Površje NI VODOTESNO. Število nepravilnih robov: {napacni_robovi}")
    return vodotesen

# VOKSELIZACIJA POVRŠJA
def konveksna_lupina_2d(tocke):
    tocke = sorted(list(set([tuple(p) for p in tocke])))
    if len(tocke) <= 1: return tocke
    def cross(o, a, b): return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for p in tocke:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0: lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(tocke):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0: upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]

def izracunaj_2d_ploscino(poligon):
    if len(poligon) < 3: return 0
    x = [p[0] for p in poligon]
    y = [p[1] for p in poligon]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def transformiraj_v_2d(tocke_3d, T_trikotnik):
    P = T_trikotnik[0]
    v1 = T_trikotnik[1] - P
    v2 = T_trikotnik[2] - P
    z_os = np.cross(v1, v2)
    norma_z = np.linalg.norm(z_os)
    if norma_z == 0: return np.zeros((len(tocke_3d), 2))
    z_os /= norma_z
    a = np.array([0.0, 0.0, 1.0]) if abs(z_os[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
    x_os = np.cross(a, z_os)
    x_os /= np.linalg.norm(x_os)
    y_os = np.cross(z_os, x_os)
    return np.array([[np.dot(pt - P, x_os), np.dot(pt - P, y_os)] for pt in tocke_3d])

def seka_ravnino_in_rob(A, B, N, D):
    v_smer = B - A
    imenovalec = np.dot(N, v_smer)
    if abs(imenovalec) < 1e-6:
        if abs(np.dot(N, A) + D) < 1e-6: return 2, A, B
        return 0, None, None
    t = -(np.dot(N, A) + D) / imenovalec
    if 0.0 <= t <= 1.0: return 1, A + t * v_smer, None
    return 0, None, None

def vokselizacija_modela(trikotniki, normale, d_voksel=1.0):
    vse_tocke = trikotniki.reshape(-1, 3)
    min_g = vse_tocke.min(axis=0) - d_voksel
    max_g = vse_tocke.max(axis=0) + d_voksel
    nx, ny, nz = np.ceil((max_g - min_g) / d_voksel).astype(int)
    mreza = np.zeros((nx, ny, nz), dtype=int)
    print(f"Mreža generirana z ločljivostjo: {nx}x{ny}x{nz} vokslov.")

    for idx, T in enumerate(trikotniki):
        N = normale[idx]
        norm_N = np.linalg.norm(N)
        if norm_N == 0: continue
        N /= norm_N
        D = -np.dot(N, T[0])
        
        t_min = np.maximum(np.floor((T.min(axis=0) - min_g) / d_voksel).astype(int), 0)
        t_max = np.minimum(np.ceil((T.max(axis=0) - min_g) / d_voksel).astype(int), [nx-1, ny-1, nz-1])
        
        for x in range(t_min[0], t_max[0] + 1):
            for y in range(t_min[1], t_max[1] + 1):
                for z in range(t_min[2], t_max[2] + 1):
                    if mreza[x, y, z] == 1: continue
                    v_min = min_g + np.array([x, y, z]) * d_voksel
                    d = d_voksel
                    v = [v_min + np.array([i, j, k])*d for i in [0,1] for j in [0,1] for k in [0,1]]
                    indeksi_robov = [(0,1), (2,3), (4,5), (6,7), (0,2), (1,3), (4,6), (5,7), (0,4), (1,5), (2,6), (3,7)]
                    
                    p_tocke = []
                    lezi_v_ravnini = False
                    for i1, i2 in indeksi_robov:
                        st_p, p1, p2 = seka_ravnino_in_rob(v[i1], v[i2], N, D)
                        if st_p == 1: p_tocke.append(p1)
                        elif st_p == 2:
                            p_tocke.extend([p1, p2])
                            lezi_v_ravnini = True
                            
                    if len(p_tocke) > 0: p_tocke = np.unique(np.round(p_tocke, 5), axis=0)
                    if len(p_tocke) < 3 or (len(p_tocke) == 4 and lezi_v_ravnini): continue
                        
                    I_2d = transformiraj_v_2d(p_tocke, T)
                    T_2d = transformiraj_v_2d(T, T)
                    I_lupina = konveksna_lupina_2d(I_2d)
                    if len(I_lupina) < 3: continue
                    
                    unija_lupina = konveksna_lupina_2d(list(I_lupina) + list(T_2d))
                    if izracunaj_2d_ploscino(unija_lupina) < (izracunaj_2d_ploscino(I_lupina) + izracunaj_2d_ploscino(T_2d) - 1e-4):
                        mreza[x, y, z] = 1
    return mreza, min_g, d_voksel

# 4.2 VOKSELIZACIJA NOTRANJOSTI (Flood Fill)
def flood_fill_3d(mreza, start_pos, isci_oznako, nova_oznako):
    nx, ny, nz = mreza.shape
    stikalo = [start_pos]
    if mreza[start_pos] != isci_oznako: return
    mreza[start_pos] = nova_oznako
    Premiki = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
    while stikalo:
        cx, cy, cz = stikalo.pop()
        for dx, dy, dz in Premiki:
            x, y, z = cx + dx, cy + dy, cz + dz
            if 0 <= x < nx and 0 <= y < ny and 0 <= z < nz:
                if mreza[x, y, z] == isci_oznako:
                    mreza[x, y, z] = nova_oznako
                    stikalo.append((x, y, z))

def vokselizacija_notranjosti(mreza):
    nx, ny, nz = mreza.shape
    
    # Delamo na kopiji, da ne uničimo prvotne mreže površja
    delovna_mreza = mreza.copy()
    
    # Poplavimo zunanjo okolico iz vseh robnih točk matrike z oznako 2
    # S tem zagotovimo, da je zunanji zrak prepoznan
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                if (x == 0 or x == nx-1 or y == 0 or y == ny-1 or z == 0 or z == nz-1):
                    if delovna_mreza[x, y, z] == 0:
                        flood_fill_3d(delovna_mreza, (x, y, z), 0, 2)
                        
    # Voksli površja (1), ki mejijo na zunanjost (2), postanejo zunanja lupina (3)
    lupina_mreza = delovna_mreza.copy()
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                if delovna_mreza[x, y, z] == 1:
                    meji_na_zunanjost = False
                    for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                        if 0 <= x+dx < nx and 0 <= y+dy < ny and 0 <= z+dz < nz:
                            if delovna_mreza[x+dx, y+dy, z+dz] == 2:
                                meji_na_zunanjost = True
                                break
                    if meji_na_zunanjost:
                        lupina_mreza[x, y, z] = 3

    # Preostali zrak (0), ki ni bil dosežen z zunanjim poplavljanjem, je vrzel (5)
    # Vse kar je ostalo 0, postane 5
    lupina_mreza[lupina_mreza == 0] = 5
    
    # Preostali snovni voksli, ki niso postali zunanja lupina, so notranjost stene (4)
    lupina_mreza[lupina_mreza == 1] = 4
    
    # Rekonstrukcija snovi
    # Snov (1) so vsi voksli, ki so pripadali površju ali notranjosti stene (oznake 3 in 4)
    koncni_model = np.zeros_like(mreza)
    koncni_model[(lupina_mreza == 3) | (lupina_mreza == 4)] = 1
    
    return koncni_model

def vizualiziraj_voksle(mreza):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(mreza, facecolors='crimson', edgecolors='k', linewidth=0.2)
    ax.set_title("Končni vokselski model ohišja")
    plt.show()


# GLAVNI ZAGON
if __name__ == "__main__":
    pot_stl = "ohisje_ports.stl" # Datoteka v ASCII formatu
    
    print("--- 1. Uvoz in vizualizacija STL ---")
    podatki = nalozi_ascii_stl(pot_stl)
    if podatki is not None:
        trikotniki, normale = podatki
        vizualiziraj_stl(trikotniki)
        
        print("\n--- 2. Preverjanje vodotesnosti površja ---")
        preveri_vodotesnost(trikotniki)
        
        # Velikost voksla (mm). 1.5 teče hitro, za lahko zmanjšamo na 1.0 ali 0.8
        velikost_voksla = 1.5 
        
        print(f"\n--- 3. Vokselizacija površja (Ločljivost = {velikost_voksla} mm) ---")
        mreza_p, min_g, d_v = vokselizacija_modela(trikotniki, normale, d_voksel=velikost_voksla)
        
        print("\n--- 4. Vokselizacija notranjosti (Flood Fill) ---")
        koncna_mreza = vokselizacija_notranjosti(mreza_p)
        
        print("\n--- 5. Izračun volumna modela ---")
        snovni_voksli = np.sum(koncna_mreza == 1)
        volumen_voksla = velikost_voksla ** 3
        skupni_volumen = snovni_voksli * volumen_voksla
        print(f"Število snovnih vokslov: {snovni_voksli}")
        print(f"Ocenjen volumen ohišja: {skupni_volumen:.2f} mm³ ({skupni_volumen/1000:.2f} cm³)")
        
        print("\nPrikazujem končni vokselski model...")
        vizualiziraj_voksle(koncna_mreza)