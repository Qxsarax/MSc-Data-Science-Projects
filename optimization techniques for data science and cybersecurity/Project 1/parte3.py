import numpy as np
import matplotlib.pyplot as plt
from metodi import soft_thresholding, ista, fista, subgradiente


# GENERAZIONE DEL PROBLEMA LASSO

rng = np.random.default_rng(41) # seed per riproducibilità

m, n = 80, 160 # dimensioni del problema
s = 12 # numero di componenti non nulle del vettore utilizzato per generare i dati
rumore = 0.02 # rumore gaussiano sulle osservazioni
max_iter = 2500 # numero massimo di iterazioni per i metodi

A = rng.normal(size=(m, n)) / np.sqrt(m) # matrice A con colonne normalizzate in media

x_vero = np.zeros(n) # vettore sparso utilizzato per generare le osservazioni
supporto_vero = rng.choice(n, size=s, replace=False) # 12 posizioni casuali del supporto
x_vero[supporto_vero] = rng.choice([-1.0, 1.0], s) * rng.uniform(1.0, 2.0, s) # moduli casuali tra 1 e 2 con segno casuale

b = A @ x_vero + rumore * rng.normal(size=m) # osservazioni con rumore gaussiano

# sceglie lam pari al 10% di ||A^T b||_inf; se lam è almeno ||A^T b||_inf, x = 0 è una soluzione del LASSO
lam = 0.1 * np.max(np.abs(A.T @ b))

# costante di Lipschitz del gradiente di g(x) = 1/2 ||Ax - b||^2
L = np.linalg.norm(A, 2)**2

x0 = np.zeros(n) # punto iniziale per tutti i metodi


# FUNZIONE OBIETTIVO E OPERATORE PROSSIMALE

# parte liscia della funzione obiettivo
def g(x):
    """Calcola il valore della parte liscia g(x) = 1/2 ||Ax - b||^2"""
    residuo = A @ x - b
    return 0.5 * (residuo @ residuo)

def gradiente_g(x):
    """Calcola il gradiente della parte liscia g(x) = 1/2 ||Ax - b||^2"""
    return A.T @ (A @ x - b)

def F(x):
    """Calcola il valore della funzione obiettivo F(x) = g(x) + lambda * ||x||_1"""
    return g(x) + lam * np.linalg.norm(x, 1)

def prox_h(v, t):
    """Calcola l'operatore prossimale della funzione h(x)"""
    return soft_thresholding(v, lam * t)

print("\nPROBLEMA LASSO")
print("-" * 70)
print(f"Dimensioni A: {A.shape}, non nulli di x_vero: {s}, rumore: {rumore}")
print(f"Lambda = 0.1*||A^T b||_inf = {lam:.6f}")
print(f"L = ||A||_2^2 = {L:.6f}, passo 1/L = {1 / L:.6f}")
print(f"F(x0) = {F(x0):.8f}")

# HESSIANA DELLA PARTE LISCIA

# l'hessiana di g è A^T A; il suo rango è al massimo m, che è minore di n
autovalori = np.linalg.eigvalsh(A.T @ A) # calcola gli autovalori reali dell'hessiana

print("\nHESSIANA DELLA PARTE LISCIA")
print("-" * 70)
print(f"Rango di A^T A: {np.linalg.matrix_rank(A.T @ A)} su {n}")
print(f"Autovalori numericamente nulli: {np.sum(autovalori < 1e-10)}")


# VERIFICA DEL SOFT-THRESHOLDING

# il prossimale si calcola separatamente per ogni componente;
# confronto la formula con il minimo approssimato su una griglia del problema scalare min_u lam*t*|u| + 1/2*(u - v)^2
t_prova = 0.7  # passo usato nella verifica
griglia = np.linspace(-5, 5, 2000001)  # valori candidati per la ricerca del minimo
errore_max_prox = 0.0  # massima differenza osservata tra i due risultati

print("\nVERIFICA DEL SOFT-THRESHOLDING")
print("-" * 70)

for v in [-3.0, -0.5, -0.1, 0.0, 0.1, 0.2, 2.5]:  # valori di v su cui verificare il prossimale

    # cerca sulla griglia il valore che minimizza il problema scalare
    u_forza_bruta = griglia[np.argmin(lam * t_prova * np.abs(griglia) + 0.5 * (griglia - v)**2)]

    # calcola lo stesso valore con la formula del soft-thresholding
    # aggiungiamo 0.0 per evitare di visualizzare -0.0
    u_formula = soft_thresholding(np.array([v]), lam * t_prova)[0] + 0.0

    # aggiorna la massima differenza trovata tra i due risultati
    errore_max_prox = max(errore_max_prox, abs(u_forza_bruta - u_formula))

    print(f"v = {v:>5.2f}   forza bruta = {u_forza_bruta:>8.5f}   soft-thresholding = {u_formula:>8.5f}")

# mostra la soglia usata e la massima differenza tra i risultati
print(f"Soglia lambda*t = {lam * t_prova:.4f}, differenza massima = {errore_max_prox:.1e}")


# ESECUZIONE DI ISTA, FISTA E SUBGRADIENTE

# ISTA e FISTA si fermano quando il gradient mapping scende sotto la tolleranza;
# il subgradiente, invece, viene eseguito per tutte le iterazioni previste
risultati = {
    "ISTA": ista(x0, gradiente_g, prox_h, L, tolleranza=1e-8, max_iter=max_iter),
    "FISTA": fista(x0, gradiente_g, prox_h, L, tolleranza=1e-8, max_iter=max_iter),
    "Subgradiente": subgradiente(x0, gradiente_g, lam, L, max_iter=max_iter)
}

stili = {"ISTA": "tab:blue", "FISTA": "tab:orange", "Subgradiente": "tab:green"} # associa un colore a ciascun metodo per i grafici


# SOLUZIONE DI RIFERIMENTO 

# calcola una soluzione di riferimento con FISTA e una tolleranza molto stretta
x_star = fista(x0, gradiente_g, prox_h, L, tolleranza=1e-13, max_iter=200000)[-1]

# calcola F in tutti i punti visitati dai metodi
valori = {nome: np.array([F(x) for x in traiettoria]) for nome, traiettoria in risultati.items()}

# stima F* con il valore più basso osservato, così i gap calcolati non risultano negativi
F_star = min(F(x_star), *(v.min() for v in valori.values()))

# controlla le condizioni KKT: sui coefficienti non nulli il gradiente
# deve bilanciare la penalità; sugli altri il suo modulo non deve superare lam

grad_star = gradiente_g(x_star) # calcola il gradiente della parte liscia in x_star
supporto_star = np.flatnonzero(x_star)  # indici dei coefficienti non nulli in x_star
fuori_supporto = np.setdiff1d(np.arange(n), supporto_star)  # indici dei coefficienti nulli

residuo_kkt = np.empty(n) # residuo delle condizioni KKT per ciascuna componente
residuo_kkt[supporto_star] = np.abs(grad_star[supporto_star] + lam * np.sign(x_star[supporto_star])) # residuo sulle componenti non nulle
residuo_kkt[fuori_supporto] = np.maximum(np.abs(grad_star[fuori_supporto]) - lam, 0.0) # residuo sulle componenti nulle

print("\nSOLUZIONE DI RIFERIMENTO")
print("-" * 70)
print(f"F* = {F_star:.12f}")
print(f"Residuo KKT massimo: {residuo_kkt.max():.2e}")
print(f"Non nulli di x*: {len(supporto_star)}, supporto di x_vero contenuto in quello di x*: "
      f"{set(supporto_vero) <= set(supporto_star)}")
print(f"Componenti spurie di x*: modulo tra {np.abs(x_star[np.setdiff1d(supporto_star, supporto_vero)]).min():.4f} "
      f"e {np.abs(x_star[np.setdiff1d(supporto_star, supporto_vero)]).max():.4f}")


# TABELLA DI CONFRONTO

print("\nCONFRONTO DEI METODI")
print("=" * 100)
print(f"{'Metodo':<14}{'Iterazioni':<12}{'F(x)-F*':<12}{'Zeri esatti':<13}{'|x_i|>1e-3':<12}"
      f"{'||x-x*||':<12}{'||x-x_vero||/||x_vero||'}")
print("-" * 100)

# per ogni metodo riassume le prestazioni usando l'ultima iterata
for nome, traiettoria in risultati.items():

    x_fin = traiettoria[-1]  # ultima soluzione calcolata dal metodo
    print(f"{nome:<14}{len(traiettoria) - 1:<12}{valori[nome][-1] - F_star:<12.2e}"
          f"{np.sum(x_fin == 0):<13}{np.sum(np.abs(x_fin) > 1e-3):<12}"
          f"{np.linalg.norm(x_fin - x_star):<12.2e}"
          f"{np.linalg.norm(x_fin - x_vero) / np.linalg.norm(x_vero):.4f}")

# mostra quanti coefficienti di riferimento sono esattamente nulli
print(f"Riferimento x*: zeri esatti = {n - len(supporto_star)}")

# trova la prima iterazione in cui il gap scende sotto ciascuna soglia
print("\nPRIMA ITERAZIONE CON F(xk) - F* SOTTO LA SOGLIA")
print("-" * 70)
print(f"{'Soglia':<10}" + "".join(f"{nome:<15}" for nome in risultati))

for soglia in [1e-1, 1e-2, 1e-4, 1e-8, 1e-12]:  # soglie di accuratezza da confrontare
    riga = f"{soglia:<10.0e}"  # prepara la riga con il valore della soglia
    for nome in risultati:
        # individua le iterazioni in cui il gap è sotto la soglia
        sotto = np.flatnonzero(valori[nome] - F_star < soglia)

        # riporta la prima iterazione trovata, oppure "mai" se non viene raggiunta
        riga += f"{(str(sotto[0]) if len(sotto) else 'mai'):<15}"

    print(riga)

# COMPORTAMENTO DEI METODI

print("\nCOMPORTAMENTO DEI METODI")
print("-" * 70)

# conta gli aumenti di F(xk) maggiori della tolleranza numerica
for nome in ["ISTA", "FISTA"]:
    print(f"{nome}: iterazioni in cui F(xk) aumenta = {np.sum(np.diff(valori[nome]) > 1e-14)}")

# salva l'iterazione da cui il supporto coincide e resta uguale nelle iterate registrate
k_supporto = {}

# confronta il supporto di ogni iterata con quello di x*
for nome in ["ISTA", "FISTA"]:
    sbagliate = np.flatnonzero([set(np.flatnonzero(x)) != set(supporto_star) for x in risultati[nome]])
    k_supporto[nome] = sbagliate[-1] + 1 if len(sbagliate) else 0
    print(f"{nome}: supporto di x* identificato definitivamente dall'iterazione {k_supporto[nome]}")

A_S = A[:, supporto_star]  # considera solo le colonne associate al supporto di x*
mu_S = np.linalg.eigvalsh(A_S.T @ A_S).min()  # minima curvatura del problema ristretto, usata nel fattore teorico di ISTA

errori_ista = np.linalg.norm(risultati["ISTA"] - x_star, axis=1)  # distanza di ogni iterata di ISTA da x*
k2 = len(errori_ista) - 1  # indice dell'ultima iterata
k1 = (k_supporto["ISTA"] + k2) // 2  # indice intermedio tra stabilizzazione del supporto e ultima iterata
fattore_misurato = (errori_ista[k2] / errori_ista[k1])**(1 / (k2 - k1))  # fattore medio di riduzione dell'errore tra k1 e k2

print(f"ISTA: fattore teorico 1 - mu_S/L = {1 - mu_S / L:.5f} (mu_S = {mu_S:.4f}), "
      f"misurato tra k = {k1} e k = {k2}: {fattore_misurato:.5f}")

# analizziamo le oscillazioni delle iterate del metodo del subgradiente
# nelle componenti esterne al supporto della soluzione di riferimento
x_sub = risultati["Subgradiente"][-1]
ultime = risultati["Subgradiente"][-101:, fuori_supporto]

print(f"Subgradiente: mediana |x_i| fuori dal supporto = {np.median(np.abs(x_sub[fuori_supporto])):.2e}, "
      f"cambi di segno nelle ultime 100 iterazioni = {np.mean(np.sign(ultime[1:]) != np.sign(ultime[:-1])):.2f}")


# GRAFICO 1: CONVERGENZA DI F(xk) - F*

fig, assi = plt.subplots(1, 2, figsize=(14, 5.5))
for nome, v in valori.items():
    gap = v - F_star # calcola il gap tra F(xk) e F* per ciascun metodo

    #teniamo solo i valori positivi per la scala logaritmica, senza tagli artificiali
    indici = np.flatnonzero(gap > 0)

    assi[0].semilogy(indici, gap[indici], color=stili[nome], linewidth=1.8, label=nome)
    assi[1].loglog(indici[indici > 0], gap[indici[indici > 0]], color=stili[nome], linewidth=1.8, label=nome)

# stime teoriche: ista O(1/k), fista O(1/k^2)
k_teo = np.arange(1, max_iter + 1)
R2 = np.linalg.norm(x0 - x_star)**2

assi[1].loglog(k_teo, L * R2 / (2 * k_teo), "--", color="tab:blue", alpha=0.6,
               label=r"Stima ISTA $L\|x_0-x^*\|^2/(2k)$")
assi[1].loglog(k_teo, 2 * L * R2 / (k_teo + 1)**2, "--", color="tab:orange", alpha=0.6,
               label=r"Stima FISTA $2L\|x_0-x^*\|^2/(k+1)^2$")
assi[0].set_title("Scala semilogaritmica")
assi[1].set_title("Scala bilogaritmica e stime teoriche")
assi[1].set_ylim(1e-16, 1e4)

for ax in assi:
    ax.set_xlabel("Numero di iterazioni")
    ax.set_ylabel(r"$F(x_k) - F^*$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9)

fig.suptitle("LASSO - Confronto della convergenza")
plt.tight_layout()
plt.savefig("grafici/parte3_convergenza.png", dpi=300)
plt.show()

#GRAFICO 2: SPARSITÀ DELLE SOLUZIONI
fig, assi = plt.subplots(2, 2, figsize=(13, 7), sharex=True, sharey=True)
vettori = {
    "Vettore vero $x_{vero}$": (x_vero, "black"),
    "ISTA": (risultati["ISTA"][-1], stili["ISTA"]),
    "FISTA": (risultati["FISTA"][-1], stili["FISTA"]),
    "Subgradiente": (x_sub, stili["Subgradiente"])}

for ax, (nome, (vettore, colore)) in zip(assi.flat, vettori.items()):

    # mostra solo le componenti non nulle per chiarezza
    nonnulle = np.flatnonzero(vettore)
    marker, steli, _ = ax.stem(nonnulle, vettore[nonnulle], basefmt=" ")
    plt.setp(marker, color=colore, markersize=4)
    plt.setp(steli, color=colore, linewidth=1)
    ax.axhline(0, color="gray", linewidth=0.6)

    ax.set_title(f"{nome}: {len(nonnulle)} componenti non nulle su {n}")
    ax.grid(alpha=0.25)

for ax in assi[1]:
    ax.set_xlabel("Indice i")
for ax in assi[:, 0]:
    ax.set_ylabel(r"$x_i$")

fig.suptitle("LASSO - Sparsità delle soluzioni")
plt.tight_layout()
plt.savefig("grafici/parte3_sparsita.png", dpi=300)
plt.show()