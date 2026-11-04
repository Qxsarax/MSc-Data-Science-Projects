import numpy as np
import matplotlib.pyplot as plt
from metodi import discesa_gradiente, newton, bfgs

# 1-FUNZIONE DI ROSENBROCK, GRADIENTE E HESSIANA

# definiamo il punto da cui parte l'ottimizzazione e il minimo della funzione
x0 = np.array([-1.2, 1.0])
x_star = np.array([1.0, 1.0])


# Funzione di Rosenbrock
def f(x):
    x1, x2 = x
    return 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2


# Gradiente
def gradiente(x):
    x1, x2 = x

    df_dx1 = -400 * x1 * (x2 - x1 ** 2) + 2 * (x1 - 1)
    df_dx2 = 200 * (x2 - x1 ** 2)

    return np.array([df_dx1, df_dx2])


# Matrice Hessiana
def hessiana(x):
    x1, x2 = x

    return np.array([
        [1200 * x1 ** 2 - 400 * x2 + 2, -400 * x1],
        [-400 * x1, 200]
    ])


# Verifica nel punto iniziale
print("\nPUNTO 1 - FUNZIONE DI ROSENBROCK")
print("-" * 70)

print("Punto iniziale:", x0)
print("Minimo globale:", x_star)
print("Valore iniziale della funzione:", f(x0))
print("Gradiente iniziale:", gradiente(x0))
print("Matrice Hessiana iniziale:")
print(hessiana(x0))


# FUNZIONE AUSILIARIA - ESECUZIONE E CONTEGGIO VALUTAZIONI

# questa funzione esegue il metodo scelto e raccoglie i risultati
def esegui_metodo(metodo, tolleranza=1e-8):
    contatori = {
        "gradiente": 0,
        "hessiana": 0
    }

    # Conteggio delle valutazioni del gradiente
    def grad_contato(x):
        contatori["gradiente"] += 1
        return gradiente(x)

    # Conteggio delle valutazioni della Hessiana
    def hess_contata(x):
        contatori["hessiana"] += 1
        return hessiana(x)

    # Discesa del gradiente con Armijo
    if metodo == "gradiente":

        traiettoria = discesa_gradiente(
            x0,
            f,
            grad_contato,
            metodo="armijo",
            tolleranza=tolleranza,
            max_iter=100000
        )

    # Newton smorzato con Armijo
    elif metodo == "newton":

        traiettoria = newton(
            x0,
            f,
            grad_contato,
            hess_contata,
            tolleranza=tolleranza,
            max_iter=1000
        )

    # Metodo quasi-Newton BFGS
    elif metodo == "bfgs":

        traiettoria = bfgs(
            x0,
            f,
            grad_contato,
            tolleranza=tolleranza,
            max_iter=1000
        )

    else:
        raise ValueError("Metodo non riconosciuto")

    # Soluzione finale e verifica della convergenza
    # prendiamo l'ultimo punto della traiettoria e controlliamo la norma del gradiente
    soluzione = traiettoria[-1]
    norma_gradiente = np.linalg.norm(gradiente(soluzione))

    return {
        "traiettoria": traiettoria,
        "iterazioni": len(traiettoria) - 1,
        "val_gradiente": contatori["gradiente"],
        "val_hessiana": contatori["hessiana"],
        "convergente": norma_gradiente < tolleranza
    }


# 2-METODO DI NEWTON SMORZATO CON ARMIJO

# eseguiamo il metodo di newton con la ricerca del passo tramite armijo
risultato_newton = esegui_metodo("newton")

# recuperiamo la soluzione finale e calcoliamo la distanza dal minimo globale
soluzione_newton = risultato_newton["traiettoria"][-1]
errore_newton = np.linalg.norm(soluzione_newton - x_star)

print("\nPUNTO 2 - METODO DI NEWTON CON ARMIJO")
print("-" * 70)

print("Punto iniziale:", x0)
print("Soluzione trovata:", soluzione_newton)
print(f"Valore finale della funzione: {f(soluzione_newton):.10e}")
print(f"Numero di iterazioni: {risultato_newton['iterazioni']}")
print(f"Valutazioni del gradiente: {risultato_newton['val_gradiente']}")
print(f"Valutazioni della Hessiana: {risultato_newton['val_hessiana']}")
print(f"Norma finale del gradiente: {np.linalg.norm(gradiente(soluzione_newton)):.10e}")
print(f"Errore rispetto al minimo: {errore_newton:.10e}")
print("Convergenza:", risultato_newton["convergente"])

# 3-METODO QUASI-NEWTON BFGS

# eseguiamo il metodo bfgs per trovare il minimo della funzione di Rosenbrock
risultato_bfgs = esegui_metodo("bfgs")

# recuperiamo la soluzione finale e calcoliamo la distanza dal minimo globale
soluzione_bfgs = risultato_bfgs["traiettoria"][-1]
errore_bfgs = np.linalg.norm(soluzione_bfgs - x_star)

print("\nPUNTO 3 - METODO QUASI-NEWTON BFGS")
print("-" * 70)

print("Punto iniziale:", x0)
print("Soluzione trovata:", soluzione_bfgs)
print(f"Valore finale della funzione: {f(soluzione_bfgs):.10e}")
print(f"Numero di iterazioni: {risultato_bfgs['iterazioni']}")
print(f"Valutazioni del gradiente: {risultato_bfgs['val_gradiente']}")
print(f"Valutazioni della Hessiana: {risultato_bfgs['val_hessiana']}")
print(f"Norma finale del gradiente: {np.linalg.norm(gradiente(soluzione_bfgs)):.10e}")
print(f"Errore rispetto al minimo: {errore_bfgs:.10e}")
print("Convergenza:", risultato_bfgs["convergente"])

# 4-CONFRONTO DEI TRE METODI

# Esecuzione della discesa del gradiente con Armijo
risultato_gradiente = esegui_metodo("gradiente")

# Raccolta dei risultati
# riuniamo i risultati dei tre metodi in un unico dizionario
risultati = {
    "Gradiente (Armijo)": risultato_gradiente,
    "Newton (Armijo)": risultato_newton,
    "BFGS": risultato_bfgs
}

# Stili dei grafici
stili = {
    "Gradiente (Armijo)": ("tab:blue", "o"),
    "Newton (Armijo)": ("tab:orange", "s"),
    "BFGS": ("tab:green", "^")
}

# TABELLA DI CONFRONTO

print("\nPUNTO 4 - CONFRONTO DEI TRE METODI")
print("=" * 100)

print(
    f"{'Metodo':<22}"
    f"{'Iterazioni':<14}"
    f"{'Gradiente':<14}"
    f"{'Hessiana':<14}"
    f"{'Convergenza':<14}"
    f"{'Errore finale':<16}"
)

print("-" * 100)

for nome, risultato in risultati.items():
    traiettoria = risultato["traiettoria"]
    errore_finale = np.linalg.norm(traiettoria[-1] - x_star)

    stato = "Si" if risultato["convergente"] else "No"

    print(
        f"{nome:<22}"
        f"{risultato['iterazioni']:<14}"
        f"{risultato['val_gradiente']:<14}"
        f"{risultato['val_hessiana']:<14}"
        f"{stato:<14}"
        f"{errore_finale:<16.2e}"
    )

# ANDAMENTO DEGLI ERRORI

print("\nANDAMENTO DELL'ERRORE")
print("=" * 70)

# analizziamo l'andamento dell'errore per ciascuno dei tre metodi
for nome, risultato in risultati.items():

    # recuperiamo tutti i punti della traiettoria
    traiettoria = risultato["traiettoria"]

    # calcoliamo la distanza di ogni iterata dal minimo globale
    errori = np.linalg.norm(traiettoria - x_star, axis=1)

    print(f"\nMetodo: {nome}")
    print("-" * 55)
    print(f"Errore iniziale: {errori[0]:.10e}")
    print(f"Errore finale: {errori[-1]:.10e}")

    print("\nPrime 5 iterazioni:")

    # mostriamo gli errori delle prime cinque iterate
    for k in range(min(5, len(errori))):
        print(f"k = {k:<5} Errore = {errori[k]:.10e}")

    print("\nUltime 5 iterazioni:")

    # individuiamo da quale iterazione iniziare a mostrare gli ultimi errori
    inizio = max(5, len(errori) - 5)

    # mostriamo gli errori delle ultime cinque iterate
    for k in range(inizio, len(errori)):
        print(f"k = {k:<5} Errore = {errori[k]:.10e}")

# CURVE DI LIVELLO E TRAIETTORIE

# Griglia
x1_vals = np.linspace(-1.6, 1.5, 450)
x2_vals = np.linspace(-0.5, 2.0, 450)

# creiamo una griglia di punti nel piano
X1, X2 = np.meshgrid(x1_vals, x2_vals)

# Funzione Rosenbrock sulla griglia
# calcoliamo il valore della funzione di rosenbrock in ogni punto della griglia
Z = 100 * (X2 - X1 ** 2) ** 2 + (1 - X1) ** 2


# Funzione per disegnare le traiettorie
def disegna_traiettoria(traiettoria, nome, colore, marker):
    # scegliamo ogni quanti punti mostrare i marker per non affollare troppo il grafico
    distanza_marker = max(1, len(traiettoria) // 50)

    # disegniamo la traiettoria con colore, simbolo e numero di iterazioni
    plt.plot(
        traiettoria[:, 0],
        traiettoria[:, 1],
        color=colore,
        linewidth=1.5,
        marker=marker,
        markevery=distanza_marker,
        markersize=4,
        alpha=0.85,
        label=f"{nome} ({len(traiettoria) - 1} iterazioni)",
        zorder=3
    )


# impostiamo la figura su cui verranno disegnate curve di livello e traiettorie
plt.figure(figsize=(11, 7))

# scegliamo livelli logaritmici per visualizzare meglio la forma della funzione
livelli = np.logspace(-2, 3.5, 24)

# Curve di livello
contorni = plt.contour(
    X1, X2, Z,
    levels=livelli,
    colors="gray",
    linewidths=0.8,
    alpha=0.65
)

# aggiungiamo le etichette numeriche alle curve di livello
plt.clabel(contorni, inline=True, fontsize=7, fmt="%.2g")

# Traiettorie dei tre metodi
for nome, risultato in risultati.items():
    traiettoria = risultato["traiettoria"]
    colore, marker = stili[nome]

    disegna_traiettoria(traiettoria, nome, colore, marker)

# Punto iniziale
plt.scatter(
    x0[0], x0[1],
    color="black",
    marker="X",
    s=120,
    label="Punto iniziale",
    zorder=5
)

# Minimo globale
plt.scatter(
    x_star[0], x_star[1],
    color="red",
    marker="*",
    s=180,
    label="Minimo",
    zorder=5
)

# aggiungiamo etichette, titolo e limiti degli assi
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("Rosenbrock - Curve di livello e traiettorie")

plt.xlim(-1.6, 1.5)
plt.ylim(-0.5, 2.0)

# completiamo il grafico con legenda, griglia e salvataggio finale
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()

plt.savefig("grafici/parte2_traiettorie.png", dpi=300)
plt.show()

# GRAFICO SEMILOGARITMICO DELL'ERRORE

# creiamo una figura con due grafici affiancati
fig, assi = plt.subplots(1, 2, figsize=(14, 5.5))

# rappresentiamo l'errore per ciascuno dei tre metodi
for nome, risultato in risultati.items():

    traiettoria = risultato["traiettoria"]

    # Errore per ciascuna iterazione
    errori = np.linalg.norm(traiettoria - x_star, axis=1)

    # Escludiamo errori nulli dalla scala logaritmica
    # teniamo solo gli errori positivi, che possono essere mostrati in scala logaritmica
    indici = np.where(errori > 0)[0]
    colore, marker = stili[nome]

    # Stessa curva su entrambi i grafici
    for ax in assi:
        ax.semilogy(
            indici,
            errori[indici],
            color=colore,
            linewidth=1.8,
            label=nome
        )

# Grafico completo
assi[0].set_title("Confronto completo")

# Zoom sulle prime 45 iterazioni
assi[1].set_title("Zoom: prime 45 iterazioni")
assi[1].set_xlim(0, 45)

# aggiungiamo etichette, griglia e legenda a entrambi i grafici
for ax in assi:
    ax.set_xlabel("Numero di iterazioni")
    ax.set_ylabel("Errore ||xk - x*||")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

# inseriamo un titolo generale e salviamo la figura finale
fig.suptitle("Confronto della convergenza - Rosenbrock")

plt.tight_layout()
plt.savefig("grafici/parte2_convergenza.png", dpi=300)
plt.show()
