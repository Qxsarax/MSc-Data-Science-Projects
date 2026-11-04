import numpy as np
import matplotlib.pyplot as plt

# 1-DEFINIZIONE DEL PROBLEMA

# impostiamo il valore di kappa, che determina il condizionamento della matrice
kappa = 10

# costruiamo la matrice diagonale A utilizzata nella funzione quadratica
A = np.array([
    [1, 0],
    [0, kappa]
])

# scegliamo il punto iniziale da cui parte la discesa del gradiente
x0 = np.array([float(kappa), 1.0])


# FUNZIONE OBIETTIVO E GRADIENTE

# definiamo la funzione quadratica che vogliamo minimizzare
def f(x):
    return 0.5 * x @ A @ x


# calcoliamo il gradiente della funzione tramite il prodotto tra A e x
def gradiente(x):
    return A @ x


# BACKTRACKING DI ARMIJO

# funzione che cerca un passo adatto per la discesa del gradiente
def armijo(x, alpha_iniziale=1.0, beta=0.5, c=1e-4):
    # impostiamo il passo iniziale e calcoliamo il gradiente
    alpha = alpha_iniziale
    grad = gradiente(x)

    # riduciamo il passo finché non viene soddisfatta la condizione di armijo
    while f(x - alpha * grad) > f(x) - c * alpha * np.linalg.norm(grad) ** 2:
        alpha = beta * alpha

    # restituiamo il valore del passo trovato
    return alpha


# 2-DISCESA DEL GRADIENTE

# funzione che esegue la discesa del gradiente con il metodo scelto
def discesa_gradiente(x0, metodo, tolleranza=1e-8, max_iter=100000):
    # partiamo da una copia del punto iniziale e lo salviamo nella traiettoria
    x = x0.copy()
    traiettoria = [x.copy()]

    # calcoliamo gli autovalori della matrice A
    autovalori = np.linalg.eigvalsh(A)

    # ricaviamo il valore minimo e massimo degli autovalori
    mu = np.min(autovalori)
    L = np.max(autovalori)

    # ripetiamo gli aggiornamenti fino al limite massimo di iterazioni
    for k in range(max_iter):

        # calcoliamo il gradiente nel punto corrente
        grad = gradiente(x)

        # ci fermiamo quando la norma del gradiente è inferiore alla tolleranza
        if np.linalg.norm(grad) < tolleranza:
            break

        # scegliamo il passo in base al metodo utilizzato
        if metodo == "costante":
            alpha = 1 / L

        elif metodo == "ottimo":
            alpha = 2 / (L + mu)

        elif metodo == "armijo":
            alpha = armijo(x)

        # se il metodo non è riconosciuto, segnaliamo un errore
        else:
            raise ValueError("Metodo non riconosciuto")

        # aggiorniamo il punto muovendoci nella direzione opposta al gradiente
        x = x - alpha * grad

        # salviamo il nuovo punto per ricostruire la traiettoria
        traiettoria.append(x.copy())

    # restituiamo tutti i punti visitati durante la discesa
    return np.array(traiettoria)


# ESECUZIONE DEI TRE METODI

# associamo a ogni metodo il nome utilizzato nel codice e un simbolo per il grafico
metodi = {
    "Passo costante": ("costante", "o"),
    "Passo ottimo": ("ottimo", "s"),
    "Armijo": ("armijo", "^")
}

# creiamo un dizionario in cui salvare i risultati dei tre metodi
risultati = {}

# eseguiamo un metodo alla volta
for nome, (metodo, marker) in metodi.items():
    # calcoliamo la traiettoria partendo dallo stesso punto iniziale
    traiettoria = discesa_gradiente(x0, metodo)

    # salviamo la traiettoria e il simbolo da utilizzare nel grafico
    risultati[nome] = (traiettoria, marker)

    print(f"{nome}: {len(traiettoria) - 1} iterazioni")

# CREAZIONE DELLE CURVE DI LIVELLO

# definiamo gli intervalli di valori per le due variabili
x1_vals = np.linspace(-12, 12, 400)
x2_vals = np.linspace(-5, 5, 400)

# creiamo una griglia di punti su cui calcolare la funzione
X1, X2 = np.meshgrid(x1_vals, x2_vals)

# calcoliamo il valore della funzione quadratica in ogni punto della griglia
Z = 0.5 * (X1 ** 2 + kappa * X2 ** 2)


# FUNZIONE PER DISEGNARE LE TRAIETTORIE

# definiamo una funzione per rappresentare il percorso seguito da ogni metodo
def disegna_traiettoria(traiettoria, nome, colore, marker):
    # calcoliamo il numero di iterazioni escludendo il punto iniziale
    n_iterazioni = len(traiettoria) - 1

    # disegniamo la traiettoria indicando colore, simbolo e numero di iterazioni
    plt.plot(
        traiettoria[:, 0],
        traiettoria[:, 1],
        color=colore,
        linewidth=1.5,
        marker=marker,
        markersize=4,
        alpha=0.85,
        label=f"{nome} ({n_iterazioni} iterazioni)",
        zorder=3
    )


# CREAZIONE DELLA GRIGLIA

# creiamo la griglia utilizzando i valori delle due variabili
X1, X2 = np.meshgrid(x1_vals, x2_vals)

# CREAZIONE DEL GRAFICO

# impostiamo le dimensioni della figura
plt.figure(figsize=(11, 6))

# curve di livello meno affollate
# scegliamo alcuni livelli per rendere il grafico più leggibile
livelli = [1, 5, 10, 20, 40, 80]

# disegniamo le curve di livello della funzione usando il colore grigio
contorni = plt.contour(
    X1, X2, Z,
    levels=livelli,
    colors="gray",
    linewidths=1,
    alpha=0.6
)

plt.clabel(contorni, inline=True, fontsize=8)

# stili delle traiettorie
# assegniamo a ogni metodo un colore e un simbolo diverso
stili = {
    "Passo costante": ("tab:blue", "o"),
    "Passo ottimo": ("tab:orange", "s"),
    "Armijo": ("tab:green", "^")
}

# disegniamo le traiettorie dei tre metodi utilizzando gli stili scelti
for nome, (traiettoria, marker) in risultati.items():
    colore, simbolo = stili[nome]
    disegna_traiettoria(traiettoria, nome, colore, simbolo)

# punto iniziale
# evidenziamo il punto da cui partono tutti i metodi
plt.scatter(
    x0[0], x0[1],
    color="black",
    marker="X",
    s=120,
    label="Punto iniziale"
)

# minimo
# indichiamo con una stella rossa il minimo della funzione
plt.scatter(
    0, 0,
    color="red",
    marker="*",
    s=180,
    label="Minimo"
)

# aggiungiamo le etichette degli assi e il titolo del grafico
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Curve di livello e traiettorie - Discesa del gradiente")

# impostiamo gli intervalli degli assi per visualizzare meglio le traiettorie
plt.xlim(-1, 11)
plt.ylim(-2, 2)

# aggiungiamo la legenda e la griglia e sistemiamo la disposizione del grafico
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

# 3-CONFRONTO PER DIVERSI VALORI DI KAPPA

# definiamo i valori di kappa da utilizzare negli esperimenti
valori_kappa = [1, 10, 100, 1000]

# creiamo un dizionario per raccogliere i risultati dei vari esperimenti
risultati_punto3 = {}

# ripetiamo il confronto per ogni valore di kappa
for kappa in valori_kappa:

    # aggiorniamo la matrice A in base al valore di kappa
    A = np.array([
        [1, 0],
        [0, kappa]
    ])

    # impostiamo il punto iniziale per il nuovo esperimento
    x0 = np.array([float(kappa), 1.0])

    # creiamo uno spazio per salvare i risultati relativi a questo kappa
    risultati_punto3[kappa] = {}

    # eseguiamo tutti e tre i metodi con la matrice aggiornata
    for nome, (metodo, marker) in metodi.items():
        # calcoliamo la traiettoria del metodo
        traiettoria = discesa_gradiente(x0, metodo)

        # contiamo le iterazioni escludendo il punto iniziale
        n_iterazioni = len(traiettoria) - 1

        # calcoliamo la distanza dal minimo per ogni punto della traiettoria
        errori = np.linalg.norm(traiettoria, axis=1)

        # verifichiamo se il gradiente finale soddisfa la tolleranza richiesta
        convergente = (
                np.linalg.norm(gradiente(traiettoria[-1])) < 1e-8
        )

        # salviamo iterazioni, errori e risultato del controllo di convergenza
        risultati_punto3[kappa][nome] = {
            "iterazioni": n_iterazioni,
            "errori": errori,
            "convergente": convergente
        }

# TABELLA DEL NUMERO DI ITERAZIONI


print("\nCONFRONTO DEL NUMERO DI ITERAZIONI")
print("-" * 60)

print(f"{'Kappa':<10} {'Costante':<15} {'Ottimo':<15} {'Armijo':<15}")

# analizziamo i risultati per ogni valore di kappa
for kappa in valori_kappa:

    # creiamo una lista per raccogliere le iterazioni dei tre metodi
    valori = []

    # recuperiamo i risultati di ciascun metodo
    for nome in metodi:

        risultato = risultati_punto3[kappa][nome]

        # se il metodo è convergente, salviamo il numero di iterazioni
        if risultato["convergente"]:
            valori.append(str(risultato["iterazioni"]))
        else:
            # altrimenti indichiamo che non ha raggiunto la convergenza
            valori.append("Non conv.")

    print(
        f"{kappa:<10} "
        f"{valori[0]:<15} "
        f"{valori[1]:<15} "
        f"{valori[2]:<15}")

# GRAFICI SEMILOGARITMICI DELL'ERRORE


fig, assi = plt.subplots(2, 2, figsize=(12, 9))

# associamo a ogni grafico il corrispondente valore di kappa
for ax, kappa in zip(assi.flat, valori_kappa):

    # impostiamo una scala logaritmica per rappresentare gli errori
    ax.set_yscale("log")

    # gestiamo separatamente il caso kappa uguale a 1
    if kappa == 1:

        # recuperiamo l'errore iniziale del passo costante
        errore_iniziale = risultati_punto3[kappa]["Passo costante"]["errori"][0]

        # evidenziamo il valore iniziale dell'errore
        ax.scatter(
            0, errore_iniziale,
            color="black",
            s=60,
            label="Errore iniziale"
        )

        # aggiungiamo un testo per indicare che tutti i metodi convergono subito
        ax.text(
            0.5, 0.5,
            "Tutti i metodi convergono\nin 1 iterazione",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11
        )

        # impostiamo i limiti degli assi per visualizzare meglio questo caso
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(0.1, 10)

    else:

        # disegniamo l'andamento dell'errore per ciascun metodo
        for nome in metodi:
            # recuperiamo gli errori salvati durante le iterazioni
            errori = risultati_punto3[kappa][nome]["errori"]

            # consideriamo solo gli errori positivi, necessari per la scala logaritmica
            indici = np.where(errori > 0)[0]

            # recuperiamo il colore e il simbolo associati al metodo
            colore, marker = stili[nome]

            # rappresentiamo l'errore in funzione del numero di iterazioni
            ax.plot(
                indici,
                errori[indici],
                label=nome,
                color=colore,
                linewidth=1.8)

    # aggiungiamo titolo ed etichette degli assi a ogni grafico
    ax.set_title(f"Kappa = {kappa}")
    ax.set_xlabel("Numero di iterazioni")
    ax.set_ylabel("Errore ||xk - x*||")

    # aggiungiamo la griglia e la legenda
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

# inseriamo un titolo generale per il confronto
fig.suptitle(
    "Confronto della convergenza dei tre metodi",
    fontsize=14
)

# sistemiamo la disposizione dei grafici e salviamo l'immagine nella cartella grafici
plt.tight_layout()
plt.savefig("grafici/parte1_confronto_convergenza.png", dpi=300)
plt.show()


# 4-STIMA TEORICA DELLA CONVERGENZA

# funzione che calcola la stima teorica dell'errore per il passo ottimo
def stima_teorica(kappa, x0, n_iterazioni):
    # calcoliamo il fattore di convergenza in base al valore di kappa
    q = (kappa - 1) / (kappa + 1)

    # calcoliamo la distanza iniziale dal minimo
    errore_iniziale = np.linalg.norm(x0)

    # creiamo gli indici delle iterazioni, partendo da zero
    k = np.arange(n_iterazioni + 1)

    # calcoliamo la stima teorica dell'errore a ogni iterazione
    stima = (q ** k) * errore_iniziale

    # restituiamo i valori ottenuti
    return stima


# FUNZIONE PER CONFRONTARE TEORIA E RISULTATI

# funzione che confronta gli errori sperimentali con la stima teorica
def confronta_convergenza(kappa, ax):
    # recuperiamo gli errori ottenuti con il passo ottimo
    errori = risultati_punto3[kappa]["Passo ottimo"]["errori"]

    # ricostruiamo il punto iniziale relativo al valore di kappa
    x_iniziale = np.array([float(kappa), 1.0])

    # calcoliamo il numero di iterazioni effettuate
    n_iterazioni = len(errori) - 1

    # calcoliamo la stima teorica usando la funzione definita prima
    stima = stima_teorica(kappa, x_iniziale, n_iterazioni)

    # creiamo gli indici delle iterazioni, compreso quello iniziale
    iterazioni = np.arange(n_iterazioni + 1)

    # consideriamo solo i valori positivi per poter usare la scala logaritmica
    validi = (errori > 0) & (stima > 0)

    # disegniamo l'andamento degli errori sperimentali
    ax.semilogy(
        iterazioni[validi],
        errori[validi],
        label="Errore sperimentale",
        linewidth=2
    )

    # aggiungiamo la stima teorica con una linea tratteggiata
    ax.semilogy(
        iterazioni[validi],
        stima[validi],
        "--",
        label="Stima teorica",
        linewidth=2
    )

    # aggiungiamo titolo, etichette degli assi, griglia e legenda
    ax.set_title(f"Kappa = {kappa}")
    ax.set_xlabel("Numero di iterazioni")
    ax.set_ylabel("Errore")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()


# GRAFICO DEL CONFRONTO TEORICO-SPERIMENTALE

# creiamo quattro grafici, uno per ogni valore di kappa
fig, assi = plt.subplots(2, 2, figsize=(12, 9))

# ripetiamo il confronto per ciascun valore di kappa
for ax, kappa in zip(assi.flat, valori_kappa):

    # gestiamo separatamente il caso kappa uguale a 1
    if kappa == 1:

        # inseriamo un testo perché la convergenza avviene in una sola iterazione
        ax.text(
            0.5, 0.5,
            "Convergenza esatta\nin 1 iterazione",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11
        )

        # aggiungiamo il titolo e le etichette degli assi
        ax.set_title("Kappa = 1")
        ax.set_xlabel("Numero di iterazioni")
        ax.set_ylabel("Errore")

    else:
        # per gli altri valori confrontiamo gli errori sperimentali con la stima teorica
        confronta_convergenza(kappa, ax)

# aggiungiamo un titolo generale alla figura
fig.suptitle(
    "Confronto tra convergenza teorica e sperimentale",
    fontsize=14,
    fontweight="bold"
)

# sistemiamo la disposizione dei grafici e salviamo l'immagine nella cartella grafici
plt.tight_layout()
plt.savefig("grafici/parte1_confronto_teorico_sperimentale.png", dpi=300)
plt.show()


# TABELLA DEL CONFRONTO TEORICO-SPERIMENTALE

# funzione che stampa il confronto tra gli errori ottenuti e quelli previsti dalla teoria
def stampa_confronto_teorico(risultati_punto3, valori_kappa):
    print("\nCONFRONTO TRA ERRORE SPERIMENTALE E STIMA TEORICA")
    print("-" * 85)

    print(
        f"{'Kappa':<10}"
        f"{'Iterazioni':<15}"
        f"{'Errore finale':<20}"
        f"{'Stima teorica':<20}"
        f"{'Differenza max':<20}"
    )

    # ripetiamo il confronto per ogni valore di kappa
    for kappa in valori_kappa:
        # recuperiamo gli errori sperimentali ottenuti con il passo ottimo
        dati = risultati_punto3[kappa]["Passo ottimo"]
        errori = dati["errori"]

        # ricostruiamo il punto iniziale relativo al valore di kappa
        x_iniziale = np.array([float(kappa), 1.0])

        # recuperiamo il numero di iterazioni effettuate
        n_iterazioni = dati["iterazioni"]

        # calcoliamo la stima teorica per tutte le iterazioni
        stima = stima_teorica(kappa, x_iniziale, n_iterazioni)

        # troviamo la differenza massima tra gli errori sperimentali e quelli teorici
        differenza_max = np.max(np.abs(errori - stima))

        print(
            f"{kappa:<10}"
            f"{n_iterazioni:<15}"
            f"{errori[-1]:<20.6e}"
            f"{stima[-1]:<20.6e}"
            f"{differenza_max:<20.6e}")


# ESECUZIONE DELLA FUNZIONE

stampa_confronto_teorico(risultati_punto3, valori_kappa)
