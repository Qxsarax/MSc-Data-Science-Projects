"""
Attività di laboratorio 2
Addestramento e confronto di modelli di apprendimento automatico
-
Esercizio 1: Effetto dell'ottimizzatore sulla convergenza
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def genera_lune(N=1000, rumore=0.2, seed=0):
    """Due 'mezze lune' intrecciate: classe 0 (lecito) e classe 1 (malevolo)."""
    rng = np.random.default_rng(seed)
    n = N // 2
    t1, t2 = rng.uniform(0, np.pi, n), rng.uniform(0, np.pi, n)
    X0 = np.c_[np.cos(t1), np.sin(t1)]
    X1 = np.c_[1 - np.cos(t2), 0.5 - np.sin(t2)]
    X = np.vstack([X0, X1]) + rumore * rng.normal(size=(N, 2))
    y = np.concatenate([np.zeros(n), np.ones(n)])
    p = rng.permutation(N)
    return X[p], y[p]

def dividi_standardizza(X, y, frazione_val=0.2, seed=0):
    """Dividiamo il dataset in training e validation e standardizziamo le caratteristiche."""
    rng = np.random.default_rng(seed)
    p = rng.permutation(len(y))
    n_va = int(round(frazione_val * len(y)))
    va, tr = p[:n_va], p[n_va:]
    mu, sd = X[tr].mean(0), X[tr].std(0)
    return (X[tr] - mu) / sd, y[tr], (X[va] - mu) / sd, y[va]


# Esercizio 1a. Retropropagazione e verifica del gradiente

def inizializza_rete(seed=42, dimensioni=(2, 16, 16, 1)):
    """Inizializza i parametri della rete neurale"""
    n0, n1, n2, n3 = dimensioni # numero di neuroni per ciascuno strato
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0, np.sqrt(2 / n0), size=(n0, n1)),
        "b1": np.zeros((1, n1)),
        "W2": rng.normal(0, np.sqrt(2 / n1), size=(n1, n2)),
        "b2": np.zeros((1, n2)),
        "W3": rng.normal(0, np.sqrt(2 / n2), size=(n2, n3)),
        "b3": np.zeros((1, n3)),
    } # pesi con inizializzazione di He e bias a zero


def relu(z):
    """La funzione ReLU restituisce il valore di input se è positivo, altrimenti restituisce zero. """
    return np.maximum(0, z)

def sigmoid(z):
    """"La funzione sigmoide restituisce un valore compreso tra 0 e 1"""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def forward(X, parametri):
    """Calcola l'output della rete neurale e salva i valori intermedi"""

    # 1° strato nascosto con attivazione relu
    Z1 = X @ parametri["W1"] + parametri["b1"]
    A1 = relu(Z1)

    # 2° strato nascosto con attivazione relu
    Z2 = A1 @ parametri["W2"] + parametri["b2"]
    A2 = relu(Z2)

    # strato di output con attivazione sigmoide
    Z3 = A2 @ parametri["W3"] + parametri["b3"]

    probabilita = sigmoid(Z3)
    cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2,
             "A2": A2, "Z3": Z3, "probabilita": probabilita}
    return probabilita, cache

def binary_cross_entropy(y, probabilita):
    """Calcola la binary cross-entropy loss"""
    y = np.asarray(y).reshape(-1, 1) # converte le etichette in un vettore colonna
    p = np.clip(probabilita, 1e-12, 1 - 1e-12) # evitiamo log(0)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)) # calcola la perdita media sulle osservazioni

def backpropagation(y, parametri, cache):
    """Calcola i gradienti della loss rispetto ai parametri della rete"""

    # recupera i valori salvati durante il forward
    X, Z1, A1 = cache["X"], cache["Z1"], cache["A1"]
    Z2, A2, p = cache["Z2"], cache["A2"], cache["probabilita"]
    y = np.asarray(y).reshape(-1, 1)

    # calcola i gradienti dello strato di output
    dZ3 = (p - y) / len(X) # derivata della loss binaria rispetto all'output della rete
    dW3 = A2.T @ dZ3
    db3 = np.sum(dZ3, axis=0, keepdims=True)

    # calcola i gradienti del secondo strato nascosto
    dZ2 = (dZ3 @ parametri["W3"].T) * (Z2 > 0)
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    # calcola i gradienti del primo strato nascosto
    dZ1 = (dZ2 @ parametri["W2"].T) * (Z1 > 0)
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2,
            "W3": dW3, "b3": db3}

def verifica_gradiente(X, y, parametri, h=1e-5):
    """Verifica il gradiente calcolato con la backpropagation confrontandolo con il gradiente numerico"""
    # copia i parametri e calcola i gradienti con la backpropagation
    pesi = {nome: valore.copy() for nome, valore in parametri.items()}
    _, cache = forward(X, pesi)
    gradienti_bp = backpropagation(y, pesi, cache)
    gradienti_num = {nome: np.zeros_like(p) for nome, p in pesi.items()}
    numero_forward = 0

    for nome, matrice in pesi.items():
        for indice in np.ndindex(matrice.shape):
            originale = matrice[indice]

            # calcola la loss aumentando di poco il parametro
            matrice[indice] = originale + h
            p_piu, _ = forward(X, pesi)
            loss_piu = binary_cross_entropy(y, p_piu)

            # calcola la loss diminuendo di poco il parametro
            matrice[indice] = originale - h
            p_meno, _ = forward(X, pesi)
            loss_meno = binary_cross_entropy(y, p_meno)

            # ripristina il parametro e calcola il gradiente numerico
            matrice[indice] = originale
            gradienti_num[nome][indice] = (loss_piu - loss_meno) / (2 * h)
            numero_forward += 2

     # raccoglie i gradienti in due vettori e calcola l'errore relativo
    g_bp = np.concatenate([g.ravel() for g in gradienti_bp.values()])
    g_num = np.concatenate([g.ravel() for g in gradienti_num.values()])
    errore = np.linalg.norm(g_bp - g_num) / max(
        np.linalg.norm(g_bp), np.linalg.norm(g_num)
    )
    return errore, numero_forward

# Esercizio 1b: cinque ottimizzatori e scelta del passo

class SGD:
    """aggiorna i parametri con la discesa del gradiente stocastica"""
    def __init__(self, alpha):
        self.alpha = alpha

    def passo(self, parametri, gradienti):
        # aggiorna i parametri usando i gradienti
        for nome in parametri:
            parametri[nome] -= self.alpha * gradienti[nome]

class Momentum:
    """aggiorna i parametri usando il gradiente e la velocità accumulata"""
    def __init__(self, alpha, beta=0.9):
        self.alpha, self.beta = alpha, beta
        self.velocita = None

    def passo(self, parametri, gradienti):
         # inizializza la velocità a zero
        if self.velocita is None:
            self.velocita = {n: np.zeros_like(p) for n, p in parametri.items()}
         # aggiorna la velocità e i parametri
        for nome in parametri:
            self.velocita[nome] = self.beta * self.velocita[nome] + gradienti[nome]
            parametri[nome] -= self.alpha * self.velocita[nome]


class Nesterov:
    """aggiorna i parametri usando il momentum con la correzione di Nesterov"""
    def __init__(self, alpha, beta=0.9):
        self.alpha = alpha
        self.beta = beta
        self.velocita = None

    def passo(self, parametri, gradienti):
        # inizializza la velocità a zero
        if self.velocita is None:
            self.velocita = {nome: np.zeros_like(p) for nome, p in parametri.items()}

        # aggiorna la velocità e applica la correzione di Nesterov
        for nome in parametri:
            self.velocita[nome] = (self.beta * self.velocita[nome] + gradienti[nome])

            parametri[nome] -= self.alpha * (self.beta * self.velocita[nome] + gradienti[nome])

class RMSProp:
    """adatta il passo usando la media dei quadrati dei gradienti"""
    def __init__(self, alpha, beta=0.9, epsilon=1e-8):
        self.alpha, self.beta, self.epsilon = alpha, beta, epsilon
        self.media_quadrati = None

    def passo(self, parametri, gradienti):
        # inizializza la media dei quadrati a zero
        if self.media_quadrati is None:
            self.media_quadrati = {n: np.zeros_like(p) for n, p in parametri.items()}
        for nome in parametri:
            # aggiorna la media dei quadrati dei gradienti
            self.media_quadrati[nome] = (self.beta * self.media_quadrati[nome] + (1 - self.beta) * gradienti[nome] ** 2)
            # adatta il passo usando la media calcolata
            parametri[nome] -= (self.alpha * gradienti[nome] / (np.sqrt(self.media_quadrati[nome]) + self.epsilon))

class Adam:
    """aggiorna i parametri usando le medie dei gradienti e dei loro quadrati"""
    def __init__(self, alpha, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.alpha, self.beta1, self.beta2 = alpha, beta1, beta2
        self.epsilon = epsilon
        self.m = self.v = None
        self.t = 0

    def passo(self, parametri, gradienti):
        # inizializza le medie e conta gli aggiornamenti
        if self.m is None:
            self.m = {n: np.zeros_like(p) for n, p in parametri.items()}
            self.v = {n: np.zeros_like(p) for n, p in parametri.items()}
        self.t += 1

        for nome in parametri:
            # aggiorna le medie dei gradienti e dei loro quadrati
            self.m[nome] = self.beta1 * self.m[nome] + (1 - self.beta1) * gradienti[nome]
            self.v[nome] = self.beta2 * self.v[nome] + (1 - self.beta2) * gradienti[nome] ** 2

            # corregge le medie inizializzate a zero
            m_corretto = self.m[nome] / (1 - self.beta1 ** self.t)
            v_corretto = self.v[nome] / (1 - self.beta2 ** self.t)

            # aggiorna i parametri usando le medie corrette
            parametri[nome] -= self.alpha * m_corretto / (np.sqrt(v_corretto) + self.epsilon)

METODI = {"SGD": SGD, "Momentum": Momentum, "Nesterov": Nesterov, "RMSProp": RMSProp, "Adam": Adam}
LEARNING_RATE = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0)
SEMI = (0, 1, 2) # tre semi per la riproducibilità degli esperimenti

def allena_rete(X_train, y_train, X_val, y_val, parametri_iniziali,
                ottimizzatore, epoche=40, batch_size=32, seed=0,
                registra_curve=True):
    """allena la rete e registra le loss se richiesto"""

    pesi = {nome: p.copy() for nome, p in parametri_iniziali.items()}  # copia i pesi iniziali

    y_train = np.asarray(y_train).reshape(-1, 1)
    y_val = np.asarray(y_val).reshape(-1, 1)
    rng = np.random.default_rng(seed)
    loss_train, loss_val = [], []

    # registra le loss iniziali
    if registra_curve:
        p_tr, _ = forward(X_train, pesi)
        p_va, _ = forward(X_val, pesi)
        loss_train.append(binary_cross_entropy(y_train, p_tr))
        loss_val.append(binary_cross_entropy(y_val, p_va))

    for _ in range(epoche):
        # mescola i dati e li divide in batch
        indici = rng.permutation(len(X_train))
        for inizio in range(0, len(indici), batch_size):
            batch = indici[inizio:inizio + batch_size]
            X_batch, y_batch = X_train[batch], y_train[batch]
            
            # calcola i gradienti e aggiorna i pesi
            _, cache = forward(X_batch, pesi)
            gradienti = backpropagation(y_batch, pesi, cache)

            # aggiorna i pesi utilizzando l'ottimizzatore scelto
            ottimizzatore.passo(pesi, gradienti)

        if registra_curve:
            # registra le loss al termine di ogni epoca
            p_tr, _ = forward(X_train, pesi)
            p_va, _ = forward(X_val, pesi)
            loss_train.append(binary_cross_entropy(y_train, p_tr))
            loss_val.append(binary_cross_entropy(y_val, p_va))

    if not registra_curve:
        # se le curve non sono richieste, salva solo la loss finale di training
        p_tr, _ = forward(X_train, pesi)
        loss_train = [binary_cross_entropy(y_train, p_tr)]

    # restituisce i pesi allenati e le loss registrate
    return pesi, loss_train, loss_val


def ricerca_learning_rate(X_tr, y_tr, X_va, y_va):
    """confronta i learning rate per ogni ottimizzatore usando la loss di training"""
    risultati = {}

    # considera un ottimizzatore alla volta
    for nome, classe in METODI.items():
        risultati[nome] = {}

        # prova ogni learning rate
        for alpha in LEARNING_RATE:
            perdite = []

            # ripete l'allenamento con semi diversi
            for seme in SEMI:
                pesi = inizializza_rete(seed=seme)
                ottimizzatore = classe(alpha=alpha)

                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    _, loss_tr, _ = allena_rete(
                        X_tr, y_tr, X_va, y_va, pesi, ottimizzatore,
                        epoche=40, batch_size=32, seed=seme, registra_curve=False
                    )
                 # salva la loss finale di training
                perdite.append(loss_tr[-1])

            # calcola la loss media, oppure salva nan se ci sono valori non finiti
            risultati[nome][alpha] = (float(np.mean(perdite)) if np.all(np.isfinite(perdite)) else np.nan)

    return risultati


def scegli_passi(risultati):
    """sceglie il learning rate con la loss finale media più bassa per ogni ottimizzatore"""
    print("\nLOSS FINALE MEDIA DOPO 40 EPOCHE (3 SEMI)")
    print(f"{'Alpha':<12}" + "".join(f"{nome:>14}" for nome in METODI))
    
    for alpha in LEARNING_RATE:
        print(f"{alpha:<12g}" + "".join(f"{risultati[nome][alpha]:>14.6f}" for nome in METODI))

    migliori = {}
    print("\nMIGLIOR LEARNING RATE E FINESTRA CON LOSS <= 0.10")

    for nome, valori in risultati.items():
        # considera solo le loss con valori finiti
        validi = {a: loss for a, loss in valori.items() if np.isfinite(loss)}
        
        if not validi:
            # segnala se non ci sono risultati validi
            raise RuntimeError(f"Nessun learning rate valido per {nome}")
        
        migliore = min(validi, key=validi.get)  # trova il learning rate con la loss più bassa

         # raccoglie i learning rate con loss minore o uguale a 0.10
        finestra = [a for a, loss in validi.items() if loss <= 0.10]
        migliori[nome] = migliore
        print(f"{nome:<9} alpha={migliore:<7g} loss={validi[migliore]:.6f}  finestra={finestra}")

    return migliori


def prima_epoca_sotto_soglia(curva, soglia=0.10):
    """restituisce la prima epoca con loss sotto la soglia, oppure None se non esiste"""
    return next((epoca for epoca, loss in enumerate(curva) if loss < soglia), None)


def esperimenti_100_epoche(X_tr, y_tr, X_va, y_va, migliori):
    """allena la rete per 100 epoche con i learning rate scelti e stampa i risultati"""
    
    risultati = {}
    for nome, classe in METODI.items():
        # recupera il learning rate scelto per il metodo
        alpha = migliori[nome]

        # usa gli stessi pesi iniziali per tutti i metodi
        iniziali = inizializza_rete(seed=0)

        # allena la rete e registra le loss
        _, train, val = allena_rete(
            X_tr, y_tr, X_va, y_va, iniziali, classe(alpha=alpha),
            epoche=100, batch_size=32, seed=0
        )

        # salva le curve e le prime epoche con loss sotto la soglia
        risultati[nome] = {
            "alpha": alpha, "train": train, "val": val,
            "epoca_train": prima_epoca_sotto_soglia(train),
            "epoca_val": prima_epoca_sotto_soglia(val),
        }
    
    print("\nRISULTATI DOPO 100 EPOCHE")

    print(
        f"{'Metodo':<12}"
        f"{'Alpha':<10}"
        f"{'Loss train':<14}"
        f"{'Loss val':<14}"
        f"{'Val minima':<14}"
        f"{'Epoca min':<12}"
        f"{'Epoca train':<14}"
        f"{'Epoca val'}"
    )

    for nome, dati in risultati.items():
        # trova la loss di validazione minima e la relativa epoca
        loss_val_minima = np.min(dati["val"])
        epoca_val_minima = np.argmin(dati["val"])

        print(
            f"{nome:<12}"
            f"{dati['alpha']:<10g}"
            f"{dati['train'][-1]:<14.6f}"
            f"{dati['val'][-1]:<14.6f}"
            f"{loss_val_minima:<14.6f}"
            f"{epoca_val_minima:<12}"
            f"{str(dati['epoca_train']):<14}"
            f"{dati['epoca_val']}"
        )

    return risultati


def grafico_dataset(X, y, cartella):
    """mostra il dataset delle due mezze lune e salva il grafico"""
    
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(X[y == 0, 0], X[y == 0, 1], alpha=0.7,
               label="Classe 0 - lecita")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], alpha=0.7,
               label="Classe 1 - malevola")
    ax.set(title="Dataset delle due mezze lune", xlabel="Caratteristica x1",
           ylabel="Caratteristica x2")
    ax.axis("equal")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(cartella / "dataset_lune.png", dpi=300)
    plt.show()
    plt.close(fig)

def grafico_learning_rate(risultati, cartella):
    """mostra e salva il confronto delle loss medie per ogni learning rate"""
    
    fig, ax = plt.subplots(figsize=(10, 6))

    for nome, valori in risultati.items():
        # seleziona i learning rate con loss finita e positiva
        x = [a for a, loss in valori.items() if np.isfinite(loss) and loss > 0]

        # disegna la curva con entrambi gli assi in scala logaritmica
        ax.loglog(x, [valori[a] for a in x], marker="o", label=nome)

    ax.axhline(0.10, color="gray", linestyle="--", label="Soglia 0.10")
    ax.set(title="Ricerca del learning rate - cinque ottimizzatori",
           xlabel="Learning rate", ylabel="Training loss finale media")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(cartella / "ricerca_learning_rate.png", dpi=300)
    plt.show()
    plt.close(fig)


def grafico_100_epoche(risultati, cartella):
    """mostra e salva le curve di training e validation per ogni ottimizzatore"""

    fig, assi = plt.subplots(2, 3, figsize=(15, 9)) # crea una griglia con sei grafici

    for ax, (nome, dati) in zip(assi.flat, risultati.items()):
        epoche = range(len(dati["train"]))
        ax.plot(epoche, dati["train"], label="Training")
        ax.plot(epoche, dati["val"], label="Validation")
        ax.axhline(0.10, color="gray", linestyle="--", label="Soglia 0.10")
        ax.set(title=f"{nome} - alpha = {dati['alpha']:g}",
               xlabel="Epoche", ylabel="Binary Cross-Entropy")
        ax.grid(alpha=0.3)
        ax.legend()
    assi[1, 2].axis("off")
    fig.suptitle("Training e validation loss - 100 epoche")
    fig.tight_layout()
    fig.savefig(cartella / "curve_100_epoche.png", dpi=300)
    plt.show()
    plt.close(fig)


def main():
    """esegue gli esperimenti e salva i grafici"""

    # crea la cartella per le figure
    cartella = Path(__file__).resolve().parent / "figure"
    cartella.mkdir(exist_ok=True)

    # genera il dataset, lo divide e standardizza le caratteristiche
    X, y = genera_lune()
    X_tr, y_tr, X_va, y_va = dividi_standardizza(X, y)

    print("DATASET")
    print(f"Osservazioni: {len(X)}; caratteristiche: {X.shape[1]}")
    print(f"Classe 0: {np.sum(y == 0)}; classe 1: {np.sum(y == 1)}")
    print(f"Training: {len(X_tr)}; validation: {len(X_va)}")
    print(f"Training classi 0/1: {np.sum(y_tr == 0)}/{np.sum(y_tr == 1)}")
    print(f"Validation classi 0/1: {np.sum(y_va == 0)}/{np.sum(y_va == 1)}")
    grafico_dataset(X, y, cartella)

    # inizializza la rete e seleziona un batch di 32 osservazioni
    iniziali = inizializza_rete(seed=42)
    X_batch, y_batch = X_tr[:32], y_tr[:32].reshape(-1, 1)

    # calcola le probabilità e i gradienti sul batch
    probabilita, cache = forward(X_batch, iniziali)
    gradienti = backpropagation(y_batch, iniziali, cache)
    n_parametri = sum(p.size for p in iniziali.values())

    print("\nRETE E BACKPROPAGATION")
    print(f"Architettura: 2-16-16-1; mini-batch: {len(X_batch)}")
    print(f"Loss iniziale: {binary_cross_entropy(y_batch, probabilita):.6f}")
    for nome, valore in gradienti.items():
        print(f"{nome}: forma {valore.shape}, norma {np.linalg.norm(valore):.6f}")
    print(f"Parametri e gradienti: {n_parametri}")

    # verifica i gradienti su 50 osservazioni
    errore, forward_num = verifica_gradiente(X_tr[:50], y_tr[:50], iniziali)
    print("\nGRADIENT CHECKING")
    print(f"50 osservazioni; h=1e-5; forward numerici={forward_num}")
    print(f"Errore relativo: {errore:.10e}")
    print("Per il gradiente analitico servono 1 forward e 1 backpropagation.")

    # confronta i learning rate e sceglie il migliore per ogni metodo
    risultati_lr = ricerca_learning_rate(X_tr, y_tr, X_va, y_va)
    migliori = scegli_passi(risultati_lr)
    grafico_learning_rate(risultati_lr, cartella)

    # allena per 100 epoche e mostra le curve delle loss
    risultati_100 = esperimenti_100_epoche(X_tr, y_tr, X_va, y_va, migliori)
    grafico_100_epoche(risultati_100, cartella)
    print(f"\nFigure salvate in: {cartella}")

if __name__ == "__main__":
    main()