"""
Attività di laboratorio 2
Addestramento e confronto di modelli di apprendimento automatico
-
Esercizio 2: Addestramento e confronto di modelli
"""

import numpy as np

from es1_ottimizzatori import (
    sigmoid,
    binary_cross_entropy,
    Adam,
    inizializza_rete,
    forward,
    backpropagation,
)

def genera_connessioni(N=2000, seed=7):
    """Righe in ordine temporale. y = 1 attacco, y = 0 lecito."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(N, 8))
    s = (1.2 * Z[:, 0] * Z[:, 1] + 0.8 * Z[:, 2] ** 2 + 0.8 * Z[:, 3]
         - 0.5 * Z[:, 0] + 0.4 * rng.normal(size=N))
    y = (s > 1.5).astype(float)
    scala = np.array([1e5, 1e2, 1.0, 10.0, 1e3, 1.0, 1e4, 0.1])
    centro = np.array([5e5, 3e2, 0.0, 50.0, 2e3, 0.0, 1e4, 1.0])
    X = centro + scala * Z # le colonne 4..7 sono rumore
    return X, y

def suddivisione_temporale(X, y, f_tr=0.6, f_va=0.2):
    N = len(y)
    a, b = int(f_tr * N), int((f_tr + f_va) * N)
    return (X[:a], y[:a]), (X[a:b], y[a:b]), (X[b:], y[b:])

# preparazione del dataset

def standardizza(X_tr, X_va, X_te):
    """standardizza i dati usando media e deviazione standard del training"""

    # calcola media e deviazione standard per ogni caratteristica
    media = X_tr.mean(axis=0)
    deviazione = X_tr.std(axis=0)

    deviazione = np.where(deviazione == 0, 1.0, deviazione) # evita la divisione per zero

    # applica la stessa trasformazione ai tre insiemi
    X_tr = (X_tr - media) / deviazione
    X_va = (X_va - media) / deviazione
    X_te = (X_te - media) / deviazione

    return (X_tr, X_va, X_te)


def prepara_dataset():
    """genera e prepara il dataset, poi stampa le informazioni principali"""

    X, y = genera_connessioni()
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = suddivisione_temporale(X, y)
    X_tr, X_va, X_te = standardizza(X_tr, X_va, X_te)

    print('\nDATASET')
    print(f'Osservazioni totali: {len(X)}')
    print(f'Numero di caratteristiche: {X.shape[1]}')
    print(f'Connessioni lecite: {np.sum(y == 0)}')
    print(f'Connessioni malevole: {np.sum(y == 1)}')

    print('\nSUDDIVISIONE DEI DATI')
    print(f'Training: {X_tr.shape}')
    print(f'Validation: {X_va.shape}')
    print(f'Test: {X_te.shape}')

    print('\nDISTRIBUZIONE DELLE CLASSI')
    for nome, etichette in [('Training', y_tr), ('Validation', y_va), ('Test', y_te)]:
        lecite = np.sum(etichette == 0)
        malevole = np.sum(etichette == 1)
        print(f'{nome}: lecite={lecite}, malevole={malevole}')

    print('\nVERIFICA STANDARDIZZAZIONE')
    print('Media training:', np.round(X_tr.mean(axis=0), 4))
    print('Deviazione standard:', np.round(X_tr.std(axis=0), 4))

    return (X_tr, y_tr, X_va, y_va, X_te, y_te)

def prepara_dati_finali(X_tr, y_tr, X_va, y_va, X_te):
    """unisce training e validation e prepara i dati per l'addestramento finale"""

    X_finale = np.vstack((X_tr, X_va))
    y_finale = np.concatenate((y_tr, y_va))

    media = X_finale.mean(axis=0)
    deviazione = X_finale.std(axis=0)

    deviazione = np.where(deviazione == 0, 1.0, deviazione) # evita la divisione per zero

    X_finale = (X_finale - media) / deviazione
    X_test_finale = (X_te - media) / deviazione

    return (X_finale, y_finale, X_test_finale)

def calcola_metriche(y_reali, probabilita, soglia=0.5):
    """calcola le metriche confrontando le classi reali con quelle predette"""

    # trasforma etichette e probabilità in vettori
    y_reali = np.asarray(y_reali).ravel()
    probabilita = np.asarray(probabilita).ravel()

    # assegna la classe 1 alle probabilità maggiori o uguali alla soglia
    y_predette = (probabilita >= soglia).astype(int)

    # conta le previsioni corrette e gli errori per ogni classe
    TP = np.sum((y_reali == 1) & (y_predette == 1))
    TN = np.sum((y_reali == 0) & (y_predette == 0))
    FP = np.sum((y_reali == 0) & (y_predette == 1))
    FN = np.sum((y_reali == 1) & (y_predette == 0))

    # calcola accuratezza, precisione e richiamo evitando divisioni per zero
    accuratezza = (TP + TN) / len(y_reali)
    precisione = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    richiamo = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    # combina precisione e richiamo nell'f1-score
    f1 = (2 * precisione * richiamo / (precisione + richiamo) if (precisione + richiamo) > 0 else 0.0)

    return {
        "Accuratezza": accuratezza,
        "Precisione": precisione,
        "Recall": richiamo,
        "F1": f1,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN}


def stampa_metriche(metriche):
    """stampa le metriche e i conteggi delle previsioni"""
    print(f"Accuratezza: {metriche['Accuratezza']:.6f}")
    print(f"Precisione: {metriche['Precisione']:.6f}")
    print(f"Recall: {metriche['Recall']:.6f}")
    print(f"F1-score: {metriche['F1']:.6f}")
    print(f"Veri positivi: {metriche['TP']}")
    print(f"Veri negativi: {metriche['TN']}")
    print(f"Falsi positivi: {metriche['FP']}")
    print(f"Falsi negativi: {metriche['FN']}")

# Esercizio 2a. Selezione degli iperparametri

# 2.1a regressione logistica

def inizializza_logistica():
    """inizializza a zero i pesi e il bias della regressione logistica"""
    return {'W': np.zeros((8, 1)), 'b': np.zeros((1, 1))}

def forward_logistica(X, parametri):
    """calcola la probabilità della classe 1"""

    # calcola la combinazione lineare e applica la sigmoide
    z = X @ parametri['W'] + parametri['b']
    return sigmoid(z)

def gradienti_logistica(X, y, parametri):
    """calcola i gradienti della loss rispetto ai pesi e al bias"""

    # trasforma le etichette in un vettore colonna
    y = y.reshape(-1, 1)

    # calcola le probabilità e l'errore diviso per il numero di osservazioni
    probabilita = forward_logistica(X, parametri)
    errore = (probabilita - y) / len(X)
    return {'W': X.T @ errore, 'b': np.sum(errore, axis=0, keepdims=True)}

def addestra_logistica(X_tr, y_tr, X_va, y_va, alpha=0.01, epoche=300, batch_size=32, pazienza=15, seed=0):
    """allena la regressione logistica con Adam e arresto anticipato"""

    # inizializza i parametri, l'ottimizzatore e il generatore casuale
    parametri = inizializza_logistica()
    ottimizzatore = Adam(alpha=alpha)
    rng = np.random.default_rng(seed)

    # prepara le variabili per salvare il modello migliore
    migliore_loss = np.inf
    migliore_epoca = 0
    migliori_parametri = None
    senza_miglioramenti = 0
    loss_train = []
    loss_val = []

    # trasforma le etichette in vettori colonna
    y_tr = y_tr.reshape(-1, 1)
    y_va = y_va.reshape(-1, 1)

    for epoca in range(1, epoche + 1):
        # mescola gli indici e seleziona un batch alla volta
        indici = rng.permutation(len(X_tr))
        for inizio in range(0, len(X_tr), batch_size):
            idx = indici[inizio:inizio + batch_size]
            X_batch = X_tr[idx]
            y_batch = y_tr[idx]

            # calcola i gradienti e aggiorna i parametri
            gradienti = gradienti_logistica(
                X_batch, y_batch, parametri
            )
            ottimizzatore.passo(parametri, gradienti)

        # calcola e salva le loss di training e validation
        p_train = forward_logistica(X_tr, parametri)
        p_val = forward_logistica(X_va, parametri)
        errore_train = binary_cross_entropy(y_tr, p_train)
        errore_val = binary_cross_entropy(y_va, p_val)
        loss_train.append(errore_train)
        loss_val.append(errore_val)

        # salva i parametri se la loss di validation migliora di oltre 1e-6
        if errore_val < migliore_loss - 1e-6:
            migliore_loss = errore_val
            migliore_epoca = epoca
            migliori_parametri = {nome: valore.copy() for nome, valore in parametri.items()}
            senza_miglioramenti = 0
        else:
            senza_miglioramenti += 1

        # interrompe l'allenamento dopo troppe epoche senza miglioramenti
        if senza_miglioramenti >= pazienza:
            break

    return (migliori_parametri, migliore_epoca, loss_train, loss_val)

def esegui_logistica(X_tr, y_tr, X_va, y_va):
    """allena la regressione logistica e valuta i risultati sulla validation"""

    print('\nREGRESSIONE LOGISTICA')

    # inizializza i parametri e calcola probabilità, loss e gradienti
    parametri = inizializza_logistica()
    probabilita = forward_logistica(X_tr, parametri)
    loss = binary_cross_entropy(y_tr.reshape(-1, 1), probabilita)
    gradienti = gradienti_logistica(X_tr, y_tr, parametri)

    print('Architettura: 8-1')
    print(f'Parametri addestrabili: {sum((p.size for p in parametri.values()))}')
    print(f'Probabilità previste: {probabilita.shape}')
    print(f'Loss iniziale: {loss:.6f}')
    print(f"Gradiente W: {gradienti['W'].shape}")
    print(f"Gradiente b: {gradienti['b'].shape}")

    # allena il modello e recupera i pesi dell'epoca migliore
    print('\nADDESTRAMENTO REGRESSIONE LOGISTICA')
    pesi_log, epoca_log, loss_tr, loss_va = addestra_logistica(X_tr, y_tr, X_va, y_va)

    print('Learning rate: 0.01')
    print(f'Epoche eseguite: {len(loss_tr)}')
    print(f'Epoca migliore: {epoca_log}')
    print(f'Loss training iniziale: {loss_tr[0]:.6f}')
    print(f'Loss training finale: {loss_tr[-1]:.6f}')
    print(f'Loss validation minima: {min(loss_va):.6f}')

    # calcola le metriche sulla validation usando i pesi migliori
    probabilita_val = forward_logistica(X_va, pesi_log)
    metriche_log = calcola_metriche(y_va, probabilita_val)
    print('\nMETRICHE REGRESSIONE LOGISTICA - VALIDATION')
    stampa_metriche(metriche_log)

    return (pesi_log, epoca_log, metriche_log)

# 2.2a SVM gaussiana

def kernel_gaussiano(X1, X2, gamma):
    """calcola il kernel gaussiano tra i due insiemi di dati"""

    # calcola le distanze al quadrato tra le osservazioni
    norme1 = np.sum(X1 ** 2, axis=1)[:, None]
    norme2 = np.sum(X2 ** 2, axis=1)[None, :]
    distanze_quadrate = norme1 + norme2 - 2 * X1 @ X2.T

    # elimina eventuali valori negativi dovuti agli arrotondamenti
    distanze_quadrate = np.maximum(distanze_quadrate, 0.0)
    return np.exp(-gamma * distanze_quadrate)


def addestra_svm_smo(X, y, C=1.0, gamma=0.1,
                     tolleranza=1e-3, max_iter=100000):
    """allena una SVM con kernel gaussiano usando il metodo SMO"""

    y = np.where(np.asarray(y).ravel() == 1, 1.0, -1.0)
    n = len(y)

    # calcola il kernel e inizializza le variabili
    K = kernel_gaussiano(X, X, gamma)
    alpha = np.zeros(n)
    g = np.zeros(n)
    eps = 1e-10
    aggiornamenti = 0
    convergente = False

    for _ in range(max_iter):
        F = y - g

        # individua gli indici che possono essere aggiornati
        I_up = (((y == 1) & (alpha < C - eps)) | ((y == -1) & (alpha > eps)))
        I_low = (((y == 1) & (alpha > eps)) | ((y == -1) & (alpha < C - eps)))

        # sceglie la coppia con la maggiore violazione
        i = int(np.argmax(np.where(I_up, F, -np.inf)))
        j = int(np.argmin(np.where(I_low, F, np.inf)))
        violazione = F[i] - F[j]

        # termina se la violazione è entro la tolleranza
        if violazione <= tolleranza:
            convergente = True
            break

        # calcola la curvatura lungo la direzione di aggiornamento
        eta = K[i, i] + K[j, j] - 2 * K[i, j]
        if eta <= 1e-12:
            break

        # limita il passo per mantenere gli alpha tra 0 e C
        limite_i = C - alpha[i] if y[i] == 1 else alpha[i]
        limite_j = alpha[j] if y[j] == 1 else C - alpha[j]
        delta = min(violazione / eta, limite_i, limite_j)
        if delta <= 1e-12:
            break

        # aggiorna i due alpha e i valori del modello senza bias
        alpha[i] += y[i] * delta
        alpha[j] -= y[j] * delta
        g += delta * (K[:, i] - K[:, j])
        aggiornamenti += 1

    # cerca gli alpha strettamente compresi tra 0 e C
    F = y - g
    liberi = (alpha > eps) & (alpha < C - eps)

    # calcola il bias usando i vettori di supporto liberi
    if np.any(liberi):
        b = float(np.mean(F[liberi]))
    else:
        # se non ci sono, usa i limiti ricavati dagli altri alpha
        I_up = (((y == 1) & (alpha < C - eps)) |((y == -1) & (alpha > eps)))
        I_low = (((y == 1) & (alpha > eps)) |((y == -1) & (alpha < C - eps)))
        b = float((np.max(F[I_up]) + np.min(F[I_low])) / 2)

    # calcola i valori delle funzioni obiettivo primale e duale
    norma2 = float(np.dot(alpha * y, g))
    primale = (0.5 * norma2 + C * np.maximum(0, 1 - y * (g + b)).sum())
    duale = alpha.sum() - 0.5 * norma2

    return {
        "alpha": alpha,
        "b": b,
        "X": X,
        "y": y,
        "gamma": gamma,
        "iterazioni": aggiornamenti,
        "convergente": convergente,
        "violazione": float(max(0, violazione)),
        "gap_duale": float(primale - duale),
        "support_vectors": int(np.sum(alpha > eps))
    }

def predici_svm(X, modello):
    """assegna a ogni osservazione la classe 0 oppure 1"""

    # seleziona i vettori di supporto e i relativi alpha
    alpha = modello['alpha']
    indici_sv = alpha > 1e-10
    X_sv = modello['X'][indici_sv]
    y_sv = modello['y'][indici_sv]
    alpha_sv = alpha[indici_sv]

    # calcola i punteggi usando il kernel e il bias
    K = kernel_gaussiano(X, X_sv, modello['gamma'])
    punteggi = K @ (alpha_sv * y_sv) + modello['b']

    # assegna la classe 1 ai punteggi non negativi
    return (punteggi >= 0).astype(int)


def ricerca_iperparametri_svm(X_tr, y_tr, X_va, y_va):
    """sceglie C e gamma in base all'f1 sulla validation tra i modelli convergenti"""

    # definisce i valori da provare
    valori_C = [0.01, 0.1, 1, 10, 100]
    valori_gamma = [0.01, 0.03, 0.1, 0.3, 1]

    risultati = []
    modello_migliore = None
    migliore_f1 = -1
    parametri_migliori = None

    # prova tutte le combinazioni di C e gamma
    for C in valori_C:
        for gamma in valori_gamma:
            modello = addestra_svm_smo(X_tr, y_tr, C=C, gamma=gamma)

            # valuta il modello sulla validation
            previsioni = predici_svm(X_va, modello)
            metriche = calcola_metriche(y_va, previsioni)

            risultati.append({
                "C": C,
                "gamma": gamma,
                "F1": metriche["F1"],
                "convergente": modello["convergente"]})
            print(
                f"C={C:<7g} gamma={gamma:<5g} "
                f"F1={metriche['F1']:.6f} "
                f"Convergenza={modello['convergente']}"
            )

            # aggiorna il modello migliore se è convergente e migliora l'f1
            if modello["convergente"] and metriche["F1"] > migliore_f1:
                migliore_f1 = metriche["F1"]
                modello_migliore = modello
                parametri_migliori = (C, gamma)

    return modello_migliore, parametri_migliori, risultati


def esegui_svm(X_tr, y_tr, X_va, y_va):
    """verifica il kernel, allena la SVM e cerca gli iperparametri migliori"""

    # calcola il kernel su cinque osservazioni
    print("\nSVM GAUSSIANA")
    K_prova = kernel_gaussiano(X_tr[:5], X_tr[:5], gamma=0.1)

    print("Dimensione matrice kernel:", K_prova.shape)
    print("Matrice kernel:")
    print(np.round(K_prova, 4))
    print("Matrice simmetrica:", np.allclose(K_prova, K_prova.T))
    print("Diagonale:", np.round(np.diag(K_prova), 4))

    print("\nPROVA ADDESTRAMENTO SVM - SMO")
    modello_svm = addestra_svm_smo(X_tr, y_tr, C=1.0, gamma=0.1)
    print("C:", 1.0)
    print("Gamma:", 0.1)
    print("Aggiornamenti SMO:", modello_svm["iterazioni"])
    print("Convergenza:", modello_svm["convergente"])
    print("Violazione finale:", modello_svm["violazione"])
    print("Vettori di supporto:", modello_svm["support_vectors"])
    print("Duality gap:", modello_svm["gap_duale"])

    previsioni_svm = predici_svm(X_va, modello_svm)
    metriche_svm = calcola_metriche(y_va, previsioni_svm)
    print("\nMETRICHE SVM - VALIDATION")
    stampa_metriche(metriche_svm)

    print("\nRICERCA IPERPARAMETRI SVM")
    svm_migliore, parametri_svm, risultati_svm = ricerca_iperparametri_svm(
        X_tr, y_tr, X_va, y_va
    )

    if svm_migliore is not None:
        # valuta il modello migliore sulla validation
        C_migliore, gamma_migliore = parametri_svm
        previsioni = predici_svm(X_va, svm_migliore)
        metriche = calcola_metriche(y_va, previsioni)

        print("\nMIGLIORE SVM")
        print(f"C migliore: {C_migliore}")
        print(f"Gamma migliore: {gamma_migliore}")
        print(f"F1 validation: {metriche['F1']:.6f}")
        print(f"Recall: {metriche['Recall']:.6f}")
        print(f"Falsi negativi: {metriche['FN']}")
        print(f"Vettori di supporto: {svm_migliore['support_vectors']}")
        print(f"Duality gap: {svm_migliore['gap_duale']}")
    else:
        print("Nessuna combinazione ha raggiunto la convergenza.")

    return svm_migliore, parametri_svm, risultati_svm

# 2.3a Rete 8–32–32–1 con Adam

def inizializza_rete_connessioni(seed=0):
    """inizializza la rete con architettura 8-32-32-1"""
    return inizializza_rete(seed=seed, dimensioni=(8, 32, 32, 1))

def loss_rete_l2(y, probabilita, parametri, lam):
    """calcola la loss aggiungendo la penalizzazione L2 sui pesi"""

    loss_bce = binary_cross_entropy(y, probabilita)

    # penalizza i pesi grandi senza includere i bias
    penalizzazione = 0.5 * lam * sum(
        np.sum(parametri[nome] ** 2)
        for nome in ("W1", "W2", "W3")
    )
    return loss_bce + penalizzazione

def gradienti_rete_l2(y, parametri, cache, lam):
    """calcola i gradienti includendo il contributo della penalizzazione L2"""

    gradienti = backpropagation(y, parametri, cache)

    # aggiunge il contributo L2 solo ai gradienti dei pesi
    for nome in ('W1', 'W2', 'W3'):
        gradienti[nome] += lam * parametri[nome]
    return gradienti

def addestra_rete_l2(X_tr, y_tr, X_va, y_va,
                     lam, alpha=0.003,
                     epoche=300, batch_size=32,
                     pazienza=15, seed=0):
    """allena la rete con L2, Adam e arresto anticipato"""

    # inizializza la rete, l'ottimizzatore e il generatore casuale
    parametri = inizializza_rete_connessioni(seed=seed)
    ottimizzatore = Adam(alpha=alpha)
    rng = np.random.default_rng(seed)

    # prepara le variabili per salvare il modello migliore e le loss
    migliore_loss = np.inf
    migliore_epoca = 0
    migliori_parametri = None
    senza_miglioramenti = 0
    loss_train = []
    loss_val = []

    for epoca in range(1, epoche + 1):
        # mescola gli indici e seleziona un batch alla volta
        indici = rng.permutation(len(X_tr))
        for inizio in range(0, len(X_tr), batch_size):
            idx = indici[inizio:inizio + batch_size]
            X_batch = X_tr[idx]
            y_batch = y_tr[idx]

            # calcola i gradienti con L2 e aggiorna i parametri
            _, cache = forward(X_batch, parametri)
            gradienti = gradienti_rete_l2(y_batch, parametri, cache, lam)
            ottimizzatore.passo(parametri, gradienti)

        # calcola le probabilità su training e validation
        p_train, _ = forward(X_tr, parametri)
        p_val, _ = forward(X_va, parametri)

        # usa la penalizzazione L2 solo nella loss di training
        errore_train = loss_rete_l2(y_tr, p_train, parametri, lam)
        errore_val = binary_cross_entropy(y_va, p_val)
        loss_train.append(errore_train)
        loss_val.append(errore_val)

        # salva il modello se la loss di validation migliora di oltre 1e-6
        if errore_val < migliore_loss - 1e-6:
            migliore_loss = errore_val
            migliore_epoca = epoca
            migliori_parametri = {nome: valore.copy() for nome, valore in parametri.items()}
            senza_miglioramenti = 0
        else:
            senza_miglioramenti += 1

        # interrompe l'allenamento dopo troppe epoche senza miglioramenti
        if senza_miglioramenti >= pazienza:
            break

    return {
        "parametri": migliori_parametri,
        "epoca_migliore": migliore_epoca,
        "epoche_eseguite": len(loss_train),
        "loss_val_minima": migliore_loss,
        "loss_train": loss_train,
        "loss_val": loss_val
    }


def sezione_aurea_rete(X_tr, y_tr, X_va, y_va):
    """cerca un valore di lambda con la sezione aurea usando la loss di validation"""

    # definisce l'intervallo di ricerca per t, dove lambda = 10 ** t
    a = -6.0
    b = -1.0
    phi = (np.sqrt(5) - 1) / 2
    risultati = []

    def valuta(t):
        """allena la rete con il lambda scelto e salva il risultato"""

        lam = 10 ** t
        risultato = addestra_rete_l2(
            X_tr, y_tr, X_va, y_va,
            lam=lam,
            alpha=0.003,
            epoche=300,
            pazienza=15,
            seed=0
        )

        prova = {
            "t": t,
            "lambda": lam,
            "loss": risultato["loss_val_minima"],
            "epoca": risultato["epoca_migliore"],
            "risultato": risultato
        }
        risultati.append(prova)
        print(
            f"{len(risultati):<7}"
            f"{t:<12.6f}"
            f"{lam:<14.8f}"
            f"{prova['loss']:<14.6f}"
            f"{prova['epoca']}"
        )
        return prova

    print("\nRICERCA LAMBDA - SEZIONE AUREA")
    print(
        f"{'Prova':<7}"
        f"{'t':<12}"
        f"{'Lambda':<14}"
        f"{'Loss val':<14}"
        f"{'Epoca'}")

    # valuta i primi due punti interni all'intervallo
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    risultato_c = valuta(c)
    risultato_d = valuta(d)

    # restringe l'intervallo confrontando le loss nei due punti
    for _ in range(6):
        if risultato_c["loss"] <= risultato_d["loss"]:
            b = d
            d = c
            risultato_d = risultato_c
            c = b - phi * (b - a)
            risultato_c = valuta(c)
        else:
            a = c
            c = d
            risultato_c = risultato_d
            d = a + phi * (b - a)
            risultato_d = valuta(d)

    # sceglie la prova con la loss di validation più bassa
    migliore = min(risultati, key=lambda r: r["loss"])
    return migliore, risultati


def griglia_uniforme_rete(X_tr, y_tr, X_va, y_va):
    """confronta otto valori di lambda con esponenti equidistanti"""

    # sceglie otto valori di t tra -6 e -1
    valori_t = np.linspace(-6, -1, 8)
    risultati = []

    print("\nRICERCA LAMBDA - GRIGLIA UNIFORME")
    print(
        f"{'Prova':<7}"
        f"{'t':<12}"
        f"{'Lambda':<14}"
        f"{'Loss val':<14}"
        f"{'Epoca'}"
    )

    for t in valori_t:
        # calcola lambda e allena la rete
        lam = 10 ** t
        risultato = addestra_rete_l2(
            X_tr, y_tr, X_va, y_va,
            lam=lam,
            alpha=0.003,
            epoche=300,
            pazienza=15,
            seed=0
        )

        prova = {
            "t": t,
            "lambda": lam,
            "loss": risultato["loss_val_minima"],
            "epoca": risultato["epoca_migliore"],
            "risultato": risultato
        }
        risultati.append(prova)
        print(
            f"{len(risultati):<7}"
            f"{t:<12.6f}"
            f"{lam:<14.8f}"
            f"{prova['loss']:<14.6f}"
            f"{prova['epoca']}"
        )

    # sceglie la prova con la loss di validation più bassa
    migliore = min(risultati, key=lambda r: r["loss"])
    return migliore, risultati

def esegui_rete_l2(X_tr, y_tr, X_va, y_va):
    """prova la rete con L2, confronta le ricerche di lambda e valuta il modello scelto"""

    print("\nRETE NEURALE L2")

    # inizializza la rete e calcola le probabilità su un batch
    parametri_rete = inizializza_rete_connessioni(seed=0)
    probabilita_rete, cache_rete = forward(X_tr[:32], parametri_rete)

    # calcola le loss e i gradienti con un lambda di prova
    lam_prova = 0.001
    loss_bce = binary_cross_entropy(y_tr[:32], probabilita_rete)
    loss_l2 = loss_rete_l2(y_tr[:32], probabilita_rete, parametri_rete, lam_prova)
    gradienti_rete = gradienti_rete_l2(y_tr[:32], parametri_rete, cache_rete, lam_prova)

    print("Architettura: 8-32-32-1")
    print("Parametri addestrabili:", sum(p.size for p in parametri_rete.values()))
    print("Dimensione previsioni:", probabilita_rete.shape)
    print(f"Loss BCE iniziale: {loss_bce:.6f}")
    print(f"Loss con L2: {loss_l2:.6f}")
    print("Dimensione gradiente W1:", gradienti_rete["W1"].shape)
    print("Dimensione gradiente W2:", gradienti_rete["W2"].shape)
    print("Dimensione gradiente W3:", gradienti_rete["W3"].shape)

    print("\nADDESTRAMENTO RETE NEURALE L2")
    risultato_rete = addestra_rete_l2(X_tr, y_tr, X_va, y_va, lam=0.001)
    print("Lambda:", 0.001)
    print("Learning rate:", 0.003)
    print("Epoche eseguite:", risultato_rete["epoche_eseguite"])
    print("Epoca migliore:", risultato_rete["epoca_migliore"])
    print("Loss training finale:", f"{risultato_rete['loss_train'][-1]:.6f}")
    print("Loss validation minima:", f"{risultato_rete['loss_val_minima']:.6f}")

    print("\nOTTIMIZZAZIONE LAMBDA")
    migliore_aurea, prove_aurea = sezione_aurea_rete(X_tr, y_tr, X_va, y_va)
    print("\nMIGLIORE RISULTATO - SEZIONE AUREA")
    print(f"Addestramenti effettuati: {len(prove_aurea)}")
    print(f"Lambda migliore: {migliore_aurea['lambda']:.8f}")
    print(f"t migliore: {migliore_aurea['t']:.6f}")
    print(f"Loss validation minima: {migliore_aurea['loss']:.6f}")
    print(f"Epoca migliore: {migliore_aurea['epoca']}")

    # cerca lambda con la griglia uniforme
    migliore_griglia, prove_griglia = griglia_uniforme_rete(X_tr, y_tr, X_va, y_va)

    print("\nCONFRONTO DEI METODI DI RICERCA")
    print(
        f"{'Metodo':<20}"
        f"{'Addestramenti':<17}"
        f"{'Lambda migliore':<20}"
        f"{'Loss val minima'}")
    print(
        f"{'Sezione aurea':<20}"
        f"{len(prove_aurea):<17}"
        f"{migliore_aurea['lambda']:<20.8f}"
        f"{migliore_aurea['loss']:.6f}")
    print(
        f"{'Griglia uniforme':<20}"
        f"{len(prove_griglia):<17}"
        f"{migliore_griglia['lambda']:<20.8f}"
        f"{migliore_griglia['loss']:.6f}")

    # seleziona il risultato con la loss di validation più bassa
    if migliore_aurea["loss"] < migliore_griglia["loss"]:
        migliore_rete = migliore_aurea
        print("\nMetodo migliore: Sezione aurea")
    elif migliore_griglia["loss"] < migliore_aurea["loss"]:
        migliore_rete = migliore_griglia
        print("\nMetodo migliore: Griglia uniforme")
    else:
        migliore_rete = migliore_aurea
        print("\nI due metodi hanno ottenuto la stessa loss.")
    print(f"Lambda selezionato: {migliore_rete['lambda']:.8f}")
    print(f"Epoca migliore: {migliore_rete['epoca']}")

    # calcola le metriche sulla validation usando i parametri migliori
    parametri_migliori_rete = migliore_rete["risultato"]["parametri"]
    probabilita_rete_val, _ = forward(X_va, parametri_migliori_rete)
    metriche_rete = calcola_metriche(y_va, probabilita_rete_val)
    print("\nMETRICHE RETE NEURALE L2 - VALIDATION")
    print(f"Lambda: {migliore_rete['lambda']:.8f}")
    print(f"Epoca migliore: {migliore_rete['epoca']}")
    stampa_metriche(metriche_rete)

    return migliore_rete, prove_aurea, prove_griglia, metriche_rete

# Esercizio 2b: valutazione finale

def riaddestra_logistica(X, y, epoche, alpha=0.01, batch_size=32, seed=0):
    """riaddestra la regressione logistica per il numero di epoche scelto"""

    # inizializza i parametri, l'ottimizzatore e il generatore casuale
    parametri = inizializza_logistica()
    ottimizzatore = Adam(alpha=alpha)
    rng = np.random.default_rng(seed)

    for epoca in range(epoche):
        # mescola gli indici e seleziona un batch alla volta
        indici = rng.permutation(len(X))
        for inizio in range(0, len(X), batch_size):
            idx = indici[inizio:inizio + batch_size]

            # calcola i gradienti e aggiorna i parametri
            gradienti = gradienti_logistica(X[idx], y[idx], parametri)
            ottimizzatore.passo(parametri, gradienti)

    return parametri


def riaddestra_rete_l2(X, y, lam, epoche, alpha=0.003, batch_size=32, seed=0):
    """riaddestra la rete con L2 per il numero di epoche scelto"""

    # inizializza la rete, l'ottimizzatore e il generatore casuale
    parametri = inizializza_rete_connessioni(seed=seed)
    ottimizzatore = Adam(alpha=alpha)
    rng = np.random.default_rng(seed)

    for epoca in range(epoche):
        # mescola gli indici e seleziona un batch alla volta
        indici = rng.permutation(len(X))
        for inizio in range(0, len(X), batch_size):
            idx = indici[inizio:inizio + batch_size]
            X_batch = X[idx]
            y_batch = y[idx]

            # calcola l'output e i gradienti con L2
            _, cache = forward(X_batch, parametri)
            gradienti = gradienti_rete_l2(y_batch, parametri, cache, lam)

            # aggiorna i parametri con Adam
            ottimizzatore.passo(parametri, gradienti)

    return parametri

def esegui_punto_2b(
        X_tr, y_tr, X_va, y_va, X_te, y_te,
        epoca_log, metriche_log_val, svm_val, parametri_svm,
        migliore_rete, metriche_rete_val):
    """riaddestra i tre modelli sui dati uniti e confronta i risultati sul test"""

    # unisce training e validation e prepara il test
    print("\nPREPARAZIONE DATI FINALI")
    X_finale, y_finale, X_test_finale = prepara_dati_finali(
        X_tr, y_tr, X_va, y_va, X_te
    )

    print("Training finale:", X_finale.shape)
    print("Etichette training:", y_finale.shape)
    print("Test finale:", X_test_finale.shape)
    print("Attacchi nel training:", int(np.sum(y_finale == 1)))
    print("Attacchi nel test:", int(np.sum(y_te == 1)))
    print("Media training:", np.round(X_finale.mean(axis=0), 4))
    print("Deviazione standard:", np.round(X_finale.std(axis=0), 4))

    print("\nRIADDESTRAMENTO REGRESSIONE LOGISTICA")
    modello_log_finale = riaddestra_logistica(
        X_finale,
        y_finale,
        epoche=epoca_log
    )
    print("Osservazioni utilizzate:", len(y_finale))
    print("Epoche di addestramento:", epoca_log)
    print("Learning rate:", 0.01)
    print(
        "Parametri addestrabili:",
        sum(p.size for p in modello_log_finale.values())
    )

    # riaddestra la SVM usando C e gamma selezionati
    print("\nRIADDESTRAMENTO SVM")
    C_migliore, gamma_migliore = parametri_svm
    modello_svm_finale = addestra_svm_smo(
        X_finale,
        y_finale,
        C=C_migliore,
        gamma=gamma_migliore,
        max_iter=100000
    )

    print("Osservazioni utilizzate:", len(y_finale))
    print("C selezionato:", C_migliore)
    print("Gamma selezionato:", gamma_migliore)
    print("Aggiornamenti SMO:", modello_svm_finale["iterazioni"])
    print("Convergenza:", modello_svm_finale["convergente"])
    print("Violazione finale:", modello_svm_finale["violazione"])
    print("Vettori di supporto:", modello_svm_finale["support_vectors"])
    print("Duality gap finale:", modello_svm_finale["gap_duale"])

    # recupera lambda e il numero di epoche scelti per la rete
    print("\nRIADDESTRAMENTO RETE NEURALE")
    lambda_migliore = migliore_rete["lambda"]
    epoca_migliore_rete = migliore_rete["epoca"]

    # riaddestra la rete sui dati uniti
    modello_rete_finale = riaddestra_rete_l2(X_finale, y_finale, lam=lambda_migliore,epoche=epoca_migliore_rete)

    print("Osservazioni utilizzate:", len(y_finale))
    print(f"Lambda selezionato: {lambda_migliore:.8f}")
    print("Learning rate:", 0.003)
    print("Epoche di addestramento:", epoca_migliore_rete)
    print("Parametri addestrabili:", sum(p.size for p in modello_rete_finale.values()))

    # calcola le previsioni dei tre modelli sul test
    print("\nVALUTAZIONE FINALE")
    probabilita_log = forward_logistica(X_test_finale, modello_log_finale)
    previsioni_svm = predici_svm(X_test_finale, modello_svm_finale)
    probabilita_rete, _ = forward(X_test_finale, modello_rete_finale)

    # calcola le metriche sul test
    test_log = calcola_metriche(y_te, probabilita_log)
    test_svm = calcola_metriche(y_te, previsioni_svm)
    test_rete = calcola_metriche(y_te, probabilita_rete)

    # recupera le metriche della SVM sulla validation
    metriche_svm_val = calcola_metriche(y_va, predici_svm(X_va, svm_val))

    # raccoglie i risultati dei tre modelli per il confronto
    print("\nCONFRONTO FINALE DEI TRE MODELLI")
    risultati_finali = [
        ("Regressione logistica",
            f"alpha=0.01, ep={epoca_log}",
            1,
            metriche_log_val,
            test_log,
            "9 parametri",
            "-"
        ),
        (
            "SVM",
            f"C={C_migliore}, gamma={gamma_migliore}",
            25,
            metriche_svm_val,
            test_svm,
            f"{modello_svm_finale['support_vectors']} SV",
            f"{modello_svm_finale['gap_duale']:.6f}"
        ),
        (
            "Rete neurale",
            f"alpha=0.003, lambda={lambda_migliore:.6f}, ep={epoca_migliore_rete}",
            16,
            metriche_rete_val,
            test_rete,
            f"{sum(p.size for p in modello_rete_finale.values())} parametri",
            "-"
        )
    ]

    intestazione = (
        f"{'Modello':<23}"
        f"{'Iperparametri':<43}"
        f"{'Add.':>6}"
        f"{'F1 val':>9}"
        f"{'Accuracy':>10}"
        f"{'Precision':>11}"
        f"{'Recall':>9}"
        f"{'F1 test':>9}"
        f"{'FN':>5}"
        f"{'FP':>5}"
        f"{'Complessita':>16}"
        f"{'Duality gap':>14}"
    )
    print(intestazione)
    print("-" * len(intestazione))

    for nome, iperparametri, n_add, val, test, complessita, gap in risultati_finali:
        print(
            f"{nome:<23}"
            f"{iperparametri:<43}"
            f"{n_add:>6}"
            f"{val['F1']:>9.6f}"
            f"{test['Accuratezza']:>10.6f}"
            f"{test['Precisione']:>11.6f}"
            f"{test['Recall']:>9.6f}"
            f"{test['F1']:>9.6f}"
            f"{test['FN']:>5}"
            f"{test['FP']:>5}"
            f"{complessita:>16}"
            f"{gap:>14}"
        )

    print("\nIPERPARAMETRI E COMPLESSITA")
    print(f"Logistica: alpha=0.01, epoche={epoca_log}, parametri=9")
    print(f"SVM: C={C_migliore}, gamma={gamma_migliore}, "
          f"SV={modello_svm_finale['support_vectors']}")
    print(f"Rete: lambda={lambda_migliore:.8f}, "
          f"epoche={epoca_migliore_rete}, parametri=1377")

    print("\nADDESTRAMENTI SPESI NELLA SELEZIONE")
    print("Regressione logistica: 1")
    print("SVM: 25 nella griglia + 1 prova preliminare")
    print("Rete neurale: 8 sezione aurea + 8 griglia + 1 prova preliminare")

    print("\nDUALITY GAP FINALE SVM")
    print(f"{modello_svm_finale['gap_duale']:.6f}")

def main():
    """prepara i dati, seleziona i modelli e li valuta sul test"""

    X_tr, y_tr, X_va, y_va, X_te, y_te = prepara_dataset()

    _, epoca_log, metriche_log_val = esegui_logistica(X_tr, y_tr, X_va, y_va)
    svm_val, parametri_svm, _ = esegui_svm(X_tr, y_tr, X_va, y_va)
    migliore_rete, _, _, metriche_rete_val = esegui_rete_l2(X_tr, y_tr, X_va, y_va)

    esegui_punto_2b(X_tr, y_tr, X_va, y_va, X_te, y_te, epoca_log, metriche_log_val, svm_val, parametri_svm, migliore_rete, metriche_rete_val)

if __name__ == "__main__":
    main()