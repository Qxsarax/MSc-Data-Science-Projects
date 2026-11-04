import numpy as np

#BACKTRACKING DI ARMIJO

#questa funzione cerca un passo che garantisca una diminuzione sufficiente della funzione
def armijo(x, f, gradiente, direzione=None, grad=None,
           alpha_iniziale=1.0, beta=0.5, c=1e-4):

    #partiamo dal valore iniziale del passo
    alpha = alpha_iniziale

    #se il gradiente non è già disponibile, lo calcoliamo nel punto corrente
    if grad is None:
        grad = gradiente(x)

    #se non viene indicata una direzione, usiamo quella opposta al gradiente
    if direzione is None:
        direzione = -grad

    #controlliamo che la direzione scelta sia effettivamente di discesa
    if grad @ direzione >= 0:
        raise ValueError("La direzione non è di discesa")

    #riduciamo il passo finché non viene rispettata la condizione di armijo
    while f(x + alpha * direzione) > f(x) + c * alpha * (grad @ direzione):
        alpha = beta * alpha

    #restituiamo il passo trovato
    return alpha

#DISCESA DEL GRADIENTE

#funzione che esegue la discesa del gradiente in base al metodo scelto
def discesa_gradiente(x0, f, gradiente, metodo,
                     mu=None, L=None,
                     tolleranza=1e-8, max_iter=100000):

    #partiamo da una copia del punto iniziale e salviamo la prima posizione
    x = x0.copy()
    traiettoria = [x.copy()]

    #ripetiamo gli aggiornamenti fino al raggiungimento del limite di iterazioni
    for k in range(max_iter):

        #calcoliamo il gradiente nel punto corrente
        grad = gradiente(x)

        #ci fermiamo quando la norma del gradiente è inferiore alla tolleranza
        if np.linalg.norm(grad) < tolleranza:
            break

        #scegliamo il passo in base al metodo indicato
        if metodo == "costante":
            alpha = 1 / L

        elif metodo == "ottimo":
            alpha = 2 / (L + mu)

        elif metodo == "armijo":
            alpha = armijo(x, f, gradiente, grad=grad)

        #se il metodo non è riconosciuto, segnaliamo un errore
        else:
            raise ValueError("Metodo non riconosciuto")

        #aggiorniamo il punto muovendoci nella direzione opposta al gradiente
        x = x - alpha * grad

        #salviamo il nuovo punto per ricostruire la traiettoria
        traiettoria.append(x.copy())

    #restituiamo tutti i punti visitati sotto forma di array
    return np.array(traiettoria)


#METODO DI NEWTON SMORZATO

#funzione che applica il metodo di newton con ricerca del passo tramite armijo
def newton(x0, f, gradiente, hessiana,
           tolleranza=1e-8, max_iter=1000):

    #partiamo da una copia del punto iniziale e salviamo la prima posizione
    x = np.array(x0, dtype=float).copy()
    traiettoria = [x.copy()]

    #ripetiamo il procedimento fino al limite massimo di iterazioni
    for k in range(max_iter):

        #calcoliamo il gradiente nel punto corrente
        grad = gradiente(x)

        #interrompiamo il ciclo quando raggiungiamo la tolleranza richiesta
        if np.linalg.norm(grad) < tolleranza:
            break

        #calcoliamo l'hessiana nel punto corrente
        H = hessiana(x)

        #troviamo la direzione di newton risolvendo il sistema lineare
        try:
            direzione = np.linalg.solve(H, -grad)

        #se il sistema non può essere risolto, utilizziamo l'antigradiente
        except np.linalg.LinAlgError:
            direzione = -grad

        #verifichiamo che la direzione permetta di diminuire la funzione
        if grad @ direzione >= 0:
            direzione = -grad

        #cerchiamo un passo adatto utilizzando armijo
        alpha = armijo(x, f, gradiente, direzione=direzione, grad=grad)

        #aggiorniamo il punto utilizzando la direzione e il passo trovati
        x = x + alpha * direzione

        #salviamo il nuovo punto nella traiettoria
        traiettoria.append(x.copy())

    #restituiamo tutte le iterate ottenute
    return np.array(traiettoria)


#METODO QUASI-NEWTON BFGS

#funzione che applica bfgs approssimando l'inversa dell'hessiana
def bfgs(x0, f, gradiente, tolleranza=1e-8, max_iter=1000):

    #partiamo da una copia del punto iniziale
    x = np.array(x0, dtype=float).copy()

    #inizializziamo le matrici identità della stessa dimensione del problema
    n = len(x)
    H = np.eye(n)
    I = np.eye(n)

    #salviamo il punto iniziale nella traiettoria
    traiettoria = [x.copy()]

    #calcoliamo il gradiente iniziale, che verrà poi riutilizzato
    grad = gradiente(x)

    #ripetiamo il procedimento fino al limite massimo di iterazioni
    for k in range(max_iter):

        #ci fermiamo quando la norma del gradiente è inferiore alla tolleranza
        if np.linalg.norm(grad) < tolleranza:
            break

        #calcoliamo la direzione usando la matrice H
        direzione = -H @ grad

        #se la direzione non è valida o non è di discesa, usiamo l'antigradiente
        if not np.all(np.isfinite(direzione)) or grad @ direzione >= 0:
            direzione = -grad
            H = np.eye(n)

        #cerchiamo un passo che soddisfi la condizione di armijo
        alpha = armijo(x, f, gradiente, direzione=direzione, grad=grad)

        #calcoliamo il nuovo punto e il gradiente corrispondente
        x_nuovo = x + alpha * direzione
        grad_nuovo = gradiente(x_nuovo)

        #calcoliamo le variazioni del punto e del gradiente
        s = x_nuovo - x
        y = grad_nuovo - grad

        #calcoliamo il prodotto scalare necessario per l'aggiornamento
        ys = y @ s

        #aggiorniamo H solo se la condizione di curvatura è soddisfatta
        if ys > 1e-10 * np.linalg.norm(s) * np.linalg.norm(y):

            #calcoliamo i termini necessari per la formula di bfgs
            rho = 1 / ys
            V = I - rho * np.outer(s, y)

            #aggiorniamo l'approssimazione dell'inversa dell'hessiana
            H = V @ H @ V.T + rho * np.outer(s, s)

        #prepariamo il punto e il gradiente per l'iterazione successiva
        x = x_nuovo
        grad = grad_nuovo

        #aggiungiamo il nuovo punto alla traiettoria
        traiettoria.append(x.copy())

    #restituiamo tutte le iterate ottenute
    return np.array(traiettoria)


#OPERATORE PROSSIMALE DELLA NORMA L1 (SOFT-THRESHOLDING)

#questa funzione calcola il prox di t*lambda*||x||_1, cioè sign(v)*max(|v| - soglia, 0)
#la soglia da passare è il prodotto lambda*t
def soft_thresholding(v, soglia):

    # le componenti con modulo minore o uguale alla soglia vengono azzerate,
    # le altre vengono avvicinate a zero della quantità soglia
    return np.sign(v) * np.maximum(np.abs(v) - soglia, 0.0)


#METODO DEL GRADIENTE PROSSIMALE (ISTA)

#funzione che minimizza F = g + h facendo un passo di gradiente su g e un passo prossimale su h
#prox(v, t) deve restituire il prox di t*h calcolato in v
def ista(x0, gradiente_g, prox, L, tolleranza=1e-8, max_iter=10000):

    #partiamo da una copia del punto iniziale e salviamo la prima posizione
    x = np.array(x0, dtype=float).copy()
    traiettoria = [x.copy()]

    #usiamo il passo costante 1/L, con L costante di Lipschitz del gradiente di g
    alpha = 1 / L

    #ripetiamo gli aggiornamenti fino al limite massimo di iterazioni
    for k in range(max_iter):

        #passo di gradiente sulla parte liscia seguito dal passo prossimale sulla parte non liscia
        x_nuovo = prox(x - alpha * gradiente_g(x), alpha)

        #salviamo il nuovo punto nella traiettoria
        traiettoria.append(x_nuovo.copy())

        #il "gradient mapping" (x - x_nuovo)/alpha sostituisce il gradiente nel criterio di arresto:
        #si annulla esattamente nei punti di minimo di F
        if np.linalg.norm(x - x_nuovo) / alpha < tolleranza:
            break

        #prepariamo il punto per l'iterazione successiva
        x = x_nuovo

    #restituiamo tutte le iterate ottenute
    return np.array(traiettoria)


#METODO DEL GRADIENTE PROSSIMALE ACCELERATO (FISTA)

#funzione che applica ista nel punto estrapolato y invece che nell'ultima iterata
def fista(x0, gradiente_g, prox, L, tolleranza=1e-8, max_iter=10000):

    #partiamo da una copia del punto iniziale e salviamo la prima posizione
    x = np.array(x0, dtype=float).copy()
    traiettoria = [x.copy()]

    #inizialmente il punto estrapolato coincide con il punto iniziale e t vale 1
    y = x.copy()
    t = 1.0

    #usiamo lo stesso passo costante di ista
    alpha = 1 / L

    #ripetiamo gli aggiornamenti fino al limite massimo di iterazioni
    for k in range(max_iter):

        #passo prossimale calcolato nel punto estrapolato y
        x_nuovo = prox(y - alpha * gradiente_g(y), alpha)

        #salviamo il nuovo punto nella traiettoria
        traiettoria.append(x_nuovo.copy())

        #criterio di arresto basato sul gradient mapping valutato in y
        if np.linalg.norm(y - x_nuovo) / alpha < tolleranza:
            break

        #aggiorniamo il parametro t secondo la regola di Nesterov
        t_nuovo = (1 + np.sqrt(1 + 4 * t**2)) / 2

        #costruiamo il nuovo punto estrapolato aggiungendo un termine di "momento"
        y = x_nuovo + ((t - 1) / t_nuovo) * (x_nuovo - x)

        #prepariamo le variabili per l'iterazione successiva
        x = x_nuovo
        t = t_nuovo

    #restituiamo tutte le iterate ottenute
    return np.array(traiettoria)


#METODO DEL SUBGRADIENTE PER IL PROBLEMA LASSO

#funzione che sostituisce il gradiente di lambda*||x||_1 con il subgradiente lambda*sign(x)
def subgradiente(x0, gradiente_g, lam, L, max_iter=10000):

    #partiamo da una copia del punto iniziale e salviamo la prima posizione
    x = np.array(x0, dtype=float).copy()
    traiettoria = [x.copy()]

    #il metodo non ha un criterio di arresto affidabile, perché il subgradiente
    #non tende a zero: eseguiamo quindi un numero fissato di iterazioni
    for k in range(max_iter):

        #calcoliamo un subgradiente di F; np.sign(0) = 0 è un elemento valido di [-1, 1]
        sub = gradiente_g(x) + lam * np.sign(x)

        #passo decrescente alpha_k = 1/(L*sqrt(k+1))
        alpha = 1 / (L * np.sqrt(k + 1))

        #aggiorniamo il punto muovendoci nella direzione opposta al subgradiente
        x = x - alpha * sub

        #salviamo il nuovo punto nella traiettoria
        traiettoria.append(x.copy())

    #restituiamo tutte le iterate ottenute
    return np.array(traiettoria)