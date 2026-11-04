import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from metodi import discesa_gradiente, newton, bfgs, ista, fista

# GENERAZIONE DEI DATI

rng = np.random.default_rng(41) # seed per riproducibilità
m, n = 250, 20 # numero di campioni e dimensione del problema
mu = 0.01 # parametro di regolarizzazione
x0 = np.zeros(n) # punto iniziale per tutti i metodi
tolleranza = 1e-8 # tolleranza per tutti i metodi

def sigmoide(t):
    """Calcola la sigmoide usando tanh, così da evitare overflow nell'esponenziale."""
    return 0.5 * (1 + np.tanh(t / 2))

# assegna alle colonne di A fattori di scala diversi tra 0.4 e 2
A = rng.normal(size=(m, n)) * np.linspace(0.4, 2.0, n)

# genera i coefficienti usati per assegnare le etichette
x_generatore = rng.normal(scale=0.5, size=n)

# assegna a ogni osservazione l'etichetta +1 con probabilità sigmoide(a_i^T x_generatore), e -1 altrimenti
y = np.where(rng.random(m) < sigmoide(A @ x_generatore), 1.0, -1.0)


# FUNZIONE, GRADIENTE, HESSIANA, COSTANTE DI LIPSCHITZ E PROX

def g(x):
    """Calcola la media della perdita logistica su tutti i campioni."""
    return np.mean(np.logaddexp(0.0, -y * (A @ x)))

def gradiente_g(x):
    """Calcola il gradiente della media della perdita logistica su tutti i campioni."""
    return -(A.T @ (y * sigmoide(-y * (A @ x)))) / m 

def hessiana_g(x):
    """Calcola l'hessiana della media della perdita logistica su tutti i campioni."""
    z = y * (A @ x) # vettore di dimensione m con i prodotti y_i * (a_i^T x)
    return (A.T * (sigmoide(z) * sigmoide(-z))) @ A / m 

def F(x):
    """Calcola la funzione obiettivo regolarizzata F(x) = g(x) + mu/2 ||x||^2."""
    return g(x) + 0.5 * mu * (x @ x)

def gradiente_F(x):
    """Calcola il gradiente della funzione obiettivo regolarizzata F(x) = g(x) + mu/2 ||x||^2."""
    return gradiente_g(x) + mu * x

def hessiana_F(x):
    """Calcola l'hessiana della funzione obiettivo regolarizzata F(x) = g(x) + mu/2 ||x||^2."""
    return hessiana_g(x) + mu * np.eye(n)

def prox_h(v, t):
    """Calcola il prox di h(x) = mu/2 ||x||^2, che è un semplice fattore di riduzione."""
    return v / (1.0 + mu * t)

# costante di Lipschitz del gradiente e passi usati dai metodi
L_g = np.linalg.norm(A, 2) ** 2 / (4 * m)  # limite di Lipschitz del gradiente della perdita logistica
L = L_g + mu  # limite corrispondente per la funzione regolarizzata

print("\nREGRESSIONE LOGISTICA REGOLARIZZATA")
print("-" * 70)
print(f"Dimensioni: m = {m}, n = {n}, mu = {mu}")
print(f"Etichette: +1 = {np.sum(y == 1)}, -1 = {np.sum(y == -1)}")
print(f"L_g = {L_g:.6f}, L = L_g + mu = {L:.6f}")
print(f"Passo del gradiente 1/L = {1 / L:.6f}, passo prossimale 1/L_g = {1 / L_g:.6f}")

# sceglie un punto casuale e un piccolo passo per verificare gradiente e hessiana
x_prova, passo = rng.normal(size=n), 1e-6

# approssima il gradiente con differenze finite centrate lungo le coordinate
grad_df = np.array([(F(x_prova + passo * e) - F(x_prova - passo * e)) / (2 * passo) for e in np.eye(n)])

# approssima la hessiana calcolando le variazioni del gradiente nelle stesse direzioni
hess_df = np.array([(gradiente_F(x_prova + passo * e) - gradiente_F(x_prova - passo * e)) / (2 * passo)
                    for e in np.eye(n)])

# verifica la condizione di ottimo del problema prossimale nel punto calcolato
v_prova, t_prova = rng.normal(size=n), 0.7
u_prova = prox_h(v_prova, t_prova)

print("\nVERIFICHE")
print("-" * 70)
print(f"Errore relativo del gradiente: "
      f"{np.linalg.norm(grad_df - gradiente_F(x_prova)) / np.linalg.norm(gradiente_F(x_prova)):.2e}")
print(f"Errore relativo dell'hessiana: "
      f"{np.linalg.norm(hess_df - hessiana_F(x_prova)) / np.linalg.norm(hessiana_F(x_prova)):.2e}")
print(f"Prox: norma del gradiente del problema prossimale = "
      f"{np.linalg.norm(mu * t_prova * u_prova + u_prova - v_prova):.2e}")

#SOLUZIONE DI RIFERIMENTO
x_star = newton(x0, F, gradiente_F, hessiana_F, tolleranza=1e-13, max_iter=100) # soluzione di riferimento con tolleranza molto stretta
n_iter_rif = len(x_star) - 1 # numero di iterazioni di Newton per arrivare a x_star
x_star = x_star[-1] # punto finale della traiettoria di Newton, usato come soluzione di riferimento
residuo_rif = np.linalg.norm(gradiente_F(x_star))

# se Newton non raggiunge la precisione richiesta, il confronto non è affidabile
if residuo_rif >= 1e-12:
    raise RuntimeError("Soluzione di riferimento non sufficientemente accurata")

# autovalori delle hessiane nel minimo, usati per i fattori di convergenza teorici
autovalori_F = np.linalg.eigvalsh(hessiana_F(x_star))
autovalori_g = np.linalg.eigvalsh(hessiana_g(x_star))

print("\nSOLUZIONE DI RIFERIMENTO (NEWTON, TOLLERANZA 1e-13)")
print("-" * 70)
print(f"Iterazioni: {n_iter_rif}, F(x*) = {F(x_star):.12f}")
print(f"||grad F(x*)|| = {residuo_rif:.2e}, quindi ||x* - x_esatto|| <= ||grad F|| / mu = {residuo_rif / mu:.2e}")
print(f"Autovalori di hess F(x*): tra {autovalori_F.min():.5f} e {autovalori_F.max():.5f}, "
      f"condizionamento {autovalori_F.max() / autovalori_F.min():.2f}")

#ESECUZIONE DEI SEI METODI

# ogni metodo riceve le funzioni che deve usare; per i prossimali il "gradiente" è quello di g
def esegui(nome, f, grad_F, hess_F, grad_g):
    """Esegue il metodo indicato da nome, con le funzioni fornite come argomenti."""

    if nome == "Gradiente 1/L":
        return discesa_gradiente(x0, f, grad_F, "costante", L=L, tolleranza=tolleranza, max_iter=100000)
    if nome == "Gradiente Armijo":
        return discesa_gradiente(x0, f, grad_F, "armijo", tolleranza=tolleranza, max_iter=100000)
    if nome == "Newton":
        return newton(x0, f, grad_F, hess_F, tolleranza=tolleranza)
    if nome == "BFGS":
        return bfgs(x0, f, grad_F, tolleranza=tolleranza)
    if nome == "Gradiente prossimale":
        return ista(x0, grad_g, prox_h, L_g, tolleranza=tolleranza, max_iter=100000)
    if nome == "FISTA":
        return fista(x0, grad_g, prox_h, L_g, tolleranza=tolleranza, max_iter=100000)
    raise ValueError("Metodo non riconosciuto")


nomi = ["Gradiente 1/L", "Gradiente Armijo", "Newton", "BFGS", "Gradiente prossimale", "FISTA"]
ripetizioni = 15
risultati = {}

for nome in nomi:
    # prima esecuzione con le funzioni "contate", per la traiettoria e il numero di valutazioni
    contatori = {"F": 0, "gradiente": 0, "hessiana": 0}

    def conta(funzione, chiave):
        """Incrementa il contatore associato alla chiave e chiama la funzione originale."""
        def funzione_contata(x):
            contatori[chiave] += 1
            return funzione(x)
        return funzione_contata

    # esegue il metodo con le funzioni contate per ottenere la traiettoria e i contatori
    traiettoria = esegui(nome, conta(F, "F"), conta(gradiente_F, "gradiente"),
                         conta(hessiana_F, "hessiana"), conta(gradiente_g, "gradiente"))

    # tempo: minimo su più esecuzioni senza contatori, la misura meno disturbata dal sistema
    tempi = []
    for _ in range(ripetizioni):
        # esegue il metodo senza contatori per misurare il tempo
        inizio = time.perf_counter()
        esegui(nome, F, gradiente_F, hessiana_F, gradiente_g)
        tempi.append(time.perf_counter() - inizio)

    risultati[nome] = {
        "traiettoria": traiettoria,
        "errori": np.linalg.norm(traiettoria - x_star, axis=1),
        "tempo": min(tempi),
        "contatori": contatori
    }


# GRAFICO SEMILOGARITMICO DELL'ERRORE

stili = {
    "Gradiente 1/L": ("tab:blue", "-"),
    "Gradiente Armijo": ("tab:cyan", "-"),
    "Newton": ("tab:red", "-"),
    "BFGS": ("tab:purple", "-"),
    "Gradiente prossimale": ("tab:orange", "--"),
    "FISTA": ("tab:green", "-")
}

plt.figure(figsize=(10, 6))
for nome, risultato in risultati.items():
    # mostra solo le iterazioni con errore positivo
    errori = risultato["errori"]
    colore, linea = stili[nome]
    indici = np.flatnonzero(errori > 0)

    # marker solo per i metodi con poche iterazioni
    plt.semilogy(indici, errori[indici], linea, color=colore, linewidth=1.8,
                 marker="o" if len(errori) < 100 else None, markersize=4,
                 label=f"{nome} ({len(errori) - 1} iterazioni)")

plt.xlabel("Numero di iterazioni")
plt.ylabel(r"Errore $e_k = \|x_k - x^*\|$")
plt.title("Regressione logistica regolarizzata - Confronto dei sei metodi")
plt.grid(True, which="both", alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig("grafici/parte4_convergenza.png", dpi=300)
plt.show()


# ORDINE DI CONVERGENZA

def ordini_stimati(errori):
    """Calcola gli ordini stimati p_k = log(e_k+1/e_k) / log(e_k/e_k-1) a partire dalla sequenza di errori."""
    return np.log(errori[2:] / errori[1:-1]) / np.log(errori[1:-1] / errori[:-2])

def massimi_locali(errori):
    """Restituisce gli indici dei massimi locali della sequenza di errori e i valori corrispondenti."""
    indici = np.flatnonzero((errori[1:-1] > errori[:-2]) & (errori[1:-1] > errori[2:])) + 1
    return indici, errori[indici]

def fattore_medio(errori, indici, i, j):
    """Calcola il fattore medio di riduzione dell'errore per iterazione tra le iterate i e j."""
    return (errori[j] / errori[i]) ** (1 / (indici[j] - indici[i]))

# stime teoriche dei fattori di convergenza per i metodi con passo costante
fattori_teorici = {
    "Gradiente 1/L": np.max(np.abs(1 - autovalori_F / L)),
    "Gradiente Armijo": np.max(np.abs(1 - autovalori_F)),
    "Gradiente prossimale": np.max((L_g - autovalori_g) / (L_g + mu))
}

print("\nORDINE DI CONVERGENZA")
print("-" * 70)

for nome in ["Newton", "BFGS"]:
    # stima dell'ordine di convergenza nelle ultime cinque iterazioni
    errori = risultati[nome]["errori"]
    p = ordini_stimati(errori)
    print(f"{nome}, ultimi p_k: " + ", ".join(f"{v:.3f}" for v in p[-5:]))

# stima del fattore di riduzione medio per BFGS tra le iterazioni 20-40 e nelle ultime 5
rapporti_bfgs = risultati["BFGS"]["errori"][1:] / risultati["BFGS"]["errori"][:-1]

print(f"BFGS, rapporto medio e_k+1/e_k: iterazioni 20-40 = {np.exp(np.mean(np.log(rapporti_bfgs[20:40]))):.3f}, "
      f"ultime 5 = {np.exp(np.mean(np.log(rapporti_bfgs[-5:]))):.3f}")

indici_picchi, picchi = massimi_locali(risultati["FISTA"]["errori"])
aumenti_fista = np.sum(np.diff(risultati["FISTA"]["errori"]) > 0)
print(f"FISTA: l'errore aumenta in {aumenti_fista} iterazioni su {len(risultati['FISTA']['errori']) - 1}, "
      f"{len(picchi)} massimi locali a distanza media di {np.mean(np.diff(indici_picchi)):.1f} iterazioni")

# TABELLA FINALE
print("\nTABELLA DI CONFRONTO")
print("=" * 118)
print(f"{'Metodo':<22}{'Iter.':<7}{'Tempo [ms]':<12}{'ms/iter':<9}{'Val. F':<8}{'Val. grad':<11}"
      f"{'Val. hess':<11}{'Ordine (1)':<12}{'Fatt. misurato':<16}{'Fatt. teorico'}")
print("-" * 118)

# salva i risultati in un file CSV
righe_csv = []
for nome, risultato in risultati.items():
    errori = risultato["errori"]
    iterazioni = len(errori) - 1
    c = risultato["contatori"]
    if nome == "FISTA":
        # per FISTA, usa solo i massimi locali per stimare l'ordine e il fattore
        sequenza, posizioni = picchi, indici_picchi
    else:
        # per gli altri metodi, usa tutti gli errori
        sequenza, posizioni = errori, np.arange(len(errori))

    p = ordini_stimati(sequenza) # calcola la sequenza degli ordini di convergenza stimati
    terzo = max(3, len(p) // 3) # considera almeno tre valori della parte finale della sequenza
    ordine = np.median(p[-terzo:]) # stima l'ordine come mediana dei valori selezionati

    misurato = "-" if nome in ["Newton", "BFGS"] else \
        f"{fattore_medio(sequenza, posizioni, -terzo - 1, -1):.5f}"
    
    if nome == "BFGS":
        misurato = f"{fattore_medio(sequenza, posizioni, -terzo - 1, -1):.3f}"
    teorico = f"{fattori_teorici[nome]:.5f}" if nome in fattori_teorici else "-" 
    
    print(f"{nome:<22}{iterazioni:<7}{1000 * risultato['tempo']:<12.2f}"
          f"{1000 * risultato['tempo'] / iterazioni:<9.4f}{c['F']:<8}{c['gradiente']:<11}{c['hessiana']:<11}"
          f"{ordine:<12.2f}{misurato:<16}{teorico}")

    righe_csv.append({
        "Metodo": nome, "Iterazioni": iterazioni, "Tempo [ms]": f"{1000 * risultato['tempo']:.3f}",
        "Valutazioni F": c["F"], "Valutazioni gradiente": c["gradiente"], "Valutazioni Hessiana": c["hessiana"],
        "Ordine stimato": f"{ordine:.3f}", "Fattore misurato": misurato, "Fattore teorico": teorico,
        "Errore finale": f"{errori[-1]:.3e}"
    })

with open("risultati_pt4.csv", "w", newline="", encoding="utf-8-sig") as file:
    scrittore = csv.DictWriter(file, fieldnames=list(righe_csv[0]))
    scrittore.writeheader()
    scrittore.writerows(righe_csv)

print("\nTabella salvata in risultati_pt4.csv")


# VERIFICA CON SCIPY
# verifica della correttezza di BFGS confrontando con l'implementazione di scipy.optimize.minimize
try:
    from scipy.optimize import minimize
    iterate_scipy = [] # lista per salvare le iterazioni di scipy
    risultato_scipy = minimize(F, x0, jac=gradiente_F, method="BFGS", options={"gtol": tolleranza},
                               callback=lambda xk: iterate_scipy.append(xk.copy()))
    errori_scipy = np.linalg.norm(np.array(iterate_scipy) - x_star, axis=1)
    print(f"\nVerifica BFGS: scipy {risultato_scipy.nit} iterazioni, nostro {len(risultati['BFGS']['errori']) - 1}")
    print("Ultimi rapporti e_k+1/e_k di scipy:", np.round(errori_scipy[-5:] / errori_scipy[-6:-1], 3))

except ImportError:
    print("\nscipy non installato: verifica di BFGS saltata")