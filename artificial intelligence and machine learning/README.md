# 🧬 Carcinoma Classification — Exam Project

## Challenge Website
https://www.kaggle.com/competitions/oxml-carinoma-classification  

---

##  Project Context

Questo progetto è stato sviluppato come elaborato d’esame nell’ambito di Machine Learning / Data Science.

L’obiettivo è la classificazione di immagini istopatologiche colorate con tecnica **HES (Hematoxylin–Eosin–Saffron)** al fine di identificare la presenza di cellule carcinomatose e, in caso positivo, determinarne la natura.

Il problema è formulato come una **classificazione multi-classe a tre categorie**:

- **Carcinoma neg (-1)** → Assenza di carcinoma  
- **Carcinoma pos, benign (0)** → Presenza di carcinoma benigno  
- **Carcinoma pos, malignant (1)** → Presenza di carcinoma maligno  

---

##  Dataset

Il dataset fornito dalla challenge presenta le seguenti caratteristiche:

- **186 immagini totali**
- **62 immagini etichettate**
- **124 immagini non etichettate**

La presenza di un numero limitato di label rende il problema particolarmente interessante dal punto di vista metodologico, in quanto richiede:

- Attenta gestione della validazione
- Controllo dell’overfitting
- Possibile utilizzo di tecniche di data augmentation
- Eventuale esplorazione di approcci semi-supervised

---

##  Obiettivi del Progetto

- Analisi esplorativa del dataset
- Implementazione pipeline di preprocessing
- Addestramento e confronto di diversi modelli di classificazione
- Valutazione delle performance tramite metriche appropriate
- Generazione della submission conforme alle specifiche del challenge website

---

## Nota

Questo progetto è stato realizzato esclusivamente a fini accademici.
Non ha validità clinica e non deve essere utilizzato per scopi diagnostici.
