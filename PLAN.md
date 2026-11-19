# Proiect IAPD nr. 21 — Predicția reușitei undercut-ului vs. overcut-ului în Formula 1

**Autor:** Mihai Andrei-Dragoș
**Curs:** Învățare automată și procesarea datelor (IAPD), UPB

> **Status (19.11.2026):** Phase 1 e gata: 5.231 de dueluri, 36 de features, EDA și 5 baseline-uri. Rezultatele sunt în [README](README.md#rezultate-phase-1).
> **Schimbare față de planul inițial:** FastF1 nu mai poate fi folosit fără abonament F1TV (arhiva `livetiming.formula1.com` răspunde cu HTTP 401). L-am înlocuit cu Jolpica (timpi pe tur, poziții, opriri) și OpenF1 (anvelope din 2023). Perioadele de SC/VSC le deducem din timpii pe tur și le validăm cu mesajele oficiale din OpenF1. Secțiunile de mai jos descriu ce am implementat efectiv.

## 1. Cerințe (din cursul 00 Overview)

| Fază | Când | Puncte | Ce se cere |
|---|---|---|---|
| Phase 0 | săpt. 3 | – | echipă, scop, descriere, surse de date (făcut) |
| **Phase 1** | **săpt. 7–9** | **15p** | dataset propriu extras online: **≥1.000 rânduri, ≥10 features**; analizele planificate; **≥2 modele baseline**; prezentare de status înainte de curs |
| Phase 2 | săpt. 14 | 15p | EDA + rezultate de predicție; raport ≤10 pagini, structurat ca un articol științific |

Cod și documentație într-un **repository GitHub public**. Nu se acceptă dataseturi existente, deci datasetul de „dueluri” trebuie construit de noi.

## 2. Definiția problemei

**Unitatea de date = un duel de strategie (A, B):**
- Pilotul **A** intră la boxe în turul `L_A`.
- Pilotul **B** era la cel mult **6,0 s** de A (în față sau în spate) la finalul turului `L_A − 1`, nu a intrat în același tur și intră la boxe în turul `L_B`, cu `L_A < L_B ≤ L_A + 10`.
- Ambii termină turul `L_B + 1`, adică ciclul de opriri e complet.

**Țintă (binară):** `undercut_success = 1` dacă A (cel care oprește primul) e în fața lui B la finalul turului `L_B + 1`, comparând timpii de sesiune cumulați; `0` dacă a câștigat cel rămas pe pistă (overcut).

Păstrăm și `a_was_ahead`, ca să putem separa **atacurile reale** (A în spate) de **acoperiri** (A în față). În analiză le raportăm și împreună, și separat.

**Filtre / cazuri speciale:**
- SC/VSC/steag roșu în fereastra `[L_A−1, L_B+1]` → flag `neutralized`. Le excludem din datasetul principal și le păstrăm pentru o analiză separată.
- Abandon al unuia dintre piloți înainte de `L_B+1` → exclus.
- Opriri pentru penalizări sau aripi rupte (opriri anormal de lungi) → păstrate, cu flag `long_stop` (pot fi semnal real, conform cursului 02, secțiunea despre outlieri).
- Curse ude, cu treceri inter↔slick → păstrate, cu flag și features de compus.

## 3. Surse de date și integrare

| Sursă | Ce luăm | Cheie de join |
|---|---|---|
| **Jolpica-F1 API** | calendar, circuite (lat/long), rezultate, **timpi pe tur și poziții** (din care calculăm timpii cumulați, gap-urile, ritmul și SC/VSC), pit stops (turul și durata în pit lane) | (an, rundă, pilot, tur), `circuitId` |
| **OpenF1 API** (2023+) | stints: compusul și vârsta reală a anvelopelor; mesaje race control, folosite pentru validarea detecției SC/VSC | data cursei, numărul mașinii |
| **Open-Meteo Historical** | vreme orară la coordonatele circuitului, la ora cursei: temperatură, precipitații (inclusiv în orele dinainte, pentru pista udă), nori, vânt | lat/long din Jolpica + data și ora |
| **Wikipedia** | caracteristici circuite: lungime, nr. viraje, tip (stradal/permanent), sens | `circuitId` → mapare manuală la pagina Wikipedia |

Volum obținut: **173 de curse**, din care 171 au produs dueluri, adică **5.231 de dueluri** (~30 pe cursă). Riscul de a ieși sub 1.000 de rânduri nu s-a materializat.

## 4. Features (≥10; țintim ~25)

Le împărțim în două seturi, ca să evităm **data leakage**:

**A. Pre-decizie** (cunoscute în momentul în care A intră la boxe):
1. `gap_s`: distanța A–B la finalul turului `L_A−1`
2. `a_was_ahead`
3. `stint_age_A`, `stint_age_B`, `stint_age_delta`: tururi de la ultima oprire (toți anii); din 2023 și vârsta reală a anvelopelor (`tyre_age_*`, OpenF1)
4. `cmp_A_old`, `cmp_B_old` (ordinal: S < M < H; I, W separat), doar din 2023
5. `cmp_A_new`, doar din 2023
6. `pace_delta_3`: diferența de mediană pe ultimele 3 tururi „curate”
7. `pit_loss_circuit`: pierderea medie la boxe pe circuit (din cursele anterioare, ca să nu folosim informații din cursa curentă)
8. `traffic_at_exit`: câte mașini cad în fereastra estimată de reintrare a lui A
9. `laps_remaining`, `race_progress` (= `L_A / total tururi`)
10. `position_A`, `stop_number_A` (a câta oprire e)
11. Vreme (Open-Meteo): temperatură, umiditate, precipitații în cursă și în cele 3 ore dinainte, nori, vânt
12. Circuit: `circuit_length_km`, `n_turns`, `street_circuit` (Wikipedia), plus `overtake_difficulty`, calculat din date ca număr mediu de schimbări de poziție pe tur pe circuit
13. `season` (schimbările de regulament din 2022 contează)

**B. Post-hoc** (se cunosc doar după fapt; folosite doar pentru analiză și ablație):
- `lap_delay_B` (= `L_B − L_A`), `pit_time_A`, `pit_time_B`, `compound_B_new`, `outlap_delta`

## 5. Preprocesare (conform cursului 02)

- **Tipuri de atribute**: nominal (compus, circuit), ordinal (compus slick), ratio (gap, timpi), binar (flag-uri)
- **Valori lipsă**: tururi neprecise sau lipsă de vreme → imputare cu mediana pe cursă; raportăm procentul lipsă pe fiecare coloană
- **Outlieri**: IQR / boxplot pe `pit_time`; îi păstrăm cu flag (nu îi ștergem orb)
- **Integrare**: join pe chei; verificăm duplicatele (aceeași pereche numărată de două ori) și inconsistențele (Jolpica vs FastF1 la pit stops)
- **Transformare**: one-hot pentru nominal, z-score pentru kNN/SVM/MLP, features derivate (delta-uri)
- **Reducere**: corelații și redundanță (de ex. `race_progress` vs `laps_remaining`), PCA pentru vizualizare

## 6. Evaluare

- **Split:** 2018–2023 = train/validare cu **GroupKFold pe cursă** (perechile din aceeași cursă nu trebuie să ajungă și în train, și în test); **2024–2025 = test temporal** ținut deoparte
- **Metrici:** accuracy, balanced accuracy, F1, ROC-AUC, plus matrice de confuzie
- **Baseline-uri pentru Phase 1** (minimum 2):
  0. Dummy (clasa majoritară) și regula „ordinea se păstrează” (euristică de referință)
  1. **Gaussian Naïve Bayes**
  2. **Decision Tree** (adâncime limitată, interpretabil)
  3. (bonus) **kNN**
- **Phase 2:** toate cele 7 modele (NB, kNN, DT, SVM, RF, Gradient Boosting, MLP), tuning, importanța features-urilor (permutation importance), ablație pre-decizie vs post-hoc, analiză pe subseturi (atac vs acoperire, uscat vs ud, pre-2022 vs post-2022)

## 7. Structura repository-ului

```
f1-undercut-prediction/
├── README.md
├── requirements.txt
├── config.yaml               # sezoane, prag gap, max lap delay
├── src/
│   ├── sources/
│   │   ├── jolpica.py
│   │   ├── openf1.py
│   │   ├── openmeteo.py
│   │   └── wikipedia_circuits.py
│   ├── fetch_raw.py          # descarcă datele brute (cu cache, reluabil)
│   ├── duels.py              # extragerea duelurilor + etichetare
│   ├── features.py           # grupurile de features
│   ├── build_dataset.py      # integrare + features derivate -> duels.csv
│   ├── eda.py                # analiza exploratorie
│   ├── baselines.py          # modelele baseline
│   └── validate_sc.py        # validarea detecției SC/VSC
├── data/
│   ├── cache/                # cache FastF1 (gitignored)
│   └── processed/duels.csv   # datasetul final (versionat)
└── reports/phase1/           # slide-uri de status
```

## 8. Pași pentru Phase 1

1. ✅ Setup: mediu Python, `requirements.txt`, repo GitHub public
2. ✅ Clienți de date: Jolpica, OpenF1, Open-Meteo, Wikipedia
3. ✅ Descărcarea cu cache (`fetch_raw.py`); test pe Australia 2025 (cursă pe ploaie, cu SC)
4. ✅ Algoritmul de dueluri + etichetare; detecția SC/VSC validată cu mesajele OpenF1
5. ✅ Calculul features-urilor + join cu vremea și circuitele
6. ✅ Rulare completă pe 2018–2025 → `duels.csv` cu 5.231 de rânduri
7. ✅ EDA de bază: distribuția clasei, statistici descriptive, valori lipsă, corelații, outlieri, rata de succes pe circuit
8. ✅ Baseline-uri: Dummy, regula „ordinea se păstrează”, NB, Decision Tree, kNN (GroupKFold + test temporal)
9. ✅ Slide-uri de status pentru Phase 1 (9 slide-uri)
