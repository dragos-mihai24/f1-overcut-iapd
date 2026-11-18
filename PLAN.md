# Proiect IAPD nr. 21 — Predicția reușitei undercut-ului vs. overcut-ului în Formula 1

**Autor:** Mihai Andrei-Dragoș
**Curs:** Învățare automată și procesarea datelor (IAPD), UPB

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
| **FastF1** (live timing, 2018+) | tururi: `Time`, `LapTime`, `Stint`, `Compound`, `TyreLife`, `PitInTime/PitOutTime`, `TrackStatus`, `Position`; vreme de sesiune: AirTemp, TrackTemp, Humidity, Rainfall, Wind | (an, rundă, pilot, tur) |
| **Jolpica-F1 API** | calendar, circuite (lat/long), rezultate, pit stops (durata în pit lane), grilă | (an, rundă), `circuitId` |
| **Open-Meteo Historical** | vreme orară la coordonatele circuitului, la ora cursei: temperatură, precipitații (inclusiv în orele dinainte, pentru pista udă), nori, vânt | lat/long din Jolpica + data și ora |
| **Wikipedia** | caracteristici circuite: lungime, nr. viraje, tip (stradal/permanent), sens | `circuitId` → mapare manuală la pagina Wikipedia |

Volum estimat: 8 sezoane × ~22 curse ≈ **173 curse**. La ~6–10 dueluri pe cursă rezultă **~1.000–1.700 rânduri**.
**Risc:** dacă ieșim sub 1.000, incluzim și cursele sprint (2021+) și/sau facem o analiză de sensibilitate la pragul de 8 s (cu 6 s ca setare principală).

## 4. Features (≥10; țintim ~25)

Le împărțim în două seturi, ca să evităm **data leakage**:

**A. Pre-decizie** (cunoscute în momentul în care A intră la boxe):
1. `gap_s`: distanța A–B la finalul turului `L_A−1`
2. `a_was_ahead`
3. `tyre_age_A`, `tyre_age_B`, `tyre_age_delta`
4. `compound_A_old`, `compound_B_old` (ordinal: S < M < H; I, W separat)
5. `compound_A_new`
6. `pace_delta_3`: diferența de mediană pe ultimele 3 tururi „curate”
7. `pit_loss_circuit`: pierderea medie la boxe pe circuit (din cursele anterioare, ca să nu folosim informații din cursa curentă)
8. `traffic_at_exit`: câte mașini cad în fereastra estimată de reintrare a lui A
9. `laps_remaining`, `race_progress` (= `L_A / total tururi`)
10. `position_A`, `stop_number_A` (a câta oprire e)
11. Vreme: `air_temp`, `track_temp`, `humidity`, `rainfall`, `wind_speed` (FastF1) și `precip_prev_3h`, `cloud_cover` (Open-Meteo)
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
│   │   ├── fastf1_loader.py
│   │   ├── jolpica.py
│   │   ├── openmeteo.py
│   │   └── wikipedia_circuits.py
│   ├── duels.py              # extragerea duelurilor + etichetare
│   ├── features.py
│   └── build_dataset.py      # CLI: rulează tot pipeline-ul
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_baselines.ipynb
├── data/
│   ├── cache/                # cache FastF1 (gitignored)
│   └── processed/duels.csv   # datasetul final (versionat)
└── reports/phase1/           # slide-uri de status
```

## 8. Pași pentru Phase 1

1. Setup: mediu Python, `requirements.txt`, repo GitHub public
2. Clienți de date: Jolpica (calendar + circuite), Open-Meteo, Wikipedia (tabel circuite)
3. Loader FastF1 cu cache; rulare de test pe o singură cursă (de ex. Bahrain 2024)
4. Algoritmul de dueluri + etichetare, validat manual pe 2–3 curse cunoscute
5. Calculul features-urilor + join cu vremea și circuitele
6. Rulare completă pe 2018–2025 (în fundal, cu cache) → `duels.csv`; verificăm ≥1.000 rânduri
7. EDA de bază: distribuția clasei, statistici descriptive, valori lipsă, corelații, rata de succes pe circuit și pe compus
8. Baseline-uri: Dummy, NB, Decision Tree (+ kNN) cu GroupKFold
9. Slide-uri de status pentru Phase 1 (5–7 slide-uri)
