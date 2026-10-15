---
marp: true
theme: default
paginate: true
header: "Titanic Fallstudie – Gradient Boosting (LightGBM / HistGradientBoosting)"
footer: "Patrik | DSM Fallstudie WS 2026/27"
style: |
  section {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background-color: #f8fafc;
    color: #1e293b;
    padding: 40px;
  }
  h1 { color: #0f172a; font-size: 1.8rem; border-bottom: 2px solid #0d9488; padding-bottom: 8px; }
  h2 { color: #0d9488; font-size: 1.3rem; margin-top: 10px; }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
  .card { background: white; padding: 18px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); border: 1px solid #e2e8f0; }
  .badge { background: #0d9488; color: white; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.85rem; }
  .metric-box { text-align: center; background: #f0fdfa; border: 1px solid #ccfbf1; padding: 12px; border-radius: 6px; }
  .metric-val { font-size: 1.6rem; font-weight: 700; color: #0f766e; }
  .metric-label { font-size: 0.85rem; color: #475569; }
---

<!-- Slide 1 -->
# 1. Gradient Boosting: Konzept & Modellwahl

<div class="grid-2">
<div class="card">
  <h2>Ensemble-Prinzip</h2>
  <ul>
    <li><b>Sequenzielles Boosting:</b> Jeder neue Baum lernt gezielt aus den Fehlern (Residuen) der vorangegangenen Bäume.</li>
    <li><b>Histogramm-Binning:</b> Diskrete Bins für kontinuierliche Merkmale beschleunigen das Split-Finding und wirken als inhärente Regularisierung.</li>
    <li><b>LightGBM-Architektur:</b> Implementiert via scikit-learns nativem <code>HistGradientBoostingClassifier</code>.</li>
  </ul>
</div>

<div class="card">
  <h2>Architektur: Lose Kopplung</h2>
  <ul>
    <li><b>Isolierte Pipeline:</b> Vollständige Entkopplung vom restlichen Code via <code>clone(preprocessing)</code>.</li>
    <li><b>Keine globalen Seiteneffekte:</b> Bestehende Variablen und Modelle anderer Gruppenmitglieder (LogReg, SVM, RF) bleiben unberührt.</li>
    <li><b>100% Plattform-stabilität:</b> Keine externen C++ / OpenMP-Treiberprobleme auf macOS, Linux oder Windows.</li>
  </ul>
</div>
</div>

---

<!-- Slide 2 -->
# 2. Pipeline-Architektur & Hyperparameter-Design

<div class="grid-2">
<div class="card">
  <h2>End-to-End Pipeline</h2>
  <p><b>Numerische Merkmale:</b> <code>Age</code>, <code>Fare</code>, <code>FamilySize</code></p>
  <ul>
    <li><code>SimpleImputer(strategy='median')</code></li>
    <li><code>StandardScaler()</code></li>
  </ul>
  <p><b>Kategoriale Merkmale:</b> <code>Sex</code>, <code>Pclass</code>, <code>Embarked</code></p>
  <ul>
    <li><code>SimpleImputer(strategy='most_frequent')</code></li>
    <li><code>OneHotEncoder(handle_unknown='ignore')</code></li>
  </ul>
</div>

<div class="card">
  <h2>Begründung der Hyperparameter</h2>
  <ul>
    <li><code>learning_rate = 0.05</code>: Moderate Schrittweite (Shrinkage) gegen vorzeitiges Overfitting auf ~712 Trainingszeilen.</li>
    <li><code>max_iter = 100</code>: Ausreichend Boosting-Runden bei gedämpfter Lernrate.</li>
    <li><code>max_leaf_nodes = 15</code>: Beschränkt Baumkomplexität (Tiefe ~4); verhindert Auswendiglernen von Rauschen.</li>
    <li><code>min_samples_leaf = 10</code>: Glättung & stabile Statistiken in den Endblättern.</li>
    <li><code>l2_regularization = 1.0</code>: Dämpft extreme Blattgewichtungen.</li>
  </ul>
</div>
</div>

---

<!-- Slide 3 -->
# 3. Validierungsergebnisse & Modellgüte

<div class="grid-2">
<div class="card">
  <h2>Leistungsmetriken (Val: 179 Passagiere)</h2>
  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px;">
    <div class="metric-box">
      <div class="metric-val">79.9 %</div>
      <div class="metric-label">Accuracy</div>
    </div>
    <div class="metric-box">
      <div class="metric-val">0.839</div>
      <div class="metric-label">ROC-AUC</div>
    </div>
    <div class="metric-box">
      <div class="metric-val">0.719</div>
      <div class="metric-label">F1-Score</div>
    </div>
    <div class="metric-box">
      <div class="metric-val">78.0 %</div>
      <div class="metric-label">Precision (Überlebt)</div>
    </div>
  </div>
</div>

<div class="card">
  <h2>Confusion Matrix & Interpretation</h2>
  <table style="width: 100%; border-collapse: collapse; font-size: 0.9rem; margin-top: 8px;">
    <tr style="background: #f1f5f9; border-bottom: 1px solid #cbd5e1;">
      <th style="padding: 6px;">Klasse</th>
      <th style="padding: 6px;">Vorhergesagt: 0</th>
      <th style="padding: 6px;">Vorhergesagt: 1</th>
    </tr>
    <tr style="border-bottom: 1px solid #e2e8f0;">
      <td style="padding: 6px;"><b>Tatsächlich 0 (110)</b></td>
      <td style="padding: 6px; color: #047857; font-weight: 700;">97 (Richtig)</td>
      <td style="padding: 6px; color: #b91c1c;">13 (FP)</td>
    </tr>
    <tr>
      <td style="padding: 6px;"><b>Tatsächlich 1 (69)</b></td>
      <td style="padding: 6px; color: #b91c1c;">23 (FN)</td>
      <td style="padding: 6px; color: #047857; font-weight: 700;">46 (Richtig)</td>
    </tr>
  </table>
  <ul style="font-size: 0.85rem; margin-top: 12px; color: #334155;">
    <li>Hohe Trennschärfe (ROC-AUC 0.839) über alle Entscheidungsschwellen.</li>
    <li>Konservatives Modell: Bevorzugt sichere Überlebensvorhersagen (hohe Precision).</li>
  </ul>
</div>
</div>

---

<!-- Slide 4 -->
# 4. Feature Importance & Testdatensatz-Vorhersage

<div class="grid-2">
<div class="card">
  <h2>Permutation Feature Importance</h2>
  <p style="font-size: 0.85rem; color: #64748b;">Mittlerer Genauigkeitsverlust bei zufälliger Permutation:</p>
  <ul style="font-size: 0.9rem; line-height: 1.5;">
    <li>🥇 <b>Sex: +18.4 %</b> — Überragender Einfluss ("Frauen zuerst")</li>
    <li>🥈 <b>Pclass: +4.6 %</b> — Hohe Überlebenschancen 1. Klasse</li>
    <li>🥉 <b>Age: +4.4 %</b> — Schutz von Kindern</li>
    <li><b>FamilySize: +2.1 %</b> — Moderate Familiengrößen bevorzugt</li>
    <li><b>Fare: +2.0 %</b> & <b>Embarked: -0.5 %</b> — Geringerer Einfluss</li>
  </ul>
</div>

<div class="card">
  <h2>Testdaten-Vorhersage (test.csv)</h2>
  <ul>
    <li><b>418 Testpassagiere</b> erfolgreich klassifiziert.</li>
    <li><b>Vorhergesagte Überlebensrate: 33.5 %</b> (entspricht der historischen Verteilung).</li>
    <li>Ergebnisdatei exportiert: <code>predictions_gradient_boosting.csv</code> (Spalten: <code>PassengerId</code>, <code>Survived</code>, <code>SurvivalProbability</code>).</li>
    <li><b>Fazit:</b> Stabiles, interpretierbares und robustes Benchmark-Modell für den Gesamtvergleich.</li>
  </ul>
</div>
</div>
