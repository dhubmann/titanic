# Titanic Fallstudie

Die Fallstudie kann in Gruppen von bis zu drei Personen abgeschlossen werden und wird für alle
Gruppen am gleichen Datensatz (train und test Datensatz im Moodlekurs) ausgeführt. Die
Aufgabe besteht darin, vorherzusagen, welche Passagiere das Unglück auf der Titanic überleben.

Folgende Aufgabenstellungen sollen erfüllt werden:

- Beschreiben Sie die vorliegenden Daten in verbaler und visueller Form. Dazu ist es
  notwendig einige explorative Analyseschritte vorzunehmen.
- Untersuchen Sie den Datensatz auf ggfs. notwendige Vorverarbeitungsschritte und
  dokumentieren Sie diese.
- Wenden Sie mindestens drei verschiedene (müssen nicht aus der LV sein) Machine
  Learning Ansätze an und beschreiben Sie Ihr Vorgehen.
- Ein Ergebnis der Fallstudie ist eine Aussage darüber, welche Personen im Testdatensatz
  überleben und welche nicht.

Die Beurteilungsgrundlage für die Fallstudie bildet allerdings nicht die Qualität des Ergebnisses,
sondern die Qualität Ihrer Präsentation. Dazu zählt:

- Sind Analyseschritte und vorangegangene Überlegungen nachvollziehbar argumentiert?
  z.B. Ein K-Means Clustering wurde als explorative Methode eingesetzt, um Informationen
  über mögliche Segmentierungen innerhalb der Passagiergruppen zu erhalten.
- Sind Ergebnisse und Güte der Modelle nachvollziehbar dokumentiert? Siehe dazu Modell
  Performance in den LV Unterlagen.
- Sind die Hyperparameter der Modelle nachvollziehbar gewählt?
- Sind die Ergebnisse und Erkenntnisse aus den Modellen ansprechend präsentiert?

Abzugeben sind ein zusammenfassendes Dokument mit Ihren Begründungen und Überlegungen
sowie Ergebnissen (ca. 3-5 Seiten, inkl. Screenshots) und das jeweilige Code-File. Die Aufgaben
können in einer Programmiersprache Ihrer Wahl gelöst werden.

Die Fallstudie ist bis zum 17.10.2026 LV-Beginn fertigzustellen und wird an diesem Tag von jeder
Gruppe präsentiert. Die Präsentationszeit beträgt maximal 15 Minuten (anschließend ca. 10
Minuten Diskussion) und die Präsentationsform kann frei gewählt werden. Für die Beurteilung
wird ausschließlich Präsentation herangezogen.

Der Titanic Datensatz ist Grundlage für zahlreiche Challenges und Tutorials. Sie haben natürlich
die Möglichkeit, von bereits vorangegangenen Bearbeitungen zu profitieren. Einige gelungene
Versuche finden Sie hier:

- [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic)
- [Top-1 Titanic Solution](https://www.kaggle.com/code/nikitakudriashov/top-1-titanic-solution)

Zur Ausführung der Modelle und Analyseschritte können generative Modelle herangezogen
werden, bitte kennzeichnen Sie jene Stellen mit einer Fußnote und dem benutzten Prompt.

## How to start

- Download the data
- Understand the problem
- EDA (Exploratory Data Analysis)
- Train, tune and ensemble machine learning models

## How to improve your score

- Learn more about the data
- Experiment:
  - Design/create some new features
  - Try different preprocessing
  - Try different types of ML models
  - Combine multiple models (ensemble)
- Learn from other's code and ideas

## Data description

- survival: 0 = no, 1 = yes
- pclass: 1 = 1st, 2 = 2nd, 3 = 3rd class
- sex: male, female
- age: in years
- sibsp: # of sibling / spouses aboard the Titanic
- parch: # of parents / children aboard the Titanic
- ticket: ticket number
- fare: passenger fare (Fahrpreis)
- cabin: cabin number
- embarked: port of embarkation; C = Cherbourg, Q = Queenstown, S = Southampton
