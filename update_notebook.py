import os
import sys

try:
    import nbformat
except ImportError:
    os.system(f"{sys.executable} -m pip install nbformat")
    import nbformat

notebook_path = 'c:/Users/berta/Desktop/BERTA/UNI/Segon/Q4/PLH/laboratori/practica1/PLH_ident_idioma/projecte1.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

code_cells = [c for c in nb.cells if c.cell_type == 'code']

code_cells[0].source = '''import re
import os
import math
import pandas as pd
import nltk
from nltk.collocations import TrigramCollocationFinder
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns'''

code_cells[1].source = '''class LanguagePreprocessor:
    def __init__(self, idiomes):
        self.idiomes = idiomes

    def clean_text(self, frases):
        neteja = []
        for linia in frases:
            text = re.sub(r'\\d+', '', linia)
            text = text.lower()
            text = re.sub(r'\\s+', ' ', text).strip()
            if text:
                neteja.append(text)
        return "  " + "  ".join(neteja) + "  "

    def process_files(self, tipus_fitxers=['trn', 'tst']):
        for idioma in self.idiomes:
            for tipus in tipus_fitxers:
                nom_fitxer = f"{idioma}_{tipus}.txt"
                if os.path.exists(nom_fitxer):
                    print(f"Processant {nom_fitxer}...")
                    with open(nom_fitxer, 'r', encoding='utf-8') as f:
                        linies = f.readlines()
                    text_net = self.clean_text(linies)
                    nom_sortida = f"{idioma}_{tipus}_clean.txt"
                    with open(nom_sortida, 'w', encoding='utf-8') as f_out:
                        f_out.write(text_net)
                    print(f"Desat: {nom_sortida}")

    def obtenir_frases_netes(self, fitxer):
        with open(fitxer, 'r', encoding='utf-8') as f:
            text = f.read()
        return [frase for frase in text.split("  ") if len(frase.strip()) > 0]

# Execució del preprocessament
idiomes = ['deu', 'eng', 'fra', 'ita', 'nld', 'spa'] 
preprocessor = LanguagePreprocessor(idiomes)
preprocessor.process_files()'''

code_cells[2].source = '''class LanguageModel:
    def __init__(self, idiomes):
        self.idiomes = idiomes
        self.models_uni = {}
        self.models_bi = {}
        self.models_tri = {}
        self.N_uni = {}
        self.N_bi = {}
        self.N_tri = {}
        self.recomptes = {}
        self.vocabulari_global = set()
        self.B_global = 0

    def train(self, data_dict):
        """
        data_dict: dict amb format {idioma: string_amb_tot_el_text}
        """
        self.vocabulari_global = set()
        
        for idioma in self.idiomes:
            text_train = data_dict[idioma]
            
            # Models d'unigrames i bigrames
            llista_chars = list(text_train)
            self.models_uni[idioma] = nltk.FreqDist(llista_chars)
            self.models_bi[idioma] = nltk.FreqDist(nltk.ngrams(llista_chars, 2))
            
            self.N_uni[idioma] = len(llista_chars)
            self.N_bi[idioma] = len(llista_chars) - 1
            
            # Model principal de trigrames
            finder = TrigramCollocationFinder.from_words(text_train)
            
            # Afegim tots els trigrames trobats al vocabulari global abans de filtrar
            for trigramma in finder.ngram_fd.keys():
                self.vocabulari_global.add(trigramma)
                
            # Apliquem el filtre de freqüència >= 5 
            finder.apply_freq_filter(5)
            model = finder.ngram_fd
            
            self.models_tri[idioma] = model
            self.N_tri[idioma] = sum(model.values()) 
            self.recomptes[idioma] = len(model) 
            print(f"Model per '{idioma}' generat amb èxit. Trigrammes únics: {len(model)}")

        # Càlcul de B global
        self.B_global = len(self.vocabulari_global)
        print("\\nTots els models d'entrenament han estat generats.")

    def _get_trigram_fd(self, frase):
        finder = TrigramCollocationFinder.from_words("  " + frase + "  ")
        return finder.ngram_fd

    def calcul_probabilitat(self, frase, idioma, parametre, tecnica):
        # Generem els trigrames de la frase
        trigrames_frase = self._get_trigram_fd(frase)
        
        model3 = self.models_tri[idioma]
        N = self.N_tri[idioma] 
        V = self.recomptes[idioma] 
        
        log_p = 0.0
        
        for trig, freq in trigrames_frase.items():
            c3 = model3.get(trig, 0)
            
            if tecnica == 'lidstone':
                lamb = parametre
                p = (c3 + lamb) / (N + self.B_global * lamb)
                
            elif tecnica == 'absolute_discounting':
                delta = parametre
                N0 = self.B_global - V 
                if c3 > 0:
                    p = (c3 - delta) / N
                else:
                    p = ((self.B_global - N0) * delta / N0) / N
                    
            elif tecnica == 'interpolation':
                lamb3, lamb2, lamb1 = parametre 
                w_n_2, w_n_1, w_n = trig

                c2 = self.models_bi[idioma].get((w_n_2, w_n_1), 0)
                p3 = c3 / c2 if c2 > 0 else 0
                
                c_bigrama = self.models_bi[idioma].get((w_n_1, w_n), 0)
                c1 = self.models_uni[idioma].get(w_n_1, 0)
                p2 = c_bigrama / c1 if c1 > 0 else 0
                
                c_unigrama = self.models_uni[idioma].get(w_n, 0)
                p1 = c_unigrama / self.N_uni[idioma] if self.N_uni[idioma] > 0 else 0
                
                p = lamb1 * p1 + lamb2 * p2 + lamb3 * p3
                
            if p <= 0: 
                p = 1e-10 
                
            log_p += freq * math.log(p)
            
        return log_p

# Prova d'entrenament simple per visualitzar el funcionament tal com es feia
dades_train_simple = {}
for idioma in idiomes:
    fitxer_train = f"{idioma}_trn_clean.txt"
    if os.path.exists(fitxer_train):
        with open(fitxer_train, 'r', encoding='utf-8') as f:
            dades_train_simple[idioma] = f.read()

model_simple = LanguageModel(idiomes)
if dades_train_simple:
    model_simple.train(dades_train_simple)'''

code_cells[3].source = '''class LanguageEvaluator:
    def __init__(self, model):
        self.model = model

    def predict(self, frase, tecnica, parametre):
        max_prob = -float('inf')
        idioma_predit = None
        
        for idioma_model in self.model.idiomes:
            prob = self.model.calcul_probabilitat(frase, idioma_model, parametre, tecnica)
            if prob > max_prob:
                max_prob = prob
                idioma_predit = idioma_model
                
        return idioma_predit

    def cross_validation(self, dades_val, proves):
        resultats_cv = {}
        
        for tecnica, llista_parametres in proves.items():
            print(f"--- Avaluant tècnica: {tecnica.upper()} ---")
            millor_acc_tecnica = 0.0
            millor_param_tecnica = None
            
            for param in llista_parametres:
                encerts = 0
                total_frases = 0
                
                for idioma_real, llista_frases in dades_val.items():
                    for frase in llista_frases:
                        total_frases += 1
                        idioma_predit = self.predict(frase, tecnica, param)
                        
                        if idioma_predit == idioma_real:
                            encerts += 1
                            
                accuracy = encerts / total_frases
                print(f"  Param={str(param):<15} -> Accuracy: {accuracy:.4f}")
                
                if accuracy > millor_acc_tecnica:
                    millor_acc_tecnica = accuracy
                    millor_param_tecnica = param
                    
            resultats_cv[tecnica] = {'param': millor_param_tecnica, 'acc': millor_acc_tecnica}
            print(f"  >> Guanyador {tecnica}: Param={millor_param_tecnica} (Acc={millor_acc_tecnica:.4f})\\n")
            
        millor_tecnica_global = None
        millor_acc_global = 0.0

        for tec, res in resultats_cv.items():
            if res['acc'] > millor_acc_global:
                millor_acc_global = res['acc']
                millor_tecnica_global = tec

        print(f"\\nLa millor tècnica és: {millor_tecnica_global.upper()} amb un accuracy de {millor_acc_global:.4f}")
        return resultats_cv, millor_tecnica_global, millor_acc_global

    def avaluar_test(self, dades_test, tecnica, parametre):
        y_true = []
        y_pred = []
        
        for idioma_real, frases in dades_test.items():
            for frase in frases:
                y_true.append(idioma_real)
                idioma_predit = self.predict(frase, tecnica, parametre)
                y_pred.append(idioma_predit)
                
        acc_final = accuracy_score(y_true, y_pred)
        print(f"\\nResultat Final")
        print(f"Accuracy en Test: {acc_final:.4f} ({int(acc_final * len(y_true))}/{len(y_true)} encerts)")
        
        cm = confusion_matrix(y_true, y_pred, labels=self.model.idiomes)
        df_cm = pd.DataFrame(cm, index=self.model.idiomes, columns=self.model.idiomes)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Matriu de Confusió ({tecnica} {str(parametre)}) - Acc: {acc_final:.4f}')
        plt.ylabel('Idioma Real')
        plt.xlabel('Idioma Predit')
        plt.show()
        
        print("\\nMatriu de Confusió (Text):")
        print(df_cm)
        return y_true, y_pred, df_cm

    def mostrar_trigrames_dominants(self, idioma, top_n=10):
        scores = {}
        model_actual = self.model.models_tri[idioma]
        
        for trig, freq in model_actual.items():
            altres_freq = sum(self.model.models_tri[i].get(trig, 0) for i in self.model.idiomes if i != idioma)
            scores[trig] = freq / (altres_freq + 1)
            
        millors = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        print(f"Trigrames més identitaris del '{idioma}':")
        for t, s in millors:
            print(f"  {''.join(t)} (Score: {s:.2f})")

# PARTICIÓ DE DADES PER A CROSS-VALIDATION
dades_train_cv = {}
dades_val_cv = {}

for idioma in idiomes:
    fitxer_train = f"{idioma}_trn_clean.txt"
    if os.path.exists(fitxer_train):
        frases = preprocessor.obtenir_frases_netes(fitxer_train)
        divisio = int(len(frases) * 0.8)
        text_train = "  " + "  ".join(frases[:divisio]) + "  "
        
        dades_train_cv[idioma] = text_train
        dades_val_cv[idioma] = frases[divisio:]

model_cv = LanguageModel(idiomes)
if dades_train_cv:
    model_cv.train(dades_train_cv)
'''

code_cells[4].source = '''# Aquesta secció s'ha integrat completament en LanguageEvaluator a dalt'''

code_cells[5].source = '''# OPTIMITZACIÓ: CROSS-VALIDATION
proves = {
    'lidstone': [0.01, 0.1, 0.25, 0.5, 0.75, 1.0],
    'absolute_discounting': [0.1, 0.5, 0.75, 0.9],
    'interpolation': [
        (0.6, 0.3, 0.1),
        (0.7, 0.2, 0.1), 
        (0.8, 0.15, 0.05), 
        (0.5, 0.4, 0.1)
    ]
}

if dades_train_cv:
    evaluator_cv = LanguageEvaluator(model_cv)
    resultats_cv, millor_tec, millor_acc = evaluator_cv.cross_validation(dades_val_cv, proves)
'''

code_cells[6].source = '''# Pas 4: Avaluació Final - Construint els models definitius (100% Train)
dades_train_full = {}
dades_test = {}

for idioma in idiomes:
    fitxer_train = f"{idioma}_trn_clean.txt"
    if os.path.exists(fitxer_train):
        with open(fitxer_train, 'r', encoding='utf-8') as f:
            dades_train_full[idioma] = f.read()
            
    fitxer_test = f"{idioma}_tst_clean.txt"
    if os.path.exists(fitxer_test):
        dades_test[idioma] = preprocessor.obtenir_frases_netes(fitxer_test)

model_definitiu = LanguageModel(idiomes)
if dades_train_full:
    model_definitiu.train(dades_train_full)

print(f"B Global Definitiva: {model_definitiu.B_global} trigrammes únics.")

evaluator = LanguageEvaluator(model_definitiu)

# Avaluant sobre el conjunt de Test (Lidstone 0.75)
if dades_test:
    print("Avaluant sobre el conjunt de Test (Lidstone 0.75)...")
    _, _, _ = evaluator.avaluar_test(dades_test, 'lidstone', 0.75)
'''

code_cells[7].source = '''# Avaluant sobre el conjunt de Test (Interpolació)
if dades_test:
    print("Avaluant sobre el conjunt de Test (Interpolació 0.8, 0.15, 0.05)...")
    _, _, _ = evaluator.avaluar_test(dades_test, 'interpolation', (0.8, 0.15, 0.05))
'''

code_cells[8].source = '''# Trigrames dominants
if dades_train_full:
    evaluator.mostrar_trigrames_dominants('eng')
    print("-" * 30)
    evaluator.mostrar_trigrames_dominants('deu')
'''

# we clear the outputs on all code cells so that when the user opens the file it's clean and ready to run
for c in code_cells:
    c.outputs = []
    c.execution_count = None

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Project Notebook updated successfully.")
