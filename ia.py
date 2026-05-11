import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

# ═══════════════════════════════════════════════════════════════
# 1. TREINAR E SALVAR O MODELO DE RISCO
# ═══════════════════════════════════════════════════════════════
dados = {
    'Temperatura': [36.5, 36.8, 37.1, 37.5, 38.2, 38.8, 39.2, 36.6, 37.9, 38.5, 38.0],
    'SpO2':        [98,   99,   97,   96,   94,   91,   88,   98,   95,   92,   96],
    'BPM':         [75,   80,   78,   85,   105,  115,  125,  70,   90,   110,  95],
    'Cansaco':     [0,    0,    1,    0,    1,    1,    1,    0,    1,    1,    1],
    'Tosses':      [0,    1,    0,    2,    5,    8,    12,   0,    3,    10,   6],
    'Risco':       [0,    0,    0,    1,    2,    2,    2,    0,    1,    2,    2]
}

df = pd.DataFrame(dados)
X = df[['Temperatura', 'SpO2', 'BPM', 'Cansaco', 'Tosses']]
y = df['Risco']

modelo = DecisionTreeClassifier(random_state=42, max_depth=4)
modelo.fit(X, y)
joblib.dump(modelo, 'modelo_saude.pkl')


# ═══════════════════════════════════════════════════════════════
# 2. EXTRACTOR DE SINTOMAS A PARTIR DA DESCRIÇÃO DO PACIENTE
# ═══════════════════════════════════════════════════════════════

# Cada chave é um sintoma; os valores são palavras/frases que
# o paciente pode usar ao descrever o seu estado.
SINTOMAS_MAP = {
    'dor_cabeca': [
        'dor de cabeça', 'cefaleia', 'dor na cabeça',
        'cabeça a doer', 'cabeça doendo', 'dor cabeça'
    ],
    'dor_abdominal': [
        'dor abdominal', 'dor na barriga', 'barriga a doer',
        'barriga doendo', 'dor de barriga', 'cólicas', 'dor estomacal'
    ],
    'diarreia': [
        'diarreia', 'fezes líquidas', 'fezes moles',
        'soltura', 'evacuações frequentes'
    ],
    'calafrios': [
        'calafrios', 'tremores', 'arrepios', 'frio intenso'
    ],
    'suor_noturno': [
        'suor noturno', 'suando à noite', 'suor de noite', 'sudorese noturna'
    ],
    'perda_peso': [
        'perda de peso', 'emagrecimento', 'perdeu peso', 'emagreceu',
        'perdi peso', 'emagreci'
    ],
    'agua_nao_tratada': [
        'poço', 'rio', 'nascente', 'água do rio', 'água de poço',
        'não filtra', 'sem filtro', 'água não tratada', 'água suja'
    ],
    'urina_escura': [
        'urina escura', 'urina amarela escura', 'urina marrom',
        'urina cor de chá'
    ],
    'vomitos': [
        'vómito', 'vomitar', 'vomitei', 'vomitando',
        'enjoo', 'náuseas', 'vômito'
    ],
    'dor_garganta': [
        'dor de garganta', 'garganta a doer', 'garganta inflamada'
    ],
    'dor_muscular': [
        'dor muscular', 'dores no corpo', 'corpo a doer',
        'mialgia', 'dores musculares', 'corpo doendo'
    ],
    'sem_apetite': [
        'não como', 'não consigo comer', 'sem apetite',
        'falta de apetite', 'perda de apetite', 'não tenho fome'
    ],
    'ictericia': [
        'olhos amarelos', 'pele amarela', 'icterícia',
        'amarelamento', 'amarelado'
    ],
}


def extrair_sintomas(descricao: str) -> dict:
    """
    Recebe a descrição textual do paciente e devolve um dicionário
    com os sintomas identificados (True/False).
    """
    desc = descricao.lower()
    sintomas = {}
    for sintoma, palavras_chave in SINTOMAS_MAP.items():
        sintomas[sintoma] = any(p in desc for p in palavras_chave)
    return sintomas


# ═══════════════════════════════════════════════════════════════
# 3. MOTOR DE DECISÃO — QUAIS ANÁLISES CLÍNICAS FAZER
# ═══════════════════════════════════════════════════════════════

def recomendar_analises(sensores: dict, sintomas: dict, risco_ml: int) -> list:
    """
    Cruza os dados dos sensores, os sintomas extraídos do texto e
    a classificação do modelo ML para decidir as análises clínicas.
    Devolve uma lista de análises com nome, urgência e justificativa.
    """
    analises = []
    temp    = sensores['temperatura']
    spo2    = sensores['spo2']
    bpm     = sensores['bpm']
    tosses  = sensores['tosses']

    # ── HEMOGRAMA COMPLETO ──────────────────────────────────────
    # Base de qualquer avaliação infecciosa
    if temp >= 37.5 or risco_ml >= 1:
        analises.append({
            'nome':     'Hemograma Completo',
            'urgencia': 'Urgente' if temp >= 38.5 else 'Prioritário',
            'motivo':   (
                'Avalia infecção bacteriana ou viral, anemia e estado geral '
                'do sangue. Indicado sempre que há febre ou risco elevado.'
            ),
        })

    # ── REAÇÃO DE WIDAL (Febre Tifóide) ────────────────────────
    # Febre ≥ 38 °C + BPM alto + cefaleia/dor abdominal/diarreia
    # + exposição a água não tratada
    if (temp >= 38.0
            and bpm > 90
            and (sintomas['dor_cabeca'] or sintomas['dor_abdominal'] or sintomas['diarreia'])
            and sintomas['agua_nao_tratada']):
        analises.append({
            'nome':     'Reação de Widal',
            'urgencia': 'Prioritário',
            'motivo':   (
                'Suspeita de febre tifóide (Salmonella typhi): febre persistente '
                'com cefaleia/sintomas gastrointestinais e exposição a água não tratada '
                '— padrão endémico em Angola.'
            ),
        })

    # ── GOTA ESPESSA — Pesquisa de Malária ─────────────────────
    # Febre + calafrios ou dores musculares (região endémica)
    if temp >= 37.8 and (sintomas['calafrios'] or sintomas['dor_muscular']):
        analises.append({
            'nome':     'Gota Espessa (Pesquisa de Malária)',
            'urgencia': 'Urgente',
            'motivo':   (
                'Febre com calafrios/dores musculares são sinais clássicos de malária '
                'por Plasmodium — endémica em Angola. Diagnóstico urgente.'
            ),
        })

    # ── BACILOSCOPIA / BK (Tuberculose) ────────────────────────
    # SpO2 < 95 % + tosses ≥ 5 por dia, ou modelo classifica risco 2
    if (spo2 < 95 and tosses >= 5) or risco_ml == 2:
        analises.append({
            'nome':     'Baciloscopia de Escarro (BK × 3)',
            'urgencia': 'Urgente',
            'motivo':   (
                'Baixa saturação de O₂ com tosse persistente e/ou risco alto pelo '
                'modelo ML — pesquisa de Mycobacterium tuberculosis obrigatória.'
            ),
        })

    # ── RADIOGRAFIA DE TÓRAX ───────────────────────────────────
    if spo2 < 95 or tosses >= 5 or risco_ml == 2:
        analises.append({
            'nome':     'Radiografia de Tórax (PA)',
            'urgencia': 'Prioritário',
            'motivo':   (
                'Avaliar comprometimento pulmonar: infiltrados, cavidades ou '
                'consolidações compatíveis com TB, pneumonia ou outras patologias.'
            ),
        })

    # ── PCR — Proteína C-Reativa ────────────────────────────────
    if temp >= 38.0:
        analises.append({
            'nome':     'PCR (Proteína C-Reativa)',
            'urgencia': 'Prioritário',
            'motivo':   (
                'Marcador de inflamação sistémica — quantifica a gravidade '
                'do processo infeccioso e orienta o acompanhamento terapêutico.'
            ),
        })

    # ── TRANSAMINASES TGO / TGP ────────────────────────────────
    # Urina escura, icterícia ou dor abdominal → avaliar fígado
    if sintomas['urina_escura'] or sintomas['ictericia'] or sintomas['dor_abdominal']:
        analises.append({
            'nome':     'Transaminases (TGO / TGP)',
            'urgencia': 'Prioritário',
            'motivo':   (
                'Urina escura, olhos amarelos ou dor abdominal indicam possível '
                'lesão hepática — avaliar hepatite viral, leptospirose ou malária grave.'
            ),
        })

    # ── TESTE RÁPIDO DE HIV ─────────────────────────────────────
    # Perda de peso + suor noturno + comprometimento respiratório
    if (sintomas['perda_peso']
            and sintomas['suor_noturno']
            and (spo2 < 95 or tosses >= 3)):
        analises.append({
            'nome':     'Teste Rápido de HIV',
            'urgencia': 'Prioritário',
            'motivo':   (
                'Perda de peso + suor noturno + sintomas respiratórios sugerem '
                'imunossupressão — descartar infecção por HIV essencial neste quadro.'
            ),
        })

    # ── UREIA E CREATININA ──────────────────────────────────────
    # Taquicardia + febre alta → risco de desidratação / insuficiência renal
    if bpm > 100 and temp >= 38.5:
        analises.append({
            'nome':     'Ureia e Creatinina',
            'urgencia': 'Prioritário',
            'motivo':   (
                'Taquicardia com febre alta indica risco de desidratação grave '
                '— avaliar função renal para prevenir lesão aguda.'
            ),
        })

    # ── GLICEMIA ────────────────────────────────────────────────
    if risco_ml >= 1 or temp >= 38.0:
        analises.append({
            'nome':     'Glicemia em Jejum',
            'urgencia': 'Rotina',
            'motivo':   (
                'Avaliação metabólica de rotina. Hiperglicemia pode agravar '
                'quadros infecciosos e mascarar cetoacidose diabética.'
            ),
        })

    return analises


# ═══════════════════════════════════════════════════════════════
# 4. FUNÇÃO PRINCIPAL — DIAGNÓSTICO COMPLETO
# ═══════════════════════════════════════════════════════════════

def diagnosticar(temperatura, spo2, bpm, cansaco, tosses, descricao_paciente):
    """
    Parâmetros
    ----------
    temperatura         : float — temperatura corporal em °C
    spo2                : int   — saturação de oxigénio (%)
    bpm                 : int   — batimentos por minuto
    cansaco             : int   — 0 = Não  |  1 = Sim
    tosses              : int   — número de tosses por dia
    descricao_paciente  : str   — descrição livre dos sintomas
    """
    # Carregar modelo e classificar risco
    modelo_carregado = joblib.load('modelo_saude.pkl')
    X_novo = pd.DataFrame(
        [[temperatura, spo2, bpm, cansaco, tosses]],
        columns=['Temperatura', 'SpO2', 'BPM', 'Cansaco', 'Tosses']
    )
    risco_ml = modelo_carregado.predict(X_novo)[0]
    risco_labels = {0: 'Normal', 1: 'Infeção Leve', 2: 'Alta Suspeita de TB'}

    # Extrair sintomas da descrição
    sintomas = extrair_sintomas(descricao_paciente)

    # Gerar lista de análises recomendadas
    sensores = {'temperatura': temperatura, 'spo2': spo2, 'bpm': bpm, 'tosses': tosses}
    analises = recomendar_analises(sensores, sintomas, risco_ml)

    # Ordenar por urgência: Urgente → Prioritário → Rotina
    ordem = {'Urgente': 0, 'Prioritário': 1, 'Rotina': 2}
    analises = sorted(analises, key=lambda x: ordem[x['urgencia']])

    # ── IMPRIMIR RELATÓRIO ──────────────────────────────────────
    L = 58
    print("\n" + "═" * L)
    print("         RELATÓRIO DE TRIAGEM CLÍNICA")
    print("═" * L)
    print(f"  Temperatura  : {temperatura} °C")
    print(f"  SpO2         : {spo2} %")
    print(f"  BPM          : {bpm} bpm")
    print(f"  Cansaço      : {'Sim' if cansaco else 'Não'}")
    print(f"  Tosses / dia : {tosses}")
    print(f"\n  Descrição    : \"{descricao_paciente}\"")
    print("─" * L)

    risco_emoji = {0: '🟢', 1: '🟡', 2: '🔴'}
    print(f"\n  {risco_emoji[risco_ml]} CLASSIFICAÇÃO DE RISCO : {risco_labels[risco_ml]}")

    sintomas_detectados = [k.replace('_', ' ') for k, v in sintomas.items() if v]
    if sintomas_detectados:
        print(f"\n  Sintomas identificados na descrição:")
        for s in sintomas_detectados:
            print(f"    • {s}")
    else:
        print("\n  Nenhum sintoma específico identificado na descrição.")

    print(f"\n  📋 ANÁLISES CLÍNICAS RECOMENDADAS ({len(analises)}):")
    print("─" * L)

    emoji_urg = {'Urgente': '🔴', 'Prioritário': '🟡', 'Rotina': '🟢'}
    for i, a in enumerate(analises, 1):
        e = emoji_urg[a['urgencia']]
        print(f"\n  {i}. {a['nome']}")
        print(f"     {e} Urgência : {a['urgencia']}")
        print(f"     📌 Motivo   : {a['motivo']}")

    print("\n" + "═" * L)
    print("  ⚠️  Ferramenta de apoio à decisão clínica.")
    print("  O diagnóstico final cabe ao médico responsável.")
    print("═" * L + "\n")


# ═══════════════════════════════════════════════════════════════
# 5. EXEMPLOS DE USO
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── CASO 1: Suspeita de Febre Tifóide ──────────────────────
    diagnosticar(
        temperatura=38.6,
        spo2=96,
        bpm=102,
        cansaco=1,
        tosses=1,
        descricao_paciente=(
            "Estou com dor de cabeça há dois dias, tenho dor na barriga "
            "e vomitei uma vez. Bebo água do poço. Sinto muito cansaço."
        )
    )

    # ── CASO 2: Suspeita de Malária ─────────────────────────────
    diagnosticar(
        temperatura=39.2,
        spo2=94,
        bpm=112,
        cansaco=1,
        tosses=0,
        descricao_paciente=(
            "Tenho calafrios fortes e dores no corpo há três dias. "
            "Suei muito durante a noite e sinto que tenho muito frio mesmo com calor."
        )
    )

    # ── CASO 3: Suspeita de Tuberculose ─────────────────────────
    diagnosticar(
        temperatura=38.3,
        spo2=91,
        bpm=95,
        cansaco=1,
        tosses=9,
        descricao_paciente=(
            "Tenho tosse há mais de três semanas, perdi muito peso "
            "e tenho suor noturno intenso. Não consigo comer direito."
        )
    )

    # ── CASO 4: Paciente Normal ─────────────────────────────────
    diagnosticar(
        temperatura=36.7,
        spo2=98,
        bpm=72,
        cansaco=0,
        tosses=0,
        descricao_paciente="Sinto-me bem, sem sintomas relevantes."
    )