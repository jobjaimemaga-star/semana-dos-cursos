import cv2
import numpy as np
import threading
import serial
import time
import joblib
import spacy
import random
import string
import mysql.connector
import sounddevice as sd
import webbrowser
import math
import pandas as pd
import os
import json  # ← ADICIONADO: necessário para guardar múltiplos medicamentos
import concurrent.futures
from werkzeug.utils import secure_filename
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, session
from flask_cors import CORS
from ultralytics import YOLO
from deepface import DeepFace
from collections import deque

# ==========================================
# 1. CONFIGURAÇÕES GLOBAIS E BANCO
# ==========================================
app = Flask(__name__)
app.secret_key = 'chave_secreta_angola_tb_guard' 
CORS(app)

db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '', 
    'database': 'tb_guard_db'
}

PORTA_ARDUINO = 'COM20'
BAUD_RATE = 115200

dados_atuais = {
    "temperatura": "--",
    "spo2": "--",
    "bpm": "--",
    "status": "Aguardando Sensor...",
    "cor": "gray",
    "cansaco": 0, 
    "alerta_tosse": "A escutar..."
}

historico_tosses = deque()
TEMPO_CRISE = 10

def conectar_banco():
    return mysql.connector.connect(**db_config)

# ==========================================
# 2. CARREGAMENTO DE MODELOS (IA, YOLO, NLP)
# ==========================================
try:
    modelo_ia = joblib.load('modelo_saude.pkl')
    print("✅ [IA] Modelo de saúde carregado!")
except:
    modelo_ia = None
    print("⚠️ [IA] Arquivo 'modelo_saude.pkl' não encontrado.")

try:
    try:
        nlp = spacy.load("pt_core_news_lg")
    except:
        nlp = spacy.load("pt_core_news_sm")
    print("✅ [NLP] spaCy carregado com sucesso!")
except Exception as e:
    nlp = None
    print(f"⚠️ [NLP] spaCy não encontrado. Erro: {e}")

try:
    model_yolo = YOLO('yolov8n.pt')
    print("✅ [YOLO] Modelo carregado.")
except Exception as e:
    model_yolo = None
    print(f"❌ [YOLO] Erro: {e}")

# ==========================================
# 3. VISÃO COMPUTACIONAL (EMOÇÃO + CANSAÇO)
# ==========================================
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
_estado_emocional = "Analisando..."
_emocao_future = None
def gen_frames():
    global _estado_emocional, _emocao_future
    cap = cv2.VideoCapture(0)
    
    # ✅ Reduz resolução → MUITO mais rápido
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    contador = 0
    caixas_cache = []  # guarda as últimas deteções
    traducoes = {
        "angry": "Mal (Irritado)", "disgust": "Desconforto",
        "fear": "Ansioso", "happy": "Bem (Feliz)",
        "sad": "Mau (Triste)", "surprise": "Surpreso",
        "neutral": "Estável (Neutro)"
    }

    while True:
        success, frame = cap.read()
        if not success:
            break

        # ✅ YOLO só corre a cada 5 frames (não todos)
        if model_yolo and contador % 5 == 0:
            results = model_yolo(frame, conf=0.5, classes=[0], verbose=False)
            caixas_cache = []
            for r in results:
                for box in r.boxes:
                    caixas_cache.append(list(map(int, box.xyxy[0])))

        # Usa as caixas guardadas (mesmo nos frames sem YOLO)
        for (x1, y1, x2, y2) in caixas_cache:
            h = y2 - y1
            roi_olhos = frame[y1 + h//5: y1 + h//2, x1:x2]
            if roi_olhos.size > 0:
                gray = cv2.cvtColor(roi_olhos, cv2.COLOR_BGR2GRAY)
                dados_atuais["cansaco"] = 1 if np.mean(gray) < 65 else 0

            # ✅ DeepFace em thread separada — não bloqueia o vídeo
            if contador % 30 == 0:
                if _emocao_future is None or _emocao_future.done():
                    face_roi = frame[y1:y2, x1:x2].copy()
                    def analisar(roi, trad):
                        global _estado_emocional
                        try:
                            objs = DeepFace.analyze(
                                roi, actions=['emotion'],
                                enforce_detection=False, silent=True
                            )
                            _estado_emocional = trad.get(
                                objs[0]['dominant_emotion'], "Estável"
                            )
                        except:
                            pass
                    _emocao_future = _executor.submit(analisar, face_roi, traducoes)

            cor = (0, 0, 255) if dados_atuais["cansaco"] else (0, 255, 0)
            label = f"{_estado_emocional} | {'Cansado' if dados_atuais['cansaco'] else 'Alerta'}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)

        contador += 1
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + buffer.tobytes() + b'\r\n')
# ==========================================
# 4. ÁUDIO E ARDUINO
# ==========================================
def analisar_tosse():
    global dados_atuais
    def callback(indata, frames, t, status):
        volume_norm = np.linalg.norm(indata) * 10
        if volume_norm > 0.8:
            agora = time.time()
            historico_tosses.append(agora)
            while historico_tosses and historico_tosses[0] < agora - TEMPO_CRISE:
                historico_tosses.popleft()
            num = len(historico_tosses)
            dados_atuais["alerta_tosse"] = f"CRISE ({num} tosses)" if num >= 3 else "Tosse Isolada"

    try:
        with sd.InputStream(callback=callback, channels=1):
            while True: time.sleep(1)
    except: pass

def ler_arduino():
    global dados_atuais
    conexao = None
    while conexao is None:
        try:
            conexao = serial.Serial(PORTA_ARDUINO, BAUD_RATE, timeout=0.1)
            print("✅ ESP32 Conectado!")
        except:
            time.sleep(2)

    while True:
        try:
            if conexao.in_waiting > 0:
                linha = conexao.readline().decode('utf-8', errors='ignore').strip()
                v = linha.split(',')
                if len(v) == 3:
                    temp, spo2, bpm = float(v[0]), int(v[1]), int(v[2])
                    dados_atuais["temperatura"] = f"{temp:.1f}°C"
                    if spo2 > 0:
                        dados_atuais["spo2"], dados_atuais["bpm"] = f"{spo2}%", f"{bpm}"
                        if modelo_ia:
                            tosses = 3 if "CRISE" in dados_atuais["alerta_tosse"] else 0
                            prev = modelo_ia.predict([[temp, spo2, bpm, dados_atuais["cansaco"], tosses]])[0]
                            config = {0: ("Normal", "green"), 1: ("Possível Infecção", "#e6b800"), 2: ("ALERTA: Risco TB", "red")}
                            dados_atuais["status"], dados_atuais["cor"] = config.get(prev, ("Analise...", "gray"))
                    else:
                        dados_atuais["status"], dados_atuais["cor"] = "Posicione o dedo", "gray"
        except: pass
        time.sleep(0.05)

def gerar_credenciais_paciente(nome):
    num_processo = f"TB{random.randint(1000, 9999)}"
    senha = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    return num_processo, senha

# ==========================================
# 5. ROTAS FLASK
# ==========================================
@app.route('/processar_relato', methods=['POST'])
def processar_relato():
    texto = request.json.get('texto', '').lower()
    exames = set()
    
    if nlp:
        doc = nlp(texto)
        tokens = [t.lemma_ for t in doc]
    else:
        tokens = texto.split()

    if dados_atuais["cor"] == "red":
        exames.update(["Bacilo de Koch (Escarro)", "Raio-X de Tórax"])
    
    mapeamento_sintomas = {
        "peito": "Eletrocardiograma (ECG)",
        "garganta": "Hemograma Completo",
        "urina": "Exame de Urina II",
        "febre": "Teste de Malária/Dengue",
        "tosse": "Avaliação Pulmonar"
    }
    
    for chave, exame in mapeamento_sintomas.items():
        if chave in tokens or chave in texto:
            exames.add(exame)

    lista_final = list(exames) if exames else ["Avaliação Clínica Geral"]
    return jsonify({"exames": lista_final})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/registrar_paciente', methods=['POST'])
def registrar_paciente():
    nome = request.form.get('nome')
    bi = request.form.get('bi_paciente')
    telefone = request.form.get('telefone')
    provincia = request.form.get('provincia')
    municipio = request.form.get('municipio')
    
    num_proc, senha_ia = gerar_credenciais_paciente(nome)
    
    try:    
        conn = conectar_banco()
        cursor = conn.cursor()
        sql = "INSERT INTO pacientes (nome, bi, num_processo, senha, tipo_usuario) VALUES (%s, %s, %s, %s, 'paciente')"
        cursor.execute(sql, (nome, bi, num_proc, senha_ia))
        paciente_id = cursor.lastrowid
        conn.commit()
        session['paciente_id'] = paciente_id
        session['num_processo'] = num_proc
        session['senha_paciente'] = senha_ia
        session['tipo'] = 'paciente'
        cursor.close()
        conn.close()
        return redirect(url_for('abrir_triagem'))
    except Exception as e:
        print(f"Erro no Banco: {e}")
        return f"Erro ao registrar: {e}"

@app.route('/triagem')
def abrir_triagem():
    if 'paciente_id' not in session:
        return redirect(url_for('index'))
    return render_template('index1.html')

@app.route('/cadastro')
def abrir_cadastro_nacional():
    return render_template('consultas.html')

@app.route('/dados')
def enviar_dados():
    return jsonify(dados_atuais)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/medico')
@app.route('/painel_medico')
def abrir_painel_medico(): 
    if 'medico_logado' not in session:
        return redirect(url_for('login_medico'))
        
    conn = conectar_banco()
    cursor = conn.cursor(dictionary=True)
    
    query = """
        SELECT 
            p.id, 
            p.nome, 
            p.bi, 
            p.num_processo, 
            t.temperatura, 
            t.spo2, 
            t.bpm, 
            t.status_diagnostico,
            tr.medicamento,
            tr.horario,
            tr.dias,
            tr.status AS status_tratamento
        FROM pacientes p
        LEFT JOIN triagens t ON p.id = t.paciente_id
        LEFT JOIN tratamentos tr ON p.id = tr.id_paciente
        GROUP BY p.id
        ORDER BY p.id DESC
    """
    
    try:
        cursor.execute(query)
        registros = cursor.fetchall()
    except Exception as e:
        print(f"Erro na query: {e}")
        registros = []
    finally:
        cursor.close()
        conn.close()
    
    return render_template('medico.html', pacientes=registros)


# ==========================================
# FIX 1 — ROTA /prescrever
# PROBLEMA: usava .get() em vez de .getlist() → só guardava 1 medicamento
# SOLUÇÃO:  usa .getlist() para apanhar o array completo e guarda como JSON
# ==========================================
@app.route('/prescrever', methods=['POST'])
def prescrever():
    try:
        paciente_id = request.form.get('paciente_id')

        # Apanhar TODOS os valores dos arrays do formulário
        medicamentos_lista = request.form.getlist('medicamento[]')
        doses_lista        = request.form.getlist('dose[]')
        horarios_lista     = request.form.getlist('horario[]')
        dias_lista         = request.form.getlist('dias[]')
        vias_lista         = request.form.getlist('via[]')

        # Construir lista de dicts (um por medicamento)
        lista_meds = [
            {
                "medicamento": m,
                "dose":        d,
                "horario":     h,
                "dias":        di,
                "via":         v
            }
            for m, d, h, di, v in zip(
                medicamentos_lista, doses_lista,
                horarios_lista, dias_lista, vias_lista
            )
            if m.strip()  # ignora linhas vazias
        ]

        # Serializar para JSON (vai para a coluna medicamentos_json)
        medicamentos_json = json.dumps(lista_meds, ensure_ascii=False)

        # Guardar também o 1.º medicamento no campo legado (retrocompatibilidade)
        med_principal     = medicamentos_lista[0] if medicamentos_lista else ''
        horario_principal = horarios_lista[0]     if horarios_lista     else ''
        dias_principal    = dias_lista[0]         if dias_lista         else 0

        conn = conectar_banco()
        cursor = conn.cursor()

        # Verifica se já existe tratamento para este paciente
        cursor.execute(
            "SELECT id FROM tratamentos WHERE id_paciente = %s ORDER BY id DESC LIMIT 1",
            (paciente_id,)
        )
        existente = cursor.fetchone()

        if existente:
            # Actualiza o registo existente
            sql = """
                UPDATE tratamentos
                SET medicamento = %s, horario = %s, dias = %s,
                    medicamentos_json = %s, status = 'PENDENTE'
                WHERE id_paciente = %s
            """
            cursor.execute(sql, (
                med_principal, horario_principal, dias_principal,
                medicamentos_json, paciente_id
            ))
        else:
            # Cria novo registo
            sql = """
                INSERT INTO tratamentos
                    (id_paciente, medicamento, horario, dias, medicamentos_json, status)
                VALUES (%s, %s, %s, %s, %s, 'PENDENTE')
            """
            cursor.execute(sql, (
                paciente_id, med_principal, horario_principal,
                dias_principal, medicamentos_json
            ))

        conn.commit()
        cursor.close()
        conn.close()

        return redirect(url_for('abrir_painel_medico'))

    except Exception as e:
        print(f"Erro ao salvar prescrição: {e}")
        return f"Erro: {e}", 500


@app.route('/chamar_paciente/<int:id_paciente>')
def chamar_paciente(id_paciente):
    try:
        conn = conectar_banco()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE tratamentos SET alerta_retorno = 1 WHERE id_paciente = %s",
            (id_paciente,)
        )
        conn.commit()
        cursor.close()
        conn.close()
        return redirect(url_for('abrir_painel_medico'))
    except Exception as e:
        return str(e), 500


# ==========================================
# FIX 2 — ROTA /notificar_termino
# PROBLEMA: mudava status para 'CONCLUÍDO' mas o template verifica 'NOTIFICADO'
# SOLUÇÃO:  mudar para 'NOTIFICADO' e redirecionar para painel_paciente
# ==========================================
@app.route('/notificar_termino', methods=['POST'])
def notificar_termino():
    try:
        paciente_id = session.get('paciente_id')
        if not paciente_id:
            return redirect(url_for('index'))

        conn = conectar_banco()
        cursor = conn.cursor()

        # Status correcto que o template verifica
        sql = "UPDATE tratamentos SET status = 'NOTIFICADO' WHERE id_paciente = %s"
        cursor.execute(sql, (paciente_id,))

        conn.commit()
        cursor.close()
        conn.close()

        return redirect(url_for('painel_paciente'))  # recarrega o painel actualizado

    except Exception as e:
        return f"Erro ao notificar: {e}", 500


# ==========================================
# UPLOAD DE STOCK
# ==========================================
UPLOAD_FOLDER = 'uploads_stock'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/farmacias/upload_stock', methods=['POST'])
def upload_stock_csv():
    farmacia_id = request.form.get('farmacia_id')
    ficheiro = request.files.get('ficheiro')

    if not ficheiro or not farmacia_id:
        return jsonify({"erro": "Ficheiro ou ID em falta"}), 400

    nome = secure_filename(ficheiro.filename)
    caminho = os.path.join(UPLOAD_FOLDER, nome)
    ficheiro.save(caminho)

    try:
        if nome.endswith('.csv'):
            df = pd.read_csv(caminho, encoding='utf-8-sig')
        else:
            df = pd.read_excel(caminho)
    except Exception as e:
        return jsonify({"erro": f"Ficheiro inválido: {e}"}), 400

    colunas = ['medicamento', 'principio_ativo', 'dosagem', 'preco', 'quantidade']
    for col in colunas:
        if col not in df.columns:
            return jsonify({"erro": f"Coluna '{col}' em falta no ficheiro"}), 400

    df = df.dropna(subset=['medicamento', 'preco', 'quantidade'])
    df['quantidade'] = pd.to_numeric(df['quantidade'], errors='coerce').fillna(0).astype(int)
    df['preco'] = pd.to_numeric(df['preco'], errors='coerce').fillna(0)

    try:
        conn = conectar_banco()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM estoque_farmacias WHERE farmacia_id = %s", (farmacia_id,))
        sql = """INSERT INTO estoque_farmacias
                 (farmacia_id, medicamento, principio_ativo, dosagem, preco, quantidade)
                 VALUES (%s, %s, %s, %s, %s, %s)"""
        registos = [
            (farmacia_id, row['medicamento'], row.get('principio_ativo',''),
             row.get('dosagem',''), row['preco'], row['quantidade'])
            for _, row in df.iterrows()
        ]
        cursor.executemany(sql, registos)
        conn.commit()
        cursor.close(); conn.close()
        os.remove(caminho)
        return jsonify({"sucesso": True, "importados": len(registos)})
    except Exception as e:
        return jsonify({"erro": str(e)}), 500


@app.route('/farmacias/modelo_csv')
def descarregar_modelo():
    import io
    modelo = "medicamento,principio_ativo,dosagem,preco,quantidade\n"
    modelo += "Rifampicina 600mg,Rifampicina,600mg,2500,30\n"
    modelo += "Isoniazida 300mg,Isoniazida,300mg,1800,45\n"
    return Response(
        modelo, mimetype='text/csv',
        headers={"Content-Disposition": "attachment;filename=modelo_stock.csv"}
    )

def calcular_distancia_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


@app.route('/farmacias/buscar', methods=['POST'])
def buscar_farmacias():
    dados = request.json
    medicamento = dados.get('medicamento', '').strip()
    lat_paciente = float(dados.get('latitude', 0))
    lon_paciente = float(dados.get('longitude', 0))

    if not medicamento:
        return jsonify({"erro": "Medicamento não especificado"}), 400

    try:
        conn = conectar_banco()
        cursor = conn.cursor(dictionary=True)
        query = """
            SELECT f.id, f.nome, f.endereco, f.telefone,
                   f.latitude, f.longitude,
                   e.medicamento, e.dosagem, e.preco, e.quantidade, e.atualizado_em
            FROM estoque_farmacias e
            JOIN farmacias f ON e.farmacia_id = f.id
            WHERE (e.medicamento LIKE %s OR e.principio_ativo LIKE %s)
              AND e.quantidade > 0 AND f.ativa = 1
        """
        termo = f"%{medicamento}%"
        cursor.execute(query, (termo, termo))
        resultados = cursor.fetchall()
        cursor.close()
        conn.close()

        for r in resultados:
            r['distancia_km'] = round(
                calcular_distancia_km(lat_paciente, lon_paciente,
                                      float(r['latitude']), float(r['longitude'])), 1)
            r['preco'] = float(r['preco'])
            if r.get('atualizado_em'):
                r['atualizado_em'] = r['atualizado_em'].strftime('%d/%m/%Y %H:%M')

        resultados.sort(key=lambda x: (x['preco'], x['distancia_km'], -x['quantidade']))
        return jsonify({"medicamento": medicamento, "total": len(resultados), "farmacias": resultados})

    except Exception as e:
        return jsonify({"erro": str(e)}), 500


@app.route('/farmacias/por_receita/<int:paciente_id>', methods=['GET'])
def farmacias_por_receita(paciente_id):
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)

    try:
        conn = conectar_banco()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT medicamento FROM receitas
            WHERE paciente_id = %s AND ativa = 1
        """, (paciente_id,))
        receita = cursor.fetchall()
        cursor.close()
        conn.close()

        if not receita:
            return jsonify({"mensagem": "Sem receita activa para este paciente"})

        resultado_final = {}
        for item in receita:
            med = item['medicamento']
            conn2 = conectar_banco()
            cur2 = conn2.cursor(dictionary=True)
            termo = f"%{med}%"
            cur2.execute("""
                SELECT f.nome, f.endereco, f.telefone, f.latitude, f.longitude,
                       e.preco, e.quantidade, e.atualizado_em
                FROM estoque_farmacias e JOIN farmacias f ON e.farmacia_id = f.id
                WHERE (e.medicamento LIKE %s OR e.principio_ativo LIKE %s)
                  AND e.quantidade > 0 AND f.ativa = 1
            """, (termo, termo))
            farmacias = cur2.fetchall()
            cur2.close(); conn2.close()

            for f in farmacias:
                f['distancia_km'] = round(calcular_distancia_km(lat, lon, float(f['latitude']), float(f['longitude'])), 1)
                f['preco'] = float(f['preco'])
                if f.get('atualizado_em'):
                    f['atualizado_em'] = f['atualizado_em'].strftime('%d/%m/%Y %H:%M')

            farmacias.sort(key=lambda x: (x['preco'], x['distancia_km']))
            resultado_final[med] = farmacias

        return jsonify(resultado_final)

    except Exception as e:
        return jsonify({"erro": str(e)}), 500


@app.route('/farmacias/registar', methods=['POST'])
def registar_farmacia():
    dados = request.json
    try:
        conn = conectar_banco()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO farmacias (nome, endereco, latitude, longitude, telefone, email, senha)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (dados['nome'], dados['endereco'], dados['latitude'], dados['longitude'],
              dados.get('telefone'), dados.get('email'), dados.get('senha')))
        farmacia_id = cursor.lastrowid
        conn.commit()
        cursor.close(); conn.close()
        return jsonify({"sucesso": True, "farmacia_id": farmacia_id})
    except Exception as e:
        return jsonify({"erro": str(e)}), 500


@app.route('/farmacias/atualizar_stock', methods=['POST'])
def atualizar_stock():
    dados = request.json
    farmacia_id = dados.get('farmacia_id')
    medicamentos = dados.get('medicamentos', [])

    try:
        conn = conectar_banco()
        cursor = conn.cursor()
        for med in medicamentos:
            cursor.execute("""
                INSERT INTO estoque_farmacias (farmacia_id, medicamento, principio_ativo, dosagem, preco, quantidade)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE preco=%s, quantidade=%s, atualizado_em=NOW()
            """, (farmacia_id, med['medicamento'], med.get('principio_ativo', ''),
                  med.get('dosagem', ''), med['preco'], med['quantidade'],
                  med['preco'], med['quantidade']))
        conn.commit()
        cursor.close(); conn.close()
        return jsonify({"sucesso": True, "actualizados": len(medicamentos)})
    except Exception as e:
        return jsonify({"erro": str(e)}), 500


@app.route('/farmacias')
def pagina_farmacias():
    if 'paciente_id' not in session:
        return redirect(url_for('index'))

    medicamentos = []
    try:
        conn = conectar_banco()
        cursor = conn.cursor(dictionary=True)

        # Busca o campo JSON completo e o campo legado
        cursor.execute("""
            SELECT medicamento, medicamentos_json FROM tratamentos 
            WHERE id_paciente = %s 
            ORDER BY id DESC LIMIT 1
        """, (session['paciente_id'],))
        resultado = cursor.fetchone()
        cursor.close()
        conn.close()

        if resultado:
            # Prioridade 1: ler do JSON (receita com múltiplos medicamentos)
            if resultado.get('medicamentos_json'):
                lista = json.loads(resultado['medicamentos_json'])
                medicamentos = [
                    m['medicamento'].strip()
                    for m in lista
                    if m.get('medicamento', '').strip()
                ]
            # Prioridade 2: fallback para campo legado (campo simples)
            elif resultado.get('medicamento'):
                medicamentos = [m.strip() for m in resultado['medicamento'].split(',')]

    except Exception as e:
        print(f"Erro ao buscar receita: {e}")

    return render_template('painel_farmacias.html', medicamentos=medicamentos)


@app.route('/portal-farmacia')
def portal_farmacia():
    return render_template('portal_farmacia.html')

    
@app.route('/login_medico', methods=['GET', 'POST'])
@app.route('/medico/login', methods=['GET', 'POST'])
def login_medico():
    if request.method == 'POST':
        email = request.form.get('email_medico')
        senha = request.form.get('senha')
        conn = conectar_banco()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM medicos WHERE email = %s AND senha = %s", (email, senha))
        medico = cursor.fetchone()
        cursor.close()
        conn.close()
        if medico:
            session['medico_logado'] = medico['nome']
            return redirect(url_for('abrir_painel_medico'))
        else:
            return "E-mail ou Senha incorretos!"
    return render_template('login_medico.html')


@app.route('/logout_medico')
def logout_medico():
    session.pop('medico_logado', None)
    return redirect(url_for('login_medico'))


@app.route('/login_paciente', methods=['POST'])
@app.route('/login_paciente_page', methods=['GET'])
@app.route('/paciente/login', methods=['GET'])
def login_paciente():
    if request.method == 'GET':
        return render_template('login_paciente.html')

    num_processo = request.form.get('num_processo')
    senha = request.form.get('senha')

    try:
        conn = conectar_banco()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT p.id AS id_paciente_real FROM pacientes p
            WHERE p.num_processo = %s AND p.senha = %s
        """, (num_processo, senha))
        resultado = cursor.fetchone()
        cursor.close()
        conn.close()

        if resultado:
            session['paciente_id'] = resultado['id_paciente_real']
            return redirect(url_for('painel_paciente'))
        else:
            return "Dados incorretos", 401

    except Exception as e:
        print(f"❌ Erro no Login: {e}")
        return f"Erro interno: {e}", 500


# ==========================================
# FIX 3 — ROTA /painel_paciente
# PROBLEMA 1: lia dados da sessão (definida no login) → nunca actualizava
#             quando o médico prescrevia depois do login
# PROBLEMA 2: nunca passava 'medicamentos' ao template → lista ficava vazia
# SOLUÇÃO: consultar a BD directamente em cada visita à página
# ==========================================
@app.route('/painel_paciente')
def painel_paciente():
    if 'paciente_id' not in session:
        return redirect(url_for('login_paciente'))

    paciente_id = session['paciente_id']

    conn = conectar_banco()
    cursor = conn.cursor(dictionary=True)

    # Buscar o tratamento mais recente directamente da BD (não da sessão)
    cursor.execute("""
        SELECT * FROM tratamentos
        WHERE id_paciente = %s
        ORDER BY id DESC LIMIT 1
    """, (paciente_id,))
    t = cursor.fetchone()

    cursor.close()
    conn.close()

    medicamentos = []
    dias = 0

    if t:
        # Calcular dias restantes
        dias = int(t.get('dias') or 0)

        # Desserializar a lista de medicamentos guardada em JSON
        if t.get('medicamentos_json'):
            try:
                medicamentos = json.loads(t['medicamentos_json'])
            except Exception as e:
                print(f"Erro ao ler medicamentos_json: {e}")
                medicamentos = []

        # Fallback: se não houver JSON mas houver campo legado, criar lista simples
        if not medicamentos and t.get('medicamento'):
            medicamentos = [{
                "medicamento": t['medicamento'],
                "dose":        "",
                "horario":     t.get('horario', ''),
                "dias":        t.get('dias', ''),
                "via":         "oral"
            }]

    return render_template('painel_paciente.html', t=t, dias=dias, medicamentos=medicamentos)


@app.route('/confirmar_medicamentos', methods=['POST'])
def confirmar_medicamentos():
    try:
        id_p = session.get('paciente_id')
        conn = conectar_banco()
        cursor = conn.cursor()
        cursor.execute("UPDATE tratamentos SET status = 'CONCLUÍDO' WHERE id_paciente = %s", (id_p,))
        conn.commit()
        cursor.close()
        conn.close()
        return "Obrigado! O médico foi notificado."
    except Exception as e:
        return str(e), 500


# ==========================================
# 6. FLUXO DE TRIAGEM
# ==========================================
@app.route('/finalizar_triagem', methods=['POST'])
def finalizar_triagem():
    paciente_id = session.get('paciente_id')
    dados = request.get_json()
    conn = conectar_banco()
    cursor = conn.cursor()
    sql = """INSERT INTO triagens (id_paciente, temperatura, spo2, bpm, status_diagnostico) 
         VALUES (%s, %s, %s, %s, %s)"""
    cursor.execute(sql, (paciente_id, dados.get('temp'), dados.get('spo2'), dados.get('bpm'), dados.get('status')))
    conn.commit()
    res = {
        "status": "success",
        "processo": session.get('num_processo'),
        "senha": session.get('senha_paciente')
    }
    return jsonify(res)


@app.route('/salvar_triagem', methods=['POST'])
def salvar_triagem():
    try:
        dados = request.get_json()
        paciente_id = session.get('paciente_id', 1)
        conn = conectar_banco()
        cursor = conn.cursor()
        sql = """INSERT INTO triagens 
                 (paciente_id, temperatura, spo2, bpm, alerta_tosse, status_diagnostico, cor_alerta) 
                 VALUES (%s, %s, %s, %s, %s, %s, %s)"""
        valores = (
            paciente_id, 
            str(dados.get('temp', '--')), 
            str(dados.get('spo2', '--')), 
            str(dados.get('bpm', '--')), 
            str(dados.get('alerta_tosse', 'Sem registro')), 
            str(dados.get('status', 'Sem diagnóstico')),
            str(dados.get('cor', 'gray'))
        )
        cursor.execute(sql, valores)
        conn.commit()
        cursor.close()
        conn.close()
        print("✅ Dados gravados com sucesso!")
        return jsonify({"status": "sucesso"}), 200
    except Exception as e:
        print(f"❌ ERRO NO BANCO: {e}")
        return jsonify({"status": "erro", "message": str(e)}), 500


@app.route('/credenciais')
def mostrar_credenciais():
    return render_template('credenciais.html', 
                           num_proc=session.get('num_processo'), 
                           senha=session.get('senha_paciente'))


@app.route('/get_historico')
def get_historico():
    nome = request.args.get('nome', '')
    try:
        conn = conectar_banco()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT t.*, p.provincia 
            FROM triagens t
            JOIN pacientes p ON t.paciente_id = p.id
            WHERE p.nome LIKE %s
            ORDER BY t.id DESC
        """, (f"%{nome}%",))
        dados = cursor.fetchall()
        cursor.close()
        conn.close()
        # Converter datetimes para string
        for d in dados:
            for k, v in d.items():
                if hasattr(v, 'strftime'):
                    d[k] = v.strftime('%Y-%m-%dT%H:%M:%S')
        return jsonify(dados)
    except Exception as e:
        return jsonify([])


if __name__ == '__main__':
    threading.Thread(target=ler_arduino, daemon=True).start()
    threading.Thread(target=analisar_tosse, daemon=True).start()
    threading.Timer(2, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)