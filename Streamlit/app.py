"""
Aplicación de gestión de tickets — versión con roles (usuario / soporte).

Requiere streamlit >= 1.35 (es necesario usar `on_select` en st.dataframe para poder abrir
un ticket haciendo clic en su fila, y st.container(border=True) para las
tarjetas, disponible desde 1.31).

"""

import json
import re
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
DB_PATH = Path(__file__).parent / "tickets_calibrado.db"

st.set_page_config(page_title="Gestión de tickets", layout="wide")


# Estilos -----------------------------
CSS = """
<style>
.stApp {
    background-color: #f6f7fb;
}
h1, h2, h3, h4 { letter-spacing: -0.01em; }

/* Tarjetas (st.container(border=True)) */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 14px !important;
    box-shadow: 0 1px 8px rgba(30, 41, 59, 0.05);
}

/* Botón principal en verde en vez del rojo por defecto */
button[kind="primary"],
[data-testid="stBaseButton-primary"] {
    background-color: #16a34a !important;
    border-color: #16a34a !important;
    color: #ffffff !important;
}
button[kind="primary"]:hover,
[data-testid="stBaseButton-primary"]:hover {
    background-color: #15803d !important;
    border-color: #15803d !important;
    color: #ffffff !important;
}

/* Pills de confianza / estado */
.tk-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.8rem;
}
.tk-badge-alta  { background: #e6f9ec; color: #1e7d34; }
.tk-badge-media { background: #fff8e1; color: #a16207; }
.tk-badge-baja  { background: #fdecea; color: #c0392b; }
.tk-badge-neutro { background: #eef0f6; color: #4b5563; }
.tk-badge-modificado { background: #e8edfb; color: #3448a3; }

.tk-caption { color: #6b7280; font-size: 0.85rem; }
.tk-label { color: #6b7280; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.03em; }
.tk-pred-value { font-weight: 600; font-size: 0.98rem; margin: 2px 0 6px 0; }
</style>
"""


# Artefactos del modelo -----------------------------

@st.cache_resource
def cargar_artefactos():
    faltantes = [
        f for f in [
            "model_type_calibrated.joblib", "model_priority_calibrated.joblib",
            "model_queue_calibrated.joblib",
            "tfidf_201.joblib", "tfidf_1000.joblib",
            "tags_validos_201.json", "tags_validos_1000.json", "tags_disponibles.json",
            "thresholds.json",
        ]
        if not (ARTIFACTS_DIR / f).exists()
    ]
    if faltantes:
        st.error(
            "Faltan artefactos en ./artifacts: " + ", ".join(faltantes) +
            ". Ejecuta primero los chunks de calibración en el notebook."
        )
        st.stop()

    with open(ARTIFACTS_DIR / "tags_validos_201.json") as f:
        tags_validos_201 = set(json.load(f))
    with open(ARTIFACTS_DIR / "tags_validos_1000.json") as f:
        tags_validos_1000 = set(json.load(f))
    with open(ARTIFACTS_DIR / "tags_disponibles.json") as f:
        tags_disponibles = sorted(json.load(f))
    with open(ARTIFACTS_DIR / "thresholds.json") as f:
        thresholds = json.load(f)

    model_type = joblib.load(ARTIFACTS_DIR / "model_type_calibrated.joblib")
    model_priority = joblib.load(ARTIFACTS_DIR / "model_priority_calibrated.joblib")
    model_queue = joblib.load(ARTIFACTS_DIR / "model_queue_calibrated.joblib")

    return {
        "model_type": model_type,
        "model_priority": model_priority,
        "model_queue": model_queue,
        "tfidf_201": joblib.load(ARTIFACTS_DIR / "tfidf_201.joblib"),
        "tfidf_1000": joblib.load(ARTIFACTS_DIR / "tfidf_1000.joblib"),
        "tags_validos_201": tags_validos_201,
        "tags_validos_1000": tags_validos_1000,
        "tags_disponibles": tags_disponibles,
        "thresholds": thresholds,
        "type_classes": sorted(model_type.classes_.tolist()),
        "priority_classes": sorted(model_priority.classes_.tolist()),
        "queue_classes": sorted(model_queue.classes_.tolist()),
    }


TYPE_LABELS = {
    "incident_problem": "Incident / Problem",
    "request": "Request",
    "change": "Change",
}


def etiqueta_type(valor) -> str:
    if not valor:
        return "—"
    return TYPE_LABELS.get(valor, valor)


def etiqueta_priority(valor) -> str:
    if not valor:
        return "—"
    return valor.title()


def etiqueta_queue(valor) -> str:
    if not valor:
        return "—"
    return valor.replace("_", " ").title()


# Preprocesado + predicción (idéntico a versiones anteriores)-----------------------------

def limpiar_texto(texto: str) -> str:
    if texto is None:
        return ""
    texto = str(texto).strip().lower()
    texto = BeautifulSoup(texto, "html.parser").get_text(" ", strip=True)
    return texto


def construir_text_combined(subject, body, tags, tags_validos) -> str:
    subject_limpio = limpiar_texto(subject)
    body_limpio = limpiar_texto(body)
    tags_limpios = [t.strip().lower() for t in tags]
    tags_filtrados = [t for t in tags_limpios if t in tags_validos]
    texto = f"{subject_limpio} {body_limpio} {' '.join(tags_filtrados)}"
    return re.sub(r"\s+", " ", texto).strip()


def predecir(subject, body, tags, art):
    """Devuelve {"type": (clase, confianza), "priority": (...), "queue": (...)}."""
    texto_201 = construir_text_combined(subject, body, tags, art["tags_validos_201"])
    texto_1000 = construir_text_combined(subject, body, tags, art["tags_validos_1000"])

    X_201 = art["tfidf_201"].transform([texto_201])
    X_1000 = art["tfidf_1000"].transform([texto_1000])

    resultado = {}
    for objetivo, modelo, X in [
        ("type", art["model_type"], X_201),
        ("priority", art["model_priority"], X_201),
        ("queue", art["model_queue"], X_1000),
    ]:
        proba = modelo.predict_proba(X)[0]
        idx = proba.argmax()
        resultado[objetivo] = (modelo.classes_[idx], float(proba[idx]))
    return resultado


def nivel_confianza(valor: float, umbrales_objetivo: dict) -> str:
    if valor >= umbrales_objetivo["alta"]:
        return "alta"
    if valor >= umbrales_objetivo["media"]:
        return "media"
    return "baja"


CSS_CELDA_NIVEL = {
    "alta": "background-color: #e6f9ec; color: #1e7d34; font-weight: 600;",
    "media": "background-color: #fff8e1; color: #a16207; font-weight: 600;",
    "baja": "background-color: #fdecea; color: #c0392b; font-weight: 600;",
}


# Persistencia (SQLite) -----------------------------

COLUMNAS_TICKET = [
    "ticket_id", "user_id", "subject", "body", "tags",
    "pred_type", "pred_priority", "pred_queue",
    "confidence_type", "confidence_priority", "confidence_queue",
    "created_at",
]

# tickets_resueltos añade, además, la predicción automática original
# (antes de que el técnico la corrigiera, si es que la corrigió), para
# poder mostrar si el ticket fue modificado y qué cambió.
COLUMNAS_ORIGINALES = ["pred_type_original", "pred_priority_original", "pred_queue_original"]


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()
    definicion_base = """
        ticket_id TEXT PRIMARY KEY,
        user_id TEXT,
        subject TEXT,
        body TEXT,
        tags TEXT,
        pred_type TEXT,
        pred_priority TEXT,
        pred_queue TEXT,
        confidence_type REAL,
        confidence_priority REAL,
        confidence_queue REAL,
        created_at TEXT
    """
    for tabla in ("tickets_totales", "tickets_pendientes"):
        cur.execute(f"CREATE TABLE IF NOT EXISTS {tabla} ({definicion_base})")

    definicion_resueltos = definicion_base + """,
        pred_type_original TEXT,
        pred_priority_original TEXT,
        pred_queue_original TEXT
    """
    cur.execute(f"CREATE TABLE IF NOT EXISTS tickets_resueltos ({definicion_resueltos})")

    # Migración suave por si existe una BD de una versión anterior sin
    # estas columnas.
    for tabla, columnas_nuevas in (
        ("tickets_totales", ["user_id"]),
        ("tickets_pendientes", ["user_id"]),
        ("tickets_resueltos", ["user_id"] + COLUMNAS_ORIGINALES),
    ):
        columnas_existentes = {fila[1] for fila in cur.execute(f"PRAGMA table_info({tabla})")}
        for columna in columnas_nuevas:
            if columna not in columnas_existentes:
                cur.execute(f"ALTER TABLE {tabla} ADD COLUMN {columna} TEXT")

    conn.commit()
    conn.close()


def crear_ticket_en_bd(user_id, subject, body, tags, predicciones):
    ticket_id = "TCK-" + uuid.uuid4().hex[:8].upper()
    creado = datetime.now().isoformat(timespec="seconds")
    tags_str = ",".join(tags)

    pred_type, conf_type = predicciones["type"]
    pred_priority, conf_priority = predicciones["priority"]
    pred_queue, conf_queue = predicciones["queue"]

    valores = (
        ticket_id, user_id, subject, body, tags_str,
        pred_type, pred_priority, pred_queue,
        conf_type, conf_priority, conf_queue,
        creado,
    )

    conn = get_conn()
    cur = conn.cursor()
    placeholders = ",".join(["?"] * len(COLUMNAS_TICKET))
    for tabla in ("tickets_totales", "tickets_pendientes"):
        cur.execute(
            f"INSERT INTO {tabla} ({','.join(COLUMNAS_TICKET)}) VALUES ({placeholders})",
            valores,
        )
    conn.commit()
    conn.close()
    return ticket_id


def obtener_pendientes(tipo=None, priority=None, queue=None):
    """Todos los tickets pendientes, con filtros opcionales."""
    conn = get_conn()
    query = "SELECT * FROM tickets_pendientes WHERE 1=1"
    params = []
    if tipo and tipo != "Todas":
        query += " AND pred_type = ?"
        params.append(tipo)
    if priority and priority != "Todas":
        query += " AND pred_priority = ?"
        params.append(priority)
    if queue and queue != "Todas":
        query += " AND pred_queue = ?"
        params.append(queue)
    query += " ORDER BY created_at DESC"
    filas = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(f) for f in filas]


def obtener_ticket_pendiente(ticket_id):
    conn = get_conn()
    fila = conn.execute(
        "SELECT * FROM tickets_pendientes WHERE ticket_id = ?", (ticket_id,)
    ).fetchone()
    conn.close()
    return dict(fila) if fila else None


def resolver_ticket(ticket_id, tipo_final, priority_final, queue_final):
    """
    Guarda en tickets_resueltos tanto la clasificación FINAL (pred_type,
    pred_priority, pred_queue) como la clasificación automática original
    (pred_*_original), para poder mostrar después si el técnico corrigió
    algo. confidence_* sigue representando siempre la predicción
    automática original, no la corrección del técnico.
    """
    conn = get_conn()
    cur = conn.cursor()
    fila = cur.execute(
        "SELECT * FROM tickets_pendientes WHERE ticket_id = ?", (ticket_id,)
    ).fetchone()
    if fila is None:
        conn.close()
        raise ValueError(f"El ticket {ticket_id} ya no está pendiente.")

    fila = dict(fila)
    originales = {
        "pred_type_original": fila["pred_type"],
        "pred_priority_original": fila["pred_priority"],
        "pred_queue_original": fila["pred_queue"],
    }
    fila["pred_type"] = tipo_final
    fila["pred_priority"] = priority_final
    fila["pred_queue"] = queue_final

    try:
        columnas = COLUMNAS_TICKET + COLUMNAS_ORIGINALES
        valores = [fila[c] for c in COLUMNAS_TICKET] + [originales[c] for c in COLUMNAS_ORIGINALES]
        placeholders = ",".join(["?"] * len(columnas))
        cur.execute(
            f"INSERT INTO tickets_resueltos ({','.join(columnas)}) VALUES ({placeholders})",
            valores,
        )
        cur.execute("DELETE FROM tickets_pendientes WHERE ticket_id = ?", (ticket_id,))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def obtener_resueltos():
    conn = get_conn()
    filas = conn.execute("SELECT * FROM tickets_resueltos ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(f) for f in filas]


def obtener_ticket_resuelto(ticket_id):
    conn = get_conn()
    fila = conn.execute(
        "SELECT * FROM tickets_resueltos WHERE ticket_id = ?", (ticket_id,)
    ).fetchone()
    conn.close()
    return dict(fila) if fila else None


def tiene_info_original(ticket: dict) -> bool:
    """False para tickets resueltos con una versión anterior de la app,
    que no guardaba la clasificación automática original."""
    return all(ticket.get(c) is not None for c in COLUMNAS_ORIGINALES)


def ticket_fue_modificado(ticket: dict):
    """True/False si se puede saber, o None si el ticket es de una
    versión anterior y no se guardó la clasificación original."""
    if not tiene_info_original(ticket):
        return None
    return (
        ticket.get("pred_type") != ticket.get("pred_type_original")
        or ticket.get("pred_priority") != ticket.get("pred_priority_original")
        or ticket.get("pred_queue") != ticket.get("pred_queue_original")
    )


# Componentes de UI reutilizables -----------------------------

def badge_confianza_html(valor: float, umbrales_objetivo: dict) -> str:
    nivel = nivel_confianza(valor, umbrales_objetivo)
    etiqueta = {"alta": "Alta", "media": "Media", "baja": "Baja"}[nivel]
    return f"<span class='tk-badge tk-badge-{nivel}'>{etiqueta} · {valor * 100:.0f}%</span>"


def panel_predicciones_html(ticket: dict, thresholds: dict) -> str:
    """HTML compacto con Type / Priority / Queue y su badge de confianza,
    todo en una única llamada a st.markdown (evita recuadros fantasma)."""
    filas_html = []
    campos = [
        ("Type", etiqueta_type(ticket["pred_type"]), ticket["confidence_type"], thresholds["type"]),
        ("Priority", etiqueta_priority(ticket["pred_priority"]), ticket["confidence_priority"], thresholds["priority"]),
        ("Queue", etiqueta_queue(ticket["pred_queue"]), ticket["confidence_queue"], thresholds["queue"]),
    ]
    for nombre, valor_mostrado, confianza, umbral in campos:
        filas_html.append(
            f"<div style='margin-bottom:12px;'>"
            f"<div class='tk-label'>{nombre}</div>"
            f"<div class='tk-pred-value'>{valor_mostrado}</div>"
            f"{badge_confianza_html(confianza, umbral)}"
            f"</div>"
        )
    return "".join(filas_html)


def construir_tabla_estilada(tickets: list[dict], thresholds: dict, con_color: bool = True):
    """Construye un DataFrame con estilo (colores por nivel de confianza,
    si con_color=True) o, para resueltos, con una columna de estado."""
    filas = []
    for t in tickets:
        fila = {
            "Ticket ID": t["ticket_id"],
            "Subject": (t["subject"] or "")[:70] + ("…" if t["subject"] and len(t["subject"]) > 70 else ""),
            "Type": etiqueta_type(t["pred_type"]),
            "Priority": etiqueta_priority(t["pred_priority"]),
            "Queue": etiqueta_queue(t["pred_queue"]),
            "Creado": t["created_at"],
        }
        if con_color:
            fila["__conf_type"] = t["confidence_type"]
            fila["__conf_priority"] = t["confidence_priority"]
            fila["__conf_queue"] = t["confidence_queue"]
        else:
            modificado = ticket_fue_modificado(t)
            if modificado is None:
                fila["Estado"] = "Sin datos"
            else:
                fila["Estado"] = "Modificado" if modificado else "Sin cambios"
        filas.append(fila)
    df = pd.DataFrame(filas)

    if not con_color:
        return df, df.style

    def estilo_fila(row):
        estilos = []
        for col in row.index:
            if col == "Type":
                estilos.append(CSS_CELDA_NIVEL[nivel_confianza(row["__conf_type"], thresholds["type"])])
            elif col == "Priority":
                estilos.append(CSS_CELDA_NIVEL[nivel_confianza(row["__conf_priority"], thresholds["priority"])])
            elif col == "Queue":
                estilos.append(CSS_CELDA_NIVEL[nivel_confianza(row["__conf_queue"], thresholds["queue"])])
            else:
                estilos.append("")
        return estilos

    styled = df.style.apply(estilo_fila, axis=1).hide(
        axis="columns", subset=["__conf_type", "__conf_priority", "__conf_queue"]
    )
    return df, styled


# Portal de usuario-----------------------------

def portal_usuario(art):
    st.title("Crear un ticket")
    st.info("Nota: por favor, redacta el asunto y la descripción del ticket en inglés.")

    if "form_key_counter" not in st.session_state:
        st.session_state.form_key_counter = 0
    if "ultimo_ticket_creado" not in st.session_state:
        st.session_state.ultimo_ticket_creado = None

    if st.session_state.ultimo_ticket_creado:
        with st.container(border=True):
            st.success(f"Ticket {st.session_state.ultimo_ticket_creado} registrado correctamente.")
            if st.button("Crear otro ticket", type="primary"):
                st.session_state.ultimo_ticket_creado = None
                st.rerun()
        return

    k = st.session_state.form_key_counter
    with st.container(border=True):
        user_id = st.text_input("ID de usuario", key=f"user_id_{k}")
        subject = st.text_input("Subject", key=f"subject_{k}")
        body = st.text_area("Body", height=180, key=f"body_{k}")
        tags = st.multiselect("Tags", options=art["tags_disponibles"], key=f"tags_{k}")

        if st.button("Enviar ticket", type="primary"):
            if not user_id.strip():
                st.warning("Introduce tu ID de usuario.")
            elif not subject.strip() and not body.strip():
                st.warning("Introduce al menos un subject o un body.")
            else:
                predicciones = predecir(subject, body, tags, art)
                ticket_id = crear_ticket_en_bd(user_id.strip(), subject, body, tags, predicciones)
                st.session_state.ultimo_ticket_creado = ticket_id
                st.session_state.form_key_counter += 1
                st.rerun()


# Portal de técnico — Tickets a resolver -----------------------------

def vista_detalle_pendiente(art, ticket_id):
    ticket = obtener_ticket_pendiente(ticket_id)

    if st.button("Volver a la lista"):
        st.session_state.ticket_abierto_pendiente = None
        st.session_state.pop("tabla_pendientes", None)
        st.rerun()

    if ticket is None:
        st.warning("Este ticket ya no está pendiente (puede que ya se haya resuelto).")
        return

    with st.container(border=True):
        st.subheader(f"Ticket {ticket['ticket_id']}")
        st.caption("Clasificación generada automáticamente. Revísala y corrígela si hace falta.")

        col_info, col_pred = st.columns([3, 2])
        with col_info:
            st.markdown(
                f"- **User ID:** {ticket.get('user_id') or '—'}\n"
                f"- **Subject:** {ticket['subject'] or '—'}\n"
                f"- **Body:** {ticket['body'] or '—'}\n"
                f"- **Tags:** {ticket['tags'] or '—'}\n"
                f"- **Creado:** {ticket['created_at']}"
            )
        with col_pred:
            st.markdown(panel_predicciones_html(ticket, art["thresholds"]), unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("**Corrige la clasificación si hace falta:**")
        c1, c2, c3 = st.columns(3)
        with c1:
            type_final = st.selectbox(
                "Type", art["type_classes"],
                index=art["type_classes"].index(ticket["pred_type"]),
                format_func=etiqueta_type,
            )
        with c2:
            priority_final = st.selectbox(
                "Priority", art["priority_classes"],
                index=art["priority_classes"].index(ticket["pred_priority"]),
                format_func=etiqueta_priority,
            )
        with c3:
            queue_final = st.selectbox(
                "Queue", art["queue_classes"],
                index=art["queue_classes"].index(ticket["pred_queue"]),
                format_func=etiqueta_queue,
            )

        if st.button("Marcar ticket como resuelto", type="primary"):
            resolver_ticket(ticket["ticket_id"], type_final, priority_final, queue_final)
            st.session_state.ticket_abierto_pendiente = None
            st.session_state.pop("tabla_pendientes", None)
            st.success(f"Ticket {ticket['ticket_id']} resuelto.")
            st.rerun()


def vista_lista_pendientes(art):
    st.title("Tickets a resolver")

    col1, col2, col3 = st.columns(3)
    with col1:
        type_filtro = st.selectbox(
            "Type", ["Todas"] + art["type_classes"], format_func=lambda v: v if v == "Todas" else etiqueta_type(v)
        )
    with col2:
        priority_filtro = st.selectbox(
            "Priority", ["Todas"] + art["priority_classes"], format_func=lambda v: v if v == "Todas" else etiqueta_priority(v)
        )
    with col3:
        queue_filtro = st.selectbox(
            "Queue", ["Todas"] + art["queue_classes"], format_func=lambda v: v if v == "Todas" else etiqueta_queue(v)
        )

    pendientes = obtener_pendientes(type_filtro, priority_filtro, queue_filtro)

    if not pendientes:
        st.info("No hay tickets pendientes con esos filtros.")
        return

    df, styled = construir_tabla_estilada(pendientes, art["thresholds"], con_color=True)

    evento = st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="tabla_pendientes",
    )

    st.caption("Verde: confianza alta · Amarillo: confianza media · Rojo: confianza baja. Haz clic en una fila para abrir el ticket.")

    if evento.selection.rows:
        fila_idx = evento.selection.rows[0]
        ticket_id_click = df.iloc[fila_idx]["Ticket ID"]
        st.session_state.ticket_abierto_pendiente = ticket_id_click
        st.rerun()


def vista_pendientes(art):
    if st.session_state.get("ticket_abierto_pendiente"):
        vista_detalle_pendiente(art, st.session_state.ticket_abierto_pendiente)
    else:
        vista_lista_pendientes(art)


# Portal de técnico — Tickets resueltos -----------------------------

def vista_detalle_resuelto(ticket_id):
    if st.button("Volver a la lista"):
        st.session_state.ticket_abierto_resuelto = None
        st.session_state.pop("tabla_resueltos", None)
        st.rerun()

    ticket = obtener_ticket_resuelto(ticket_id)
    if ticket is None:
        st.warning("No se ha encontrado este ticket.")
        return

    modificado = ticket_fue_modificado(ticket)

    with st.container(border=True):
        st.subheader(f"Ticket {ticket['ticket_id']}")
        if modificado is None:
            badge = "<span class='tk-badge tk-badge-neutro'>Clasificación original no disponible</span>"
        elif modificado:
            badge = "<span class='tk-badge tk-badge-modificado'>Modificado por el técnico</span>"
        else:
            badge = "<span class='tk-badge tk-badge-neutro'>Sin cambios</span>"
        st.markdown(badge, unsafe_allow_html=True)

        st.markdown(
            f"- **User ID:** {ticket.get('user_id') or '—'}\n"
            f"- **Subject:** {ticket['subject'] or '—'}\n"
            f"- **Body:** {ticket['body'] or '—'}\n"
            f"- **Tags:** {ticket['tags'] or '—'}\n"
            f"- **Creado:** {ticket['created_at']}"
        )

        st.markdown("---")

        if modificado is None:
            st.caption("Este ticket ha sido resuelto con una versión anterior de la app que no guardaba la clasificación automática original, por lo que no se puede comparar.")
        elif modificado:
            cambios = []
            if ticket["pred_type"] != ticket["pred_type_original"]:
                cambios.append(f"- **Type:** {etiqueta_type(ticket['pred_type_original'])} → {etiqueta_type(ticket['pred_type'])}")
            if ticket["pred_priority"] != ticket["pred_priority_original"]:
                cambios.append(f"- **Priority:** {etiqueta_priority(ticket['pred_priority_original'])} → {etiqueta_priority(ticket['pred_priority'])}")
            if ticket["pred_queue"] != ticket["pred_queue_original"]:
                cambios.append(f"- **Queue:** {etiqueta_queue(ticket['pred_queue_original'])} → {etiqueta_queue(ticket['pred_queue'])}")
            st.markdown("**Cambios realizados por el técnico:**\n" + "\n".join(cambios))
        else:
            st.markdown("El técnico no ha modificado la clasificación automática.")

        st.caption(
            f"Confianza automática original — "
            f"Type: {ticket['confidence_type'] * 100:.0f}% · "
            f"Priority: {ticket['confidence_priority'] * 100:.0f}% · "
            f"Queue: {ticket['confidence_queue'] * 100:.0f}%"
        )


def vista_lista_resueltos(art):
    st.title("Tickets resueltos")

    resueltos = obtener_resueltos()
    if not resueltos:
        st.info("Todavía no hay tickets resueltos.")
        return

    df, styled = construir_tabla_estilada(resueltos, art["thresholds"], con_color=False)

    evento = st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="tabla_resueltos",
    )
    st.caption("Haz clic en una fila para ver el detalle del ticket.")

    if evento.selection.rows:
        fila_idx = evento.selection.rows[0]
        ticket_id_click = df.iloc[fila_idx]["Ticket ID"]
        st.session_state.ticket_abierto_resuelto = ticket_id_click
        st.rerun()


def vista_resueltos(art):
    if st.session_state.get("ticket_abierto_resuelto"):
        vista_detalle_resuelto(st.session_state.ticket_abierto_resuelto)
    else:
        vista_lista_resueltos(art)


# Portal de técnico — navegación -----------------------------

def portal_tecnico(art):
    st.sidebar.title("Soporte técnico")
    seccion = st.sidebar.radio("Navegación", ["Tickets a resolver", "Tickets resueltos"])

    if seccion == "Tickets a resolver":
        vista_pendientes(art)
    else:
        vista_resueltos(art)


# Login / selección de rol -----------------------------

def pantalla_login():
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<h1 style='text-align:center;'>Gestión de tickets</h1>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    _, c1, c2, _ = st.columns([1, 2, 2, 1])
    with c1:
        with st.container(border=True):
            st.markdown(
                "<h4 style='text-align:center;'>Usuario</h4>"
                "<p class='tk-caption' style='text-align:center;'>Crea un ticket de soporte.</p>",
                unsafe_allow_html=True,
            )
            if st.button("Entrar como usuario", use_container_width=True):
                st.session_state.role = "usuario"
                st.rerun()
    with c2:
        with st.container(border=True):
            st.markdown(
                "<h4 style='text-align:center;'>Técnico de soporte</h4>"
                "<p class='tk-caption' style='text-align:center;'>Revisa y resuelve los tickets.</p>",
                unsafe_allow_html=True,
            )
            if st.button("Entrar como técnico", use_container_width=True):
                st.session_state.role = "tecnico"
                st.rerun()


# Main -----------------------------

def main():
    st.markdown(CSS, unsafe_allow_html=True)
    init_db()
    art = cargar_artefactos()

    if "role" not in st.session_state:
        st.session_state.role = None

    if st.session_state.role is None:
        pantalla_login()
        return

    if st.session_state.role == "usuario":
        with st.sidebar:
            st.markdown("### Usuario")
            if st.button("Cambiar de rol"):
                st.session_state.role = None
                st.session_state.ultimo_ticket_creado = None
                st.rerun()
        portal_usuario(art)
    else:
        with st.sidebar:
            if st.button("Cambiar de rol"):
                st.session_state.role = None
                st.session_state.ticket_abierto_pendiente = None
                st.session_state.ticket_abierto_resuelto = None
                st.rerun()
        portal_tecnico(art)


if __name__ == "__main__":
    main()
