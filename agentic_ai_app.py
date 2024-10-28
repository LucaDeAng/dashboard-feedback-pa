# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import random
from datetime import datetime

# Definizione di parametri e dati fittizi
AREAS = ["Area 1", "Area 2", "Area 3", "Area 4"]
INFRASTRUCTURE_TYPES = ["road", "electric grid", "water pipeline"]
ISSUE_TYPES = ["maintenance required", "overload", "normal"]

# Simulazione di una "coda di notifiche" per tenere traccia degli aggiornamenti
notifications = []

# Funzioni degli agenti
def process_citizen_report(area, infrastructure, description):
    """Processa la segnalazione del cittadino e genera dati simulati per la gestione."""
    data = {
        "timestamp": datetime.now(),
        "area": area,
        "infrastructure": infrastructure,
        "issue_type": "citizen report",
        "severity": random.randint(50, 100),
        "description": description
    }
    return predictive_analysis(data)

def predictive_analysis(data):
    """Esegue un'analisi predittiva per valutare i livelli di rischio."""
    if data["issue_type"] == "citizen report" or (data["issue_type"] == "maintenance required" and data["severity"] > 70):
        data["predicted_risk"] = "High"
    elif data["issue_type"] == "overload" and data["severity"] > 50:
        data["predicted_risk"] = "Medium"
    else:
        data["predicted_risk"] = "Low"
    return data

def allocate_resources(data):
    """Determina l'allocazione delle risorse in base al rischio previsto."""
    if data["predicted_risk"] == "High":
        data["resource_allocation"] = "Full maintenance team dispatched"
    elif data["predicted_risk"] == "Medium":
        data["resource_allocation"] = "Partial maintenance team dispatched"
    else:
        data["resource_allocation"] = "No action required"
    return data

def logistical_coordination(data):
    """Coordina le risorse logistiche in base all'allocazione e aggiorna le notifiche."""
    if data["resource_allocation"] != "No action required":
        data["logistics_plan"] = f"Transport to {data['area']} for {data['resource_allocation']}."
        notifications.append({
            "timestamp": datetime.now(),
            "message": f"Segnalazione per {data['infrastructure']} in {data['area']} presa in carico. Squadra inviata."
        })
    else:
        data["logistics_plan"] = "Standby, no logistics needed."
    return data

# Interfaccia utente
st.title("Sistema di Ottimizzazione delle Risorse per la PA con Agentic AI")
st.write("Simulazione del processo di raccolta, analisi e gestione risorse")

# Sezione per le segnalazioni dei cittadini
st.subheader("Segnalazione Anomalia Cittadina")
with st.form("citizen_report_form"):
    area = st.selectbox("Seleziona l'area", AREAS)
    infrastructure = st.selectbox("Tipo di Infrastruttura", INFRASTRUCTURE_TYPES)
    description = st.text_area("Descrizione del problema")
    submitted = st.form_submit_button("Invia Segnalazione")

    if submitted:
        citizen_data = process_citizen_report(area, infrastructure, description)
        citizen_data = allocate_resources(citizen_data)
        citizen_data = logistical_coordination(citizen_data)
        st.write("### Dettagli della segnalazione processata:")
        st.json(citizen_data)
        
        # Aggiornamento notifiche
        if citizen_data["resource_allocation"] != "No action required":
            st.success("Comunicazione inviata alla ditta per risolvere l'anomalia.")
            notifications.append({
                "timestamp": datetime.now(),
                "message": f"Aggiornamento: Intervento per {infrastructure} in {area} programmato."
            })
        else:
            st.info("Segnalazione registrata, ma non richiede intervento immediato.")

# Visualizzazione delle notifiche per il cittadino
st.subheader("Notifiche di Aggiornamento")
if notifications:
    for note in notifications:
        st.write(f"{note['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} - {note['message']}")
else:
    st.write("Nessuna notifica disponibile.")
