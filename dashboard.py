import sys
sys.path.append("../")

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import zmq
from plotly.subplots import make_subplots

from managers import data_manager

st.set_page_config(page_title="Dashboard", layout="wide")

# Rolling window — keeps memory bounded
MAX_POINTS = 1000


# --------------------------------------------------
# One-time session initialisation
# --------------------------------------------------

def _init_session() -> None:
    ctx = zmq.Context()

    sub_ped = ctx.socket(zmq.SUB)
    sub_ped.connect("tcp://192.168.0.102:5558")
    sub_ped.setsockopt_string(zmq.SUBSCRIBE, "TRACKS_ped")

    sub_lcc = ctx.socket(zmq.SUB)
    sub_lcc.connect("tcp://192.168.0.102:5556")
    sub_lcc.setsockopt_string(zmq.SUBSCRIBE, "lcc_SENSOR")

    sub_lcc_inf = ctx.socket(zmq.SUB)
    sub_lcc_inf.connect("tcp://192.168.0.102:5560")
    sub_lcc_inf.setsockopt_string(zmq.SUBSCRIBE, "lcc_INFERENCE")

    sub_rg = ctx.socket(zmq.SUB)
    sub_rg.connect("tcp://192.168.0.102:5557")
    sub_rg.setsockopt_string(zmq.SUBSCRIBE, "rg_SENSOR")

    poller = zmq.Poller()
    poller.register(sub_ped, zmq.POLLIN)
    poller.register(sub_lcc, zmq.POLLIN)
    poller.register(sub_rg, zmq.POLLIN)
    poller.register(sub_lcc_inf, zmq.POLLIN)

    ss = st.session_state
    ss.zmq_ctx = ctx
    ss.sub_ped = sub_ped
    ss.sub_lcc = sub_lcc
    ss.sub_lcc_inf = sub_lcc_inf
    ss.sub_rg = sub_rg
    ss.poller = poller
    ss.dashboard = data_manager.DataManager()

    ss.lcc_df = pd.DataFrame(columns=["timestamps", "tool_measurement"])
    ss.lcc_inf_df = pd.DataFrame(columns=["timestamps", "inference_values", "window", "og_window"])
    ss.rg_df = pd.DataFrame(columns=["timestamps", "tool_measurement"])
    ss.ped_df = pd.DataFrame(columns=["timestamps", "ped"])
    ss.temp_pressure_df = pd.DataFrame(columns=["timestamps", "temperature", "pressure"])


if "initialized" not in st.session_state:
    _init_session()
    st.session_state.initialized = True

ss = st.session_state


# --------------------------------------------------
# ZMQ polling
# --------------------------------------------------

def poll_zmq() -> dict:
    socks = dict(ss.poller.poll(timeout=1000))
    batch: dict = {}

    if ss.sub_ped in socks:
        topic, payload = ss.sub_ped.recv_string().split(" ", 1)
        data = json.loads(payload)
        if data:
            batch["ped"] = data["ped"]

    if ss.sub_lcc in socks:
        topic, payload = ss.sub_lcc.recv_string().split(" ", 1)
        batch["lcc"] = json.loads(payload)

    if ss.sub_lcc_inf in socks:
        topic, payload = ss.sub_lcc_inf.recv_string().split(" ", 1)
        batch["lcc_inference"] = json.loads(payload)

    if ss.sub_rg in socks:
        topic, payload = ss.sub_rg.recv_string().split(" ", 1)
        batch["rg"] = json.loads(payload)

    return batch


# --------------------------------------------------
# DataFrame helpers
# --------------------------------------------------

def _append_and_trim(existing: pd.DataFrame, new_rows: pd.DataFrame, sort_col: str) -> pd.DataFrame:
    combined = pd.concat([existing, new_rows], ignore_index=True)
    combined[sort_col] = pd.to_datetime(combined[sort_col])
    return combined.sort_values(sort_col).reset_index(drop=True).tail(MAX_POINTS)


def update_data(batch: dict) -> None:
    if "lcc" in batch:
        ss.lcc_df = _append_and_trim(ss.lcc_df, pd.DataFrame(batch["lcc"]), "timestamps")

        n = 5
        tp = pd.DataFrame({
            "timestamps": pd.date_range(end=pd.Timestamp.now(), periods=n, freq="s"),
            "temperature": np.random.normal(22, 0.1, n),
            "pressure": np.random.normal(1013, 0.75, n),
        })
        ss.temp_pressure_df = _append_and_trim(ss.temp_pressure_df, tp, "timestamps")

    if "lcc_inference" in batch:
        new = pd.DataFrame(batch["lcc_inference"])
        for col in ("inference_values", "window", "og_window"):
            if col in new.columns:
                new[col] = new[col].apply(
                    lambda x: x[0] if isinstance(x, (list, tuple)) and len(x) > 0 else x
                )
        ss.lcc_inf_df = _append_and_trim(ss.lcc_inf_df, new, "timestamps")

    if "rg" in batch:
        ss.rg_df = _append_and_trim(ss.rg_df, pd.DataFrame(batch["rg"]), "timestamps")

    if "ped" in batch:
        ss.ped_df = _append_and_trim(ss.ped_df, pd.DataFrame(batch["ped"]), "timestamps")


# --------------------------------------------------
# Figure builders
# --------------------------------------------------

def build_lcc_fig() -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if not ss.lcc_df.empty:
        fig.add_trace(
            go.Scatter(x=ss.lcc_df["timestamps"], y=ss.lcc_df["tool_measurement"],
                       mode="lines", name="LCC Measurement"),
            secondary_y=False,
        )
    if not ss.lcc_inf_df.empty:
        fig.add_trace(
            go.Scatter(x=ss.lcc_inf_df["timestamps"], y=ss.lcc_inf_df["inference_values"],
                       mode="lines", name="LCC Inference", line=dict(color="red")),
            secondary_y=True,
        )
    fig.update_yaxes(title_text="LCC Measurement (V)", range=[0, 2], secondary_y=False)
    fig.update_yaxes(title_text="LCC Inference", secondary_y=True)
    fig.update_layout(showlegend=True, title="LCC Logs")
    return fig


def build_rg_fig() -> go.Figure:
    fig = go.Figure()
    if not ss.rg_df.empty:
        fig.add_trace(go.Scatter(
            x=ss.rg_df["timestamps"], y=ss.rg_df["tool_measurement"],
            mode="lines", name="RG",
        ))
    fig.update_yaxes(title_text="RG Counts", range=[0, 20])
    fig.update_layout(title="RG Logs")
    return fig


def build_temp_pressure_fig() -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    df = ss.temp_pressure_df
    if not df.empty:
        fig.add_trace(
            go.Scatter(x=df["timestamps"], y=df["temperature"],
                       mode="lines+markers", name="Temperature (°C)"),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=df["timestamps"], y=df["pressure"],
                       mode="lines", name="Pressure (hPa)"),
            secondary_y=True,
        )
    fig.update_yaxes(title_text="Temperature (°C)", range=[20, 25], secondary_y=False)
    fig.update_yaxes(title_text="Pressure (hPa)", range=[1000, 1020], secondary_y=True)
    fig.update_layout(showlegend=True, title="Pressure & Temperature Logs")
    return fig


def build_ped_fig() -> go.Figure:
    fig = go.Figure()
    if not ss.ped_df.empty:
        fig.add_trace(go.Scatter(
            x=ss.ped_df["timestamps"], y=ss.ped_df["ped"],
            mode="lines", name="PED",
        ))
    fig.update_layout(title="PED Logs")
    return fig


# --------------------------------------------------
# Layout — rendered once on initial page load
# --------------------------------------------------

st.title("Dashboard")

left_col, right_col = st.columns(2)

with left_col:
    lcc_chart = st.empty()
    rg_chart = st.empty()

with right_col:
    temp_pressure_chart = st.empty()
    ped_chart = st.empty()


# --------------------------------------------------
# Streaming loop
# --------------------------------------------------

while True:
    batch = poll_zmq()
    if batch:
        update_data(batch)

    lcc_chart.plotly_chart(build_lcc_fig(), use_container_width=True, key="lcc_chart")
    rg_chart.plotly_chart(build_rg_fig(), use_container_width=True, key="rg_chart")
    temp_pressure_chart.plotly_chart(build_temp_pressure_fig(), use_container_width=True, key="temp_pressure_chart")
    ped_chart.plotly_chart(build_ped_fig(), use_container_width=True, key="ped_chart")
