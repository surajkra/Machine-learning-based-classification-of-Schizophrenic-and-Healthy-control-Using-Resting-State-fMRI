import sys
sys.path.append("../")

import json
import threading

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import zmq
from plotly.subplots import make_subplots

from managers import data_manager

st.set_page_config(page_title="Dashboard", layout="wide")

MAX_POINTS = 1000


# --------------------------------------------------
# Thread-safe data store
# --------------------------------------------------

class DataStore:
    def __init__(self):
        self._lock = threading.Lock()
        self.lcc_df = pd.DataFrame(columns=["timestamps", "tool_measurement"])
        self.lcc_inf_df = pd.DataFrame(columns=["timestamps", "inference_values", "window", "og_window"])
        self.rg_df = pd.DataFrame(columns=["timestamps", "tool_measurement"])
        self.ped_df = pd.DataFrame(columns=["timestamps", "ped"])
        self.temp_pressure_df = pd.DataFrame(columns=["timestamps", "temperature", "pressure"])
        self.dashboard = data_manager.DataManager()

    def update(self, batch: dict) -> None:
        with self._lock:
            if "lcc" in batch:
                self.lcc_df = _append_and_trim(
                    self.lcc_df, pd.DataFrame(batch["lcc"]), "timestamps"
                )
                n = 5
                tp = pd.DataFrame({
                    "timestamps": pd.date_range(end=pd.Timestamp.now(), periods=n, freq="s"),
                    "temperature": np.random.normal(22, 0.1, n),
                    "pressure": np.random.normal(1013, 0.75, n),
                })
                self.temp_pressure_df = _append_and_trim(self.temp_pressure_df, tp, "timestamps")

            if "lcc_inference" in batch:
                new = pd.DataFrame(batch["lcc_inference"])
                for col in ("inference_values", "window", "og_window"):
                    if col in new.columns:
                        new[col] = new[col].apply(
                            lambda x: x[0] if isinstance(x, (list, tuple)) and len(x) > 0 else x
                        )
                self.lcc_inf_df = _append_and_trim(self.lcc_inf_df, new, "timestamps")

            if "rg" in batch:
                self.rg_df = _append_and_trim(
                    self.rg_df, pd.DataFrame(batch["rg"]), "timestamps"
                )

            if "ped" in batch:
                self.ped_df = _append_and_trim(
                    self.ped_df, pd.DataFrame(batch["ped"]), "timestamps"
                )

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "lcc_df": self.lcc_df.copy(),
                "lcc_inf_df": self.lcc_inf_df.copy(),
                "rg_df": self.rg_df.copy(),
                "ped_df": self.ped_df.copy(),
                "temp_pressure_df": self.temp_pressure_df.copy(),
            }


def _append_and_trim(existing: pd.DataFrame, new_rows: pd.DataFrame, sort_col: str) -> pd.DataFrame:
    combined = pd.concat([existing, new_rows], ignore_index=True)
    combined[sort_col] = pd.to_datetime(combined[sort_col])
    return combined.sort_values(sort_col).reset_index(drop=True).tail(MAX_POINTS)


# --------------------------------------------------
# ZMQ background worker — never touches Streamlit
# --------------------------------------------------

def _zmq_worker(store: DataStore) -> None:
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

    while True:
        socks = dict(poller.poll(timeout=100))
        batch = {}

        if sub_ped in socks:
            topic, payload = sub_ped.recv_string().split(" ", 1)
            data = json.loads(payload)
            if data:
                batch["ped"] = data["ped"]

        if sub_lcc in socks:
            topic, payload = sub_lcc.recv_string().split(" ", 1)
            batch["lcc"] = json.loads(payload)

        if sub_lcc_inf in socks:
            topic, payload = sub_lcc_inf.recv_string().split(" ", 1)
            batch["lcc_inference"] = json.loads(payload)

        if sub_rg in socks:
            topic, payload = sub_rg.recv_string().split(" ", 1)
            batch["rg"] = json.loads(payload)

        if batch:
            store.update(batch)


@st.cache_resource
def get_store() -> DataStore:
    """Runs once per process. Starts the ZMQ thread and returns the shared store."""
    store = DataStore()
    thread = threading.Thread(target=_zmq_worker, args=(store,), daemon=True)
    thread.start()
    return store


# --------------------------------------------------
# Figure builders — pure functions, no side effects
# --------------------------------------------------

def build_lcc_fig(snap: dict) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if not snap["lcc_df"].empty:
        fig.add_trace(
            go.Scatter(x=snap["lcc_df"]["timestamps"], y=snap["lcc_df"]["tool_measurement"],
                       mode="lines", name="LCC Measurement"),
            secondary_y=False,
        )
    if not snap["lcc_inf_df"].empty:
        fig.add_trace(
            go.Scatter(x=snap["lcc_inf_df"]["timestamps"], y=snap["lcc_inf_df"]["inference_values"],
                       mode="lines", name="LCC Inference", line=dict(color="red")),
            secondary_y=True,
        )
    fig.update_yaxes(title_text="LCC Measurement (V)", range=[0, 2], secondary_y=False)
    fig.update_yaxes(title_text="LCC Inference", secondary_y=True)
    fig.update_layout(showlegend=True, title="LCC Logs")
    return fig


def build_rg_fig(snap: dict) -> go.Figure:
    fig = go.Figure()
    if not snap["rg_df"].empty:
        fig.add_trace(go.Scatter(
            x=snap["rg_df"]["timestamps"], y=snap["rg_df"]["tool_measurement"],
            mode="lines", name="RG",
        ))
    fig.update_yaxes(title_text="RG Counts", range=[0, 20])
    fig.update_layout(title="RG Logs")
    return fig


def build_temp_pressure_fig(snap: dict) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    df = snap["temp_pressure_df"]
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


def build_ped_fig(snap: dict) -> go.Figure:
    fig = go.Figure()
    if not snap["ped_df"].empty:
        fig.add_trace(go.Scatter(
            x=snap["ped_df"]["timestamps"], y=snap["ped_df"]["ped"],
            mode="lines", name="PED",
        ))
    fig.update_layout(title="PED Logs")
    return fig


# --------------------------------------------------
# App
# --------------------------------------------------

store = get_store()

st.title("Dashboard")


@st.fragment(run_every=1)
def live_charts() -> None:
    snap = store.snapshot()

    left_col, right_col = st.columns(2)
    with left_col:
        st.plotly_chart(build_lcc_fig(snap), use_container_width=True, key="lcc_chart")
        st.plotly_chart(build_rg_fig(snap), use_container_width=True, key="rg_chart")
    with right_col:
        st.plotly_chart(build_temp_pressure_fig(snap), use_container_width=True, key="temp_pressure_chart")
        st.plotly_chart(build_ped_fig(snap), use_container_width=True, key="ped_chart")


live_charts()
