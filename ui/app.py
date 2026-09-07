import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st
import pandas as pd
from sklearn.datasets import make_classification

from data.dataset import from_arrays
from data.corruptor import inject_label_noise
from models.downstream_model import DownstreamModel
from models.state_encoder import StateEncoder
from rl.ppo_agent import PPOAgent
from rl.env import ARGOEnv
from argo_selectors.remove_selector import RemoveSelector
from argo_selectors.relabel_selector import RelabelSelector
from argo_selectors.augment_selector import AugmentSelector
from argo_selectors.repair_selector import RepairSelector


st.set_page_config(page_title="ARGO MVP", layout="wide")
st.title("ARGO MVP: Interactive Data Governance")

if "env" not in st.session_state:
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=12,
        n_redundant=4,
        n_classes=3,
        weights=[0.65, 0.25, 0.10],
        random_state=42,
    )
    y_noisy, _ = inject_label_noise(y, noise_ratio=0.25, n_classes=3)
    dataset = from_arrays(X, y_noisy)

    actions = ["noop", "remove", "augment", "relabel", "repair"]
    model = DownstreamModel()
    encoder = StateEncoder(action_dim=len(actions), history_len=5)
    selectors = {
        "remove": RemoveSelector(),
        "augment": AugmentSelector(),
        "relabel": RelabelSelector(),
        "repair": RepairSelector(),
    }
    agent = PPOAgent(actions, epsilon=0.25)
    env = ARGOEnv(dataset, model, encoder, selectors, budget_ratio=0.03)

    st.session_state.env = env
    st.session_state.agent = agent
    st.session_state.logs = []

env = st.session_state.env
agent = st.session_state.agent

col1, col2, col3 = st.columns(3)
col1.metric("Train Size", len(env.dataset.y_train))
col2.metric("Val Accuracy", f"{env.model.score(env.dataset.X_val, env.dataset.y_val):.4f}")
col3.metric("Classes", env.dataset.n_classes)

st.subheader("Run One Governance Step")
manual = st.selectbox("Action", agent.action_names)
auto = st.checkbox("Auto Mode", value=True)

if st.button("Run Step"):
    if auto:
        state = env.get_state()
        action_id = agent.select_action(state)
        action_name = agent.action_name(action_id)
    else:
        action_name = manual
        action_id = agent.action_names.index(action_name)

    result = env.step(action_id, action_name)
    agent.update(action_id, result.reward)
    st.session_state.logs.append(result)
    st.success(
        f"action={result.action_name}, old={result.old_score:.4f}, "
        f"new={result.new_score:.4f}, reward={result.reward:+.4f}"
    )

st.subheader("RL Timeline")
if st.session_state.logs:
    df = pd.DataFrame([
        {
            "step": i,
            "action": log.action_name,
            "old_score": log.old_score,
            "new_score": log.new_score,
            "reward": log.reward,
            "processed": log.n_processed,
        }
        for i, log in enumerate(st.session_state.logs)
    ])
    st.dataframe(df)
    st.line_chart(df.set_index("step")[["old_score", "new_score", "reward"]])
else:
    st.info("Click Run Step to start.")
