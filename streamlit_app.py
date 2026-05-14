import io
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import streamlit as st

# ---- 日本語フォント ----
try:
    import japanize_matplotlib  # noqa: F401
    _jp = True
except ImportError:
    import matplotlib.font_manager as _fm
    _jp = False
    for _f in ["Noto Sans JP", "Yu Gothic", "Meiryo", "MS Gothic"]:
        if _f in {f.name for f in _fm.fontManager.ttflist}:
            matplotlib.rcParams["font.family"] = _f
            _jp = True
            break
matplotlib.rcParams["axes.unicode_minus"] = False

def L(ja, en): return ja if _jp else en

# ==============================================================
# 色
# ==============================================================
PLAY_CMAP = LinearSegmentedColormap.from_list("pg", ["#FF6D00", "#FFFFFF", "#2196F3"])
DARK_BG  = "#1a1a2e"
PANEL_BG = "#16213e"
TEXT_COL = "#e0e0e0"
MUTED    = "#888888"
COL_POS  = "#2196F3"
COL_NEG  = "#FF6D00"
COL_0    = "#FF6D00"
COL_1    = "#2196F3"

# ==============================================================
# データセット
# ==============================================================
def make_circle(n=300, noise=0.1, seed=42):
    rng = np.random.default_rng(seed)
    a = rng.uniform(0, 2 * np.pi, n); r = rng.uniform(0, 1.2, n)
    X = np.c_[r * np.cos(a), r * np.sin(a)]
    return X + rng.normal(0, noise, X.shape), (r < 0.6).astype(int)

def make_xor(n=300, noise=0.1, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.2, 1.2, (n, 2))
    return X + rng.normal(0, noise, X.shape), ((X[:,0]>0)==(X[:,1]>0)).astype(int)

def make_spiral(n=300, noise=0.1, seed=42):
    rng = np.random.default_rng(seed); n2 = n // 2
    r = np.linspace(0.1, 1.1, n2)
    t0 = np.linspace(0, 3.5*np.pi, n2) + rng.normal(0, noise*3, n2); t1 = t0 + np.pi
    return (np.vstack([np.c_[r*np.cos(t0), r*np.sin(t0)],
                       np.c_[r*np.cos(t1), r*np.sin(t1)]]),
            np.array([0]*n2+[1]*n2))

DATASETS = {
    L("円", "circle"):   make_circle,
    "XOR":               make_xor,
    L("螺旋", "spiral"): make_spiral,
}
LR_OPT = {"0.001":0.001,"0.003":0.003,"0.01":0.01,"0.03":0.03,"0.1":0.1,"0.3":0.3}
_NONE  = L("なし", "none")

# ==============================================================
# MLP
# ==============================================================
class MLP:
    def __init__(self, hidden_sizes, activation="tanh", lr=0.03):
        sizes = [2] + list(hidden_sizes) + [1]
        rng   = np.random.default_rng(42)
        self.lr=lr; self.act_name=activation; self.sizes=sizes
        self.W = [rng.normal(0, 1/np.sqrt(sizes[i]), (sizes[i], sizes[i+1]))
                  for i in range(len(sizes)-1)]
        self.b = [np.zeros((1, sizes[i+1])) for i in range(len(sizes)-1)]
        self.losses=[]; self.step_count=0

    def _act(self, z):
        if self.act_name=="tanh":    return np.tanh(z)
        if self.act_name=="relu":    return np.maximum(0, z)
        if self.act_name=="sigmoid": return 1/(1+np.exp(-np.clip(z,-500,500)))
        return z
    def _act_grad(self, a):
        if self.act_name=="tanh":    return 1-a**2
        if self.act_name=="relu":    return (a>0).astype(float)
        if self.act_name=="sigmoid": return a*(1-a)
        return np.ones_like(a)
    @staticmethod
    def _sig(z): return 1/(1+np.exp(-np.clip(z,-500,500)))

    def forward(self, X, store=True):
        h, acts = X, [X]
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            z = h@w+b; h = self._act(z) if i<len(self.W)-1 else self._sig(z)
            acts.append(h)
        if store: self._acts = acts
        return h

    def get_layer_acts(self, X, layer):
        h = X
        for i in range(layer):
            z = h@self.W[i]+self.b[i]
            h = self._act(z) if i<len(self.W)-1 else self._sig(z)
        return h

    def backward(self, y):
        n=len(y); d=(self._acts[-1]-y.reshape(-1,1))/n
        for i in reversed(range(len(self.W))):
            dw=self._acts[i].T@d; db=d.sum(axis=0, keepdims=True)
            if i>0: d=(d@self.W[i].T)*self._act_grad(self._acts[i])
            self.W[i]-=self.lr*np.clip(dw,-5,5)
            self.b[i]-=self.lr*np.clip(db,-5,5)

    def train_steps(self, X, y, n_steps, batch_size=10):
        rng = np.random.default_rng()
        for _ in range(n_steps):
            idx = rng.integers(0, len(X), batch_size); Xb, yb = X[idx], y[idx]
            p = self.forward(Xb)[:,0]
            self.losses.append(float(-np.mean(yb*np.log(p+1e-9)+(1-yb)*np.log(1-p+1e-9))))
            self.backward(yb); self.step_count+=1

    def accuracy(self, X, y):
        return ((self.forward(X, store=False)[:,0]>0.5)==y).mean()*100

# ==============================================================
# 描画
# ==============================================================
LIM = 1.3

def _node_ypos(n):
    if n==1: return [0.5]
    return list(np.linspace(max(0.10, 0.45-n*0.08), 1-max(0.10, 0.45-n*0.08), n))

def _grid_acts(net, res=20):
    xx, yy = np.meshgrid(np.linspace(-LIM,LIM,res), np.linspace(-LIM,LIM,res))
    grid = np.c_[xx.ravel(), yy.ravel()]
    return [net.get_layer_acts(grid, l) for l in range(len(net.sizes))], res

def make_figure(net, X, y):
    n_layers=len(net.sizes); max_nodes=max(net.sizes)
    g_acts, RES = _grid_acts(net)

    fig = plt.figure(figsize=(18, 8), dpi=88, facecolor=DARK_BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[0.7, 1.8, 1.0],
                            left=0.02, right=0.99, top=0.93, bottom=0.07,
                            hspace=0.40, wspace=0.10)
    ax_data = fig.add_subplot(gs[:, 0])
    ax_net  = fig.add_subplot(gs[:, 1])
    ax_out  = fig.add_subplot(gs[0, 2])
    ax_loss = fig.add_subplot(gs[1, 2])

    tkw = dict(color=TEXT_COL, fontsize=9, fontweight="bold", pad=5)
    for ax in [ax_data, ax_out, ax_loss]:
        ax.set_facecolor(PANEL_BG)
        for sp in ax.spines.values(): sp.set_color("#334466")
        ax.tick_params(colors=MUTED, labelsize=7)
    ax_net.set_facecolor(PANEL_BG); ax_net.axis("off")
    ax_net.set_xlim(0, 1); ax_net.set_ylim(0, 1)

    # ---- 分類前のデータ ----
    for c, col in enumerate([COL_0, COL_1]):
        m = y==c
        ax_data.scatter(X[m,0], X[m,1], c=col, s=18,
                        edgecolors="white", linewidths=0.4, alpha=0.85, zorder=3)
    ax_data.set_xlim(-LIM, LIM); ax_data.set_ylim(-LIM, LIM); ax_data.set_aspect("equal")
    ax_data.set_title(L("分類前のデータ", "Dataset"), **tkw)
    ax_data.axhline(0, color="#334466", lw=0.5); ax_data.axvline(0, color="#334466", lw=0.5)

    # ---- ネットワーク図 ----
    x_pos = np.linspace(0.10, 0.90, n_layers)
    nw    = min(0.13, 0.70/max(max_nodes, 1))
    max_wv = max((abs(w).max() for w in net.W), default=1.0); max_wv=max(max_wv, 0.01)

    for l in range(n_layers-1):
        yp_l=_node_ypos(net.sizes[l]); yp_n=_node_ypos(net.sizes[l+1])
        for j, yj in enumerate(yp_l):
            for k, yk in enumerate(yp_n):
                wv=net.W[l][j,k]; nwabs=abs(wv)/max_wv
                ax_net.plot([x_pos[l], x_pos[l+1]], [yj, yk],
                            color=COL_POS if wv>=0 else COL_NEG,
                            lw=nwabs*5+0.2, alpha=nwabs*0.75+0.1,
                            zorder=1, solid_capstyle="round")

    for l in range(n_layers):
        yp = _node_ypos(net.sizes[l])
        lbl = (L("入力","INPUT") if l==0
               else L("出力","OUTPUT") if l==n_layers-1
               else L(f"隠れ層{l}", f"HIDDEN{l}"))
        ax_net.text(x_pos[l], 0.02, lbl, ha="center", va="bottom",
                    color=MUTED, fontsize=7.5)
        if l==0:
            for j, yj in enumerate(yp):
                ax_net.text(x_pos[l]-nw/2-0.015, yj, ["X₁","X₂"][j],
                            ha="right", va="center", color=TEXT_COL, fontsize=10)
        for j, yj in enumerate(yp):
            ax_nd = ax_net.inset_axes([x_pos[l]-nw/2, yj-nw/2, nw, nw])
            heat  = g_acts[l][:, j].reshape(RES, RES)
            if l<n_layers-1:
                vmax = max(abs(heat).max(), 0.01)
                ax_nd.imshow(heat, cmap=PLAY_CMAP, aspect="auto",
                             vmin=-vmax, vmax=vmax, origin="lower")
            else:
                ax_nd.imshow(heat, cmap=PLAY_CMAP, aspect="auto",
                             vmin=0, vmax=1, origin="lower")
                sc = (RES-1)/(2*LIM)
                for c, col in enumerate([COL_0, COL_1]):
                    m = y==c
                    ax_nd.scatter((X[m,0]+LIM)*sc, (X[m,1]+LIM)*sc,
                                  c=col, s=2, alpha=0.7, linewidths=0, zorder=5)
            for sp in ax_nd.spines.values():
                sp.set_edgecolor("white"); sp.set_linewidth(1.5)
            ax_nd.set_xticks([]); ax_nd.set_yticks([])

    ax_net.set_title(
        L(f"ネットワーク  {' → '.join(map(str, net.sizes))}",
          f"Network  {' → '.join(map(str, net.sizes))}"), **tkw)

    # ---- 決定境界 ----
    RES2=150
    xx2, yy2 = np.meshgrid(np.linspace(-LIM,LIM,RES2), np.linspace(-LIM,LIM,RES2))
    Z = net.forward(np.c_[xx2.ravel(),yy2.ravel()], store=False)[:,0].reshape(RES2,RES2)
    ax_out.imshow(Z, cmap=PLAY_CMAP, aspect="equal", vmin=0, vmax=1,
                  origin="lower", extent=[-LIM,LIM,-LIM,LIM])
    ax_out.contour(xx2, yy2, Z, levels=[0.5], colors="white", linewidths=1.5, alpha=0.8)
    acc = net.accuracy(X, y)
    for c, col in enumerate([COL_0, COL_1]):
        m = y==c
        ax_out.scatter(X[m,0], X[m,1], c=col, s=18,
                       edgecolors="white", linewidths=0.4, zorder=5, alpha=0.85)
    ax_out.set_xlim(-LIM, LIM); ax_out.set_ylim(-LIM, LIM)
    ax_out.set_title(L(f"決定境界  精度: {acc:.1f}%",
                       f"Decision Boundary  Acc: {acc:.1f}%"), **tkw)

    # ---- 損失曲線 ----
    ax_loss.set_title(L("損失曲線", "Loss"), **tkw)
    ax_loss.set_xlabel(L("ステップ", "Step"), color=MUTED, fontsize=8)
    ax_loss.set_ylabel("Loss", color=MUTED, fontsize=8)
    if net.losses:
        ls = np.array(net.losses)
        ax_loss.plot(ls, color="#ff9f43", alpha=0.2, lw=0.8)
        w = min(30, len(ls)); sm = np.convolve(ls, np.ones(w)/w, mode="valid")
        ax_loss.plot(np.arange(w-1, len(ls)), sm, color="#ff9f43", lw=2)
        ax_loss.set_xlim(0, max(len(ls), 100)); ax_loss.set_ylim(bottom=0)
    ax_loss.text(0.97, 0.95, f"Step:{net.step_count:,}\nAcc:{acc:.1f}%",
                 transform=ax_loss.transAxes, color="white", fontsize=8,
                 ha="right", va="top",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#21262d", alpha=0.8))

    fig.suptitle(L("ニューラルネットワーク プレイグラウンド",
                   "Neural Network Playground"),
                 color=TEXT_COL, fontsize=13, fontweight="bold", y=0.98)
    return fig

def fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=88, facecolor=fig.get_facecolor())
    plt.close(fig); buf.seek(0)
    return buf

# ==============================================================
# Streamlit UI
# ==============================================================
st.set_page_config(page_title="NN Playground", layout="wide")
st.title(L("ニューラルネットワーク プレイグラウンド", "Neural Network Playground"))

# ---- セッション状態の初期化 ----
if "net" not in st.session_state:
    st.session_state.X, st.session_state.y = make_circle()
    st.session_state.net        = MLP([4, 2], activation="tanh", lr=0.03)
    st.session_state.is_playing = False

# ---- サイドバー：設定 ----
with st.sidebar:
    st.header(L("設定", "Settings"))

    dataset_key = st.selectbox(
        L("データセット", "Dataset"), list(DATASETS.keys()))
    act_key = st.selectbox(
        L("活性化関数", "Activation"), ["tanh", "relu", "sigmoid"])
    lr_key = st.selectbox(
        L("学習率", "Learning Rate"), list(LR_OPT.keys()), index=3)  # 0.03

    st.subheader(L("隠れ層ノード数", "Hidden Layer Nodes"))
    node_opts = [_NONE] + [str(n) for n in range(1, 9)]
    l1 = st.selectbox(L("層1", "L1"), node_opts, index=5)   # 4
    l2 = st.selectbox(L("層2", "L2"), node_opts, index=3)   # 2
    l3 = st.selectbox(L("層3", "L3"), node_opts, index=0)   # なし
    l4 = st.selectbox(L("層4", "L4"), node_opts, index=0)   # なし

    speed = st.select_slider(
        L("速度（ステップ/フレーム）", "Speed (steps/frame)"),
        options=[1, 5, 20, 80, 300], value=20)
    step_size = st.select_slider(
        L("ステップ実行数", "Steps per click"),
        options=[1, 10, 100, 1000, 10000], value=100)

# ---- ボタン ----
col1, col2, col3 = st.columns(3)
play_label = L("⏸ 停止", "⏸ Pause") if st.session_state.is_playing \
             else L("▶ 再生", "▶ Play")
play  = col1.button(play_label, type="primary", use_container_width=True)
step  = col2.button(L("⏭ ステップ", "⏭ Step"), use_container_width=True)
reset = col3.button(L("↺ リセット", "↺ Reset"), use_container_width=True)

net = st.session_state.net
X   = st.session_state.X
y   = st.session_state.y

st.caption(L(f"累計ステップ: {net.step_count:,}　　精度: {net.accuracy(X, y):.1f}%",
             f"Total steps: {net.step_count:,}   Accuracy: {net.accuracy(X, y):.1f}%"))

# ---- ボタン処理 ----
if play:
    st.session_state.is_playing = not st.session_state.is_playing
    st.rerun()

if step and not st.session_state.is_playing:
    net.train_steps(X, y, step_size)
    st.rerun()

if reset:
    hidden = [int(v) for v in [l1, l2, l3, l4] if v != _NONE]
    st.session_state.X, st.session_state.y = DATASETS[dataset_key]()
    st.session_state.net        = MLP(hidden, activation=act_key, lr=LR_OPT[lr_key])
    st.session_state.is_playing = False
    st.rerun()

# ---- アニメーション中は1フレーム学習してすぐ再描画 ----
if st.session_state.is_playing:
    net.train_steps(X, y, speed)

# ---- 描画 ----
st.image(fig_to_png(make_figure(net, X, y)), use_container_width=True)

# ---- 再生中はループ ----
if st.session_state.is_playing:
    time.sleep(0.15)
    st.rerun()
