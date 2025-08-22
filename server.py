# server.py
# FastAPI backend for PyTorch model visualization and live streaming.
# Requires: fastapi uvicorn "pydantic<2" starlette torch==2.*
# Note: For local dev only (not hardened for production).

import io
import json
import types
import traceback
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import uvicorn
from fastapi import Body, FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
MAX_NODES_PER_LAYER = 356
MAX_EDGES_PER_PAIR = 6000

# -------------------------------------------------------------------
# App setup
# -------------------------------------------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATE: Dict[str, Any] = {
    "model_code": None,
    "model": None,
    "state_dict": None,
    "device": "cpu",
    "example_input": None,
    "graph_spec": None,
}

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def sample_indices(n: int, cap: int) -> List[int]:
    """Deterministically sample up to `cap` indices from range(n)."""
    if n <= cap:
        return list(range(n))
    step = max(1, n // cap)
    return [i for i in range(0, n, step)][:cap]


def load_user_model_from_code(code: str) -> Tuple[nn.Module, torch.Tensor]:
    """
    Execute user code, expecting:
      - build_model() -> nn.Module
      - (optional) build_example_input(model) -> torch.Tensor
    """
    module = types.ModuleType("user_model")
    # Put torch + nn directly into the module’s dict
    module.__dict__.update({"torch": torch, "nn": nn, "__builtins__": __builtins__})

    try:
        exec(code, module.__dict__)
    except Exception as e:
        raise RuntimeError(f"Error executing model code: {e}\n{traceback.format_exc()}")

    if not hasattr(module, "build_model"):
        raise RuntimeError("Your code must define build_model() -> nn.Module")

    model: nn.Module = module.build_model()
    model.eval()

    example_input = (
        module.build_example_input(model)
        if hasattr(module, "build_example_input")
        else None
    )
    return model, example_input



def guess_input_from_model(model: nn.Module) -> torch.Tensor:
    """
    Heuristic: return plausible input for model.
    - Linear → [1, in_features]
    - Conv2d → [1, in_channels, 64, 64]
    - Has attr input_size → torch.randn(1, *input_size)
    - Fallback → [1, 128]
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            return torch.randn(1, m.in_features)
        if isinstance(m, nn.Conv2d):
            return torch.randn(1, m.in_channels, 64, 64)

    if hasattr(model, "input_size"):
        sz = getattr(model, "input_size")
        return torch.randn(1, *sz) if isinstance(sz, (tuple, list)) else torch.randn(1, sz)

    return torch.randn(1, 128)


def build_graph_spec(
    model: nn.Module, sample_input: torch.Tensor
) -> Dict[str, Any]:
    """
    Build a visualization-ready graph (nodes & edges).
    This version records actual activation sizes during a dry forward pass,
    then uses those sizes (downsampled by MAX_NODES_PER_LAYER) as per-layer node counts.
    """
    layers: List[Dict[str, Any]] = []
    layer_map: Dict[Any, str] = {}
    forward_order: List[nn.Module] = []
    activation_sizes: Dict[nn.Module, int] = {}  # per-module flattened size (per item)

    # Hook to capture module order and activation shapes
    def order_and_shape_hook(mod, inp, out):
        forward_order.append(mod)
        try:
            y = out[0] if isinstance(out, (list, tuple)) else out
            if isinstance(y, torch.Tensor):
                # flattened size per item
                s = y.view(y.size(0), -1).shape[1]
                activation_sizes[mod] = int(s)
        except Exception:
            # ignore modules that don't produce tensors
            pass

    # Register hooks for modules (skip top-level model)
    handles = []
    for m in model.modules():
        if m is model:
            continue
        # register broadly; we'll filter later
        try:
            handles.append(m.register_forward_hook(order_and_shape_hook))
        except Exception:
            pass

    # Dry-run to collect forward order and activation sizes
    with torch.no_grad():
        model(sample_input)

    # remove hooks
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass

    # Deduplicate forward order while preserving order
    seen = set()
    ordered_modules = []
    for m in forward_order:
        if m not in seen:
            seen.add(m)
            ordered_modules.append(m)

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    layer_idx = 0

    def add_layer(label: str, n_type: str, size: int) -> str:
        nonlocal layer_idx
        lid = f"L{layer_idx}"
        layers.append({"id": lid, "label": label, "type": n_type, "size": size})
        layer_idx += 1
        return lid

    # Input layer: infer from sample_input flattened size per item
    input_size = int(sample_input.numel() / sample_input.shape[0])
    input_count = min(input_size, MAX_NODES_PER_LAYER)
    input_layer_id = add_layer("Input", "input", input_count)
    layer_map["__input__"] = input_layer_id

    # Walk the observed modules in forward order and create layers using recorded sizes
    for m in ordered_modules:
        # determine nominal size: prefer recorded activation size, then layer attributes
        recorded = activation_sizes.get(m, None)
        if recorded is not None:
            node_count = min(recorded, MAX_NODES_PER_LAYER)
        else:
            # fallback heuristics
            if isinstance(m, nn.Linear):
                node_count = min(m.out_features, MAX_NODES_PER_LAYER)
            elif isinstance(m, nn.Conv2d):
                node_count = min(m.out_channels, MAX_NODES_PER_LAYER)
            else:
                node_count = 1

        # label by class name (better readability)
        label = m.__class__.__name__
        lid = add_layer(label, "op" if not isinstance(m, (nn.Linear, nn.Conv2d)) else ("linear" if isinstance(m, nn.Linear) else "conv2d"), node_count)
        layer_map[m] = lid

    # Build edges by walking modules again, but map edges between consecutive layers
    # We'll connect each layer to the previous layer added
    # Use module-specific weight summaries for Linear/Conv2d when available
    prev_layers = list(layer_map.values())
    # prev_layers[0] is input; subsequent correspond to ordered_modules in same order
    # Create a mapping from module -> its index in ordered_modules for alignment
    mod_to_layerid = {m: layer_map[m] for m in ordered_modules}

    # iterate through ordered_modules and create edges from previous visual layer to this
    # maintain a pointer for previous layer id (start from input)
    prev_id = input_layer_id
    for m in ordered_modules:
        lid = layer_map[m]
        # build edges based on weight summaries where possible
        try:
            if isinstance(m, nn.Linear) and hasattr(m, "weight"):
                W = m.weight.detach().cpu()  # shape [out, in]
                in_cap = next(l for l in layers if l["id"] == prev_id)["size"]
                out_cap = next(l for l in layers if l["id"] == lid)["size"]
                in_idx = sample_indices(W.shape[1], in_cap)
                out_idx = sample_indices(W.shape[0], out_cap)
                cnt = 0
                for oi in out_idx:
                    row = W[oi]
                    for ii in in_idx:
                        if cnt >= MAX_EDGES_PER_PAIR:
                            break
                        edges.append({
                            "from_layer": prev_id,
                            "to_layer": lid,
                            "i": ii,
                            "j": oi,
                            "w": float(row[ii].item())
                        })
                        cnt += 1

            elif isinstance(m, nn.Conv2d) and hasattr(m, "weight"):
                W = m.weight.detach().cpu()  # [out_ch, in_ch, kh, kw]
                in_cap = next(l for l in layers if l["id"] == prev_id)["size"]
                out_cap = next(l for l in layers if l["id"] == lid)["size"]
                in_idx = sample_indices(W.shape[1], in_cap)
                out_idx = sample_indices(W.shape[0], out_cap)
                cnt = 0
                for oc in out_idx:
                    row = W[oc].mean(dim=(1, 2))  # per in-channel mean
                    for ic in in_idx:
                        if cnt >= MAX_EDGES_PER_PAIR:
                            break
                        edges.append({
                            "from_layer": prev_id,
                            "to_layer": lid,
                            "i": ic,
                            "j": oc,
                            "w": float(row[ic].item())
                        })
                        cnt += 1

            else:
                # generic connection — no weights; create a single representative edge
                edges.append({
                    "from_layer": prev_id,
                    "to_layer": lid,
                    "i": 0, "j": 0, "w": 0.0
                })
        except Exception:
            # On any failure, still connect with a placeholder edge
            edges.append({
                "from_layer": prev_id,
                "to_layer": lid,
                "i": 0, "j": 0, "w": 0.0
            })

        prev_id = lid  # next module connects from this layer

    # Convert internal layers -> nodes for visualization (keeping counts)
    nodes = []
    for L in layers:
        nodes.append({
            "layer_id": L["id"],
            "label": L["label"],
            "type": L["type"],
            "count": L["size"]
        })

    return {
        "nodes": nodes,
        "edges": edges,
        "input_layer": input_layer_id,
        "output_layer": layers[-1]["id"] if layers else input_layer_id,
    }

# -------------------------------------------------------------------
# API routes
# -------------------------------------------------------------------
@app.post("/api/set_code")
async def set_code(payload: Dict[str, str] = Body(...)):
    code = payload.get("code", "").strip()
    if not code:
        return JSONResponse({"ok": False, "error": "Empty code"}, status_code=400)

    try:
        model, example_input = load_user_model_from_code(code)
    except Exception as e:
        print("=== Error while loading user code ===")
        traceback.print_exc()   # <-- full error printed to terminal
        print("====================================")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

    STATE.update({
        "model_code": code,
        "model": model,
        "example_input": example_input
    })
    return {"ok": True, "info": "Model code accepted."}


@app.post("/api/upload_pth")
async def upload_pth(file: UploadFile = File(...)):
    if not file.filename.endswith(".pth"):
        return JSONResponse({"ok": False, "error": "Only .pth files supported."}, status_code=400)

    try:
        state = torch.load(io.BytesIO(await file.read()), map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict) and hasattr(state, "state_dict"):
            state = state.state_dict()
        if not isinstance(state, dict):
            raise RuntimeError("Invalid .pth file: no state_dict found")
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"Failed to load .pth: {e}"}, status_code=400)

    STATE["state_dict"] = state
    return {"ok": True, "info": f"Loaded {file.filename}."}


@app.post("/api/prepare")
async def prepare_graph():
    model, state_dict = STATE.get("model"), STATE.get("state_dict")
    if not model or not state_dict:
        return JSONResponse({"ok": False, "error": "Need model code + .pth file"}, status_code=400)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    example_input = STATE["example_input"] if STATE["example_input"] is not None else guess_input_from_model(model)
    STATE["example_input"] = example_input
    graph = build_graph_spec(model, example_input)
    STATE["graph_spec"] = graph

    return {"ok": True, "graph": graph, "missing_keys": list(missing), "unexpected_keys": list(unexpected)}


@app.websocket("/ws/run")
async def ws_run(ws: WebSocket):
    await ws.accept()
    try:
        model, graph, x = STATE["model"], STATE["graph_spec"], STATE["example_input"]
        if not (model and graph and x is not None):
            await ws.send_text(json.dumps({"type": "error", "error": "Model not prepared"}))
            return

        await ws.send_text(json.dumps({"type": "graph", "graph": graph}))

        # Collect activations
        layer_outputs: Dict[str, List[float]] = {}
        hooks = []

        def mk_hook(layer_id: str):
            def hook(_, __, out):
                with torch.no_grad():
                    y = out[0] if isinstance(out, (list, tuple)) else out
                    if not isinstance(y, torch.Tensor):
                        return
                    vec = y.view(y.size(0), -1).cpu()
                    count = next(L["count"] for L in graph["nodes"] if L["layer_id"] == layer_id)
                    if vec.size(1) > count:
                        step = max(1, vec.size(1) // count)
                        vec = vec[:, ::step][:, :count]
                    layer_outputs[layer_id] = vec.abs().mean(dim=0).tolist()
            return hook

        # Run once to determine order
        run_order = []
        tmp_hooks = [m.register_forward_hook(lambda m, *_: run_order.append(m)) for m in model.modules() if m is not model]
        with torch.no_grad():
            model(x)
        for h in tmp_hooks: h.remove()

        graph_layers = [n["layer_id"] for n in graph["nodes"]][1:]  # skip input
        for m, lid in zip(run_order, graph_layers):
            hooks.append(m.register_forward_hook(mk_hook(lid)))

        # Input activations
        with torch.no_grad():
            v = x.view(x.size(0), -1)
            count = next(L["count"] for L in graph["nodes"] if L["layer_id"] == graph["input_layer"])
            if v.size(1) > count:
                step = max(1, v.size(1) // count)
                v = v[:, ::step][:, :count]
            layer_outputs[graph["input_layer"]] = v.abs().mean(dim=0).tolist()

        await ws.send_text(json.dumps({"type": "tick", "activations": layer_outputs}))
        with torch.no_grad(): model(x)
        await ws.send_text(json.dumps({"type": "tick", "activations": layer_outputs}))
        await ws.send_text(json.dumps({"type": "done"}))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await ws.send_text(json.dumps({"type": "error", "error": str(e)}))
    finally:
        for h in hooks: h.remove()
        await ws.close()


@app.get("/")
def root():
    return {"ok": True, "msg": "Backend running. Open index.html."}


if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
