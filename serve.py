"""HTTP API for your apps + a minimal browser chat page.

    python serve.py --preset ten_m
    # API:
    curl -s localhost:8000/generate -H 'content-type: application/json' \
         -d '{"prompt":"The sky is","max_new_tokens":60}'
    # Browser chat: open http://127.0.0.1:8000
"""
import argparse
import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

from common import pick_device, load_model, load_meta, get_codec

app = FastAPI(title="SLM-10M")
STATE = {}


class GenReq(BaseModel):
    prompt: str = "\n"
    max_new_tokens: int = 100
    temperature: float = 0.8
    top_k: int = 200


@app.post("/generate")
def generate(req: GenReq):
    model, device = STATE["model"], STATE["device"]
    encode, decode = STATE["enc"], STATE["dec"]
    ids = encode(req.prompt) or [0]
    x = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]
    y = model.generate(x, req.max_new_tokens, req.temperature, req.top_k)
    return {
        "prompt": req.prompt,
        "completion": decode(y[0].tolist()[len(ids):]),
        "text": decode(y[0].tolist()),
    }


PAGE = """<!doctype html><meta charset=utf-8><title>SLM-10M chat</title>
<style>body{font-family:-apple-system,Arial,sans-serif;max-width:680px;margin:40px auto;padding:0 16px}
#log{white-space:pre-wrap;border:1px solid #ddd;border-radius:8px;padding:12px;min-height:240px}
input{width:78%;padding:8px}button{padding:8px 14px}</style>
<h2>SLM-10M chat</h2><div id=log></div><p>
<input id=p placeholder="type a prompt..." autofocus><button onclick=go()>send</button></p>
<script>
const log=document.getElementById('log'),p=document.getElementById('p');
async function go(){const t=p.value;if(!t)return;log.textContent+="\\nyou> "+t+"\\n";p.value='';
const r=await fetch('/generate',{method:'POST',headers:{'content-type':'application/json'},
body:JSON.stringify({prompt:t,max_new_tokens:120})});const j=await r.json();
log.textContent+="slm> "+j.completion.trim()+"\\n";log.scrollTop=log.scrollHeight;}
p.addEventListener('keydown',e=>{if(e.key==='Enter')go();});
</script>"""


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    device = pick_device()
    model, ck = load_model(args.preset, device)
    encode, decode = get_codec(load_meta(ck["dataset"]))
    STATE.update(model=model, device=device, enc=encode, dec=decode)
    print(f"[SLM serve · {args.preset} · {device}] http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
