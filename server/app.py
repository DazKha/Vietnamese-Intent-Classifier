from fastapi import FastAPI
from pydantic import BaseModel
import torch
import os
from transformers import AutoTokenizer
from model_def import VSLIM # Import từ file model_def.py
from underthesea import word_tokenize
from label_loader import get_label_mappings
from predict import VSLIMPredictor
from types import SimpleNamespace
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

args = SimpleNamespace(
    task="vped",
    data_dir="./data",
    intent_label_file="intent_label.txt",
    slot_label_file="slot_label.txt",
    model_type="phobert",
    model_name_or_path="vinai/phobert-base-v2",
    dropout_rate=0.1,
    use_crf=False,
    num_mask=4,
    cls_token_cat=1,
    intent_attn=1,
    tag_intent=1,
    ignore_index=-100,
    intent_loss_coef=1.0,
    slot_loss_coef=2.0,
    token_intent_loss_coef=2.0,
    tag_intent_coef=1.0,
    max_seq_len=128,
    no_cuda=False
)

# Khởi tạo model khi start app
device = "cpu" # HF Space free dùng CPU
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
INTENT_LABELS, SLOT_LABELS, mappings = get_label_mappings(args)

model = VSLIM(
    model_name=args.model_name_or_path,
    num_slots=len(SLOT_LABELS),
    num_intents=len(INTENT_LABELS),
    num_token_intents=len(mappings["TOKEN_INTENT_LABELS"]),
    num_tag_intents=len(mappings["INTENT_LABELS_WITH_PAD"]),
    dropout=args.dropout_rate,
    use_crf=args.use_crf,
    num_mask=args.num_mask,
    cls_token_cat=bool(args.cls_token_cat),
    intent_attn=bool(args.intent_attn),
    args=args
)

# Load trọng số
state = torch.load("model.pt", map_location="cpu")
model.load_state_dict(state, strict=True)
model.eval()

predictor = VSLIMPredictor(model, tokenizer, mappings, device, args)

class ParseIn(BaseModel):
    utterance: str

# ====== BIO -> entities: key = đúng slot label (bỏ tiền tố B-/I-) ======
def extract_entities_from_bio(tokens, slot_tags, token_intents):
    entities = []
    n = len(tokens)
    i = 0
    while i < n:
        tag = slot_tags[i]
        if tag.startswith("B-"):
            typ = tag[2:]  # ví dụ: "fromloc.city_name"
            j = i + 1
            while j < n and slot_tags[j] == f"I-{typ}":
                j += 1
            span_tokens = tokens[i:j]
            text = " ".join([t.replace("_", " ") for t in span_tokens])
            intents_in_span = [ti for ti in token_intents[i:j] if ti != "O"]
            intent = None
            if intents_in_span:
                counts = {}
                for it in intents_in_span:
                    counts[it] = counts.get(it, 0) + 1
                intent = max(counts.items(), key=lambda x: x[1])[0]
            entities.append({"key": typ, "text": text, "intent": intent if intent else None})
            i = j
        else:
            i += 1
    return entities

def build_response_schema(utterance: str, result: dict):
    # Trả về nguyên dict result từ predictor
    return {
        "utterance": utterance,
        "debug_info": {
            "tokenized": result["tokenized_text"],
            "bpe": result["bpe_tokens"],
            "h_cls_sample": result["h_cls_vector"]
        },
        "probabilities": result["intent_probabilities"],
        "intents": result["final_intents"]
    }

@app.get("/health")
def health():
    return {"status": "ok", "device": device}

@app.get("/")
def home():
    return {"message": "VSLIM Multi-intent API is running on HF Spaces"}

@app.post("/parse")
async def parse(req: ParseIn):
    text = req.utterance.strip()
    # Tokenize bằng underthesea
    s = word_tokenize(text, format="text")
    tokens = s.split()
    
    result = predictor.predict_single(tokens, threshold=0.9)
    return build_response_schema(text, result)