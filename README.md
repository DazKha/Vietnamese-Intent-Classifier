# Intent Classifier Web Interface

Vietnamese Intent Classification powered by PhoBERT-based model.

## 🚀 Features

- **Two-column layout**: Chat interface on the left, detailed analysis on the right
- **Real-time analysis**: Instant intent classification with probability scores
- **Detailed tokenization view**: 
  - Underthesea word segmentation
  - PhoBERT BPE tokens with syntax highlighting
  - h_cls vector visualization

## 🔧 Configuration

The model is hosted at HuggingFace Space and the API URL is public, feel free to use it.

For local development, edit `config.js` directly.

## 📝 API Requirements

Your backend API should support:

### Health Check
```
GET /health
Response: { "status": "ok", "device": "cpu" }
```

### Parse Intent
```
POST /parse
Body: { "utterance": "Hôm nay tôi ăn bún bò hết 50k" }
Response: {
  "utterance": "...",
  "intents": ["add_expense"],
  "probabilities": { "add_expense": 0.99, ... },
  "debug_info": {
    "tokenized": ["Hôm_nay", "tôi", ...],
    "bpe": ["<s>", "Hôm", "nay", ...],
    "h_cls_sample": [0.1234, -0.5678, ...]
  }
}
```

## 📄 License

MIT License - feel free to use this project for your own purposes.
