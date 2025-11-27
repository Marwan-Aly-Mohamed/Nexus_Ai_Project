# Nexus AI — Multi-Modal Hybrid Financial Decision Engine

Nexus AI is a next-generation Bicameral Financial Decision System that solves a core limitation in modern trading AI: modality blindness. Traditional models rely solely on price (Quant) or news (NLP). Nexus AI fuses both worlds using a dual-brain neural architecture to generate high-confidence, multi-modal trading signals.

# Core Idea
Nexus AI acts like a hedge-fund analyst, combining:
* DLinear Time-Series Forecasting → Predicts structural price trajectories
* FinBERT Financial NLP → Interprets news sentiment with domain accuracy
Both “brains” operate independently and unify only at the final decision gate, ensuring the model never trades based on hallucinated sentiment or misleading price noise.

# Bicameral Architecture
### Left Brain — Quant Model (DLinear)
* Processes 60-day OHLCV tensors
* Decomposes Trend & Seasonality
* Multi-step forecasting (3-day vector)
* Resistant to the “LSTM Persistence Trap”

### Right Brain — NLP Model (FinBERT)
* Analyzes ticker-specific RSS headlines
* Weighted loss to handle Neutral-heavy datasets
* Time-decay scoring + relevance grading

# Training Methodology
* Log returns + Robust scaling
* Covariates: Volume, Volatility, RSI
* Mixed-domain training (Tech, Crypto, Energy, Indices)
* L1 loss + early stopping
* Class-weighted sentiment loss for rare events

# Nexus Advantage
* Hybrid Quant + NLP verification
* Structural forecasting via DLinear
* Rare-event sensitivity through weighted FinBERT
* Infinitely scalable across 10,000+ assets

# Contact
Maintainer: Mohamed Abdelmonem Zidan, Marwan Aly Mohamed, Sohaila Yasser Khalil
