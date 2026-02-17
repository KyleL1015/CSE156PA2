# add all  your Encoder and Decoder code here
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
	def __init__(self, n_embd, n_head, dropout=0.0):
		super().__init__()
		assert n_embd % n_head == 0
		self.n_head = n_head
		self.d_k = n_embd // n_head
		self.qkv_proj = nn.Linear(n_embd, 3 * n_embd) #[Q, K, V] = Wx
		self.out_proj = nn.Linear(n_embd, n_embd)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		# x: [batch, seq_len, n_embd]
		B, T, C = x.size()
		qkv = self.qkv_proj(x)  # [B, T, 3*C]
		q, k, v = qkv.chunk(3, dim=-1) #splits into 3 different tensors on last dim [B, T, C]

		# reshape for heads: [B, n_head, T, d_k]
		q = q.view(B, T, self.n_head, self.d_k).transpose(1, 2)
		k = k.view(B, T, self.n_head, self.d_k).transpose(1, 2)
		v = v.view(B, T, self.n_head, self.d_k).transpose(1, 2)

		# scaled dot-product
		scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B, n_head, T, T]
		attn = F.softmax(scores, dim=-1)
		attn = self.dropout(attn)

		out = torch.matmul(attn, v)  # [B, n_head, T, d_k]
		out = out.transpose(1, 2).contiguous().view(B, T, C)
		out = self.out_proj(out)

		return out, attn


class TransformerBlock(nn.Module):
	def __init__(self, n_embd, n_head, dropout=0.0):
		super().__init__()
		self.attn = MultiHeadSelfAttention(n_embd, n_head, dropout)
		self.ln1 = nn.LayerNorm(n_embd)
		self.ff = nn.Sequential(
			nn.Linear(n_embd, 4 * n_embd),
			nn.GELU(),
			nn.Linear(4 * n_embd, n_embd),
			nn.Dropout(dropout)
		)
		self.ln2 = nn.LayerNorm(n_embd)

	def forward(self, x):
		attn_out, attn_map = self.attn(self.ln1(x))
		x = x + attn_out
		ff_out = self.ff(self.ln2(x))
		x = x + ff_out
		return x, attn_map


class TransformerEncoder(nn.Module):
	def __init__(self, vocab_size, n_embd=64, n_head=2, n_layer=4, max_len=512, dropout=0.0):
		super().__init__()
		self.token_emb = nn.Embedding(vocab_size, n_embd)
		self.pos_emb = nn.Embedding(max_len, n_embd)
		self.layers = nn.ModuleList([TransformerBlock(n_embd, n_head, dropout) for _ in range(n_layer)])
		self.ln_f = nn.LayerNorm(n_embd)
		self.n_layer = n_layer

	def forward(self, idx):
		# idx: [B, T]
		device = idx.device
		B, T = idx.size()
		pos = torch.arange(T, device=device).unsqueeze(0)
		x = self.token_emb(idx) + self.pos_emb(pos)

		attn_maps = []
		for layer in self.layers:
			x, attn = layer(x)
			attn_maps.append(attn)

		x = self.ln_f(x)
		# return sequence embeddings and list of attention maps (one per layer)
		return x, attn_maps


class FeedForwardClassifier(nn.Module):
	def __init__(self, n_input=64, n_hidden=100, n_output=3):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(n_input, n_hidden),
			nn.ReLU(),
			nn.Linear(n_hidden, n_output)
		)

	def forward(self, x):
		# x: [B, T, n_input] -> take mean across sequence dimension
		x_mean = x.mean(dim=1)
		return self.net(x_mean)


class ClassifierModel(nn.Module):
	"""Wrapper that contains encoder + feedforward classifier. Forward accepts token ids."""
	def __init__(self, vocab_size, n_embd=64, n_head=2, n_layer=4, n_hidden=100, n_output=3, max_len=512, dropout=0.0):
		super().__init__()
		self.encoder = TransformerEncoder(vocab_size, n_embd, n_head, n_layer, max_len, dropout)
		self.classifier = FeedForwardClassifier(n_embd, n_hidden, n_output)

	def forward(self, idx):
		seq_emb, attn_maps = self.encoder(idx)
		logits = self.classifier(seq_emb)
		return logits

class MaskedMultiHeadSelfAttention(nn.Module):
	"""Multi-head self-attention with causal masking (decoder-style)."""
	def __init__(self, n_embd, n_head, dropout=0.0):
		super().__init__()
		assert n_embd % n_head == 0
		self.n_head = n_head
		self.d_k = n_embd // n_head
		self.qkv_proj = nn.Linear(n_embd, 3 * n_embd)
		self.out_proj = nn.Linear(n_embd, n_embd)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		# x: [batch, seq_len, n_embd]
		B, T, C = x.size()
		qkv = self.qkv_proj(x)  # [B, T, 3*C]
		q, k, v = qkv.chunk(3, dim=-1)

		# reshape for heads: [B, n_head, T, d_k]
		q = q.view(B, T, self.n_head, self.d_k).transpose(1, 2)
		k = k.view(B, T, self.n_head, self.d_k).transpose(1, 2)
		v = v.view(B, T, self.n_head, self.d_k).transpose(1, 2)

		# scaled dot-product
		scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B, n_head, T, T]
		
		# Apply causal mask: prevent attention to future tokens
		mask = torch.tril(torch.ones(T, T, device=x.device)) == 1
		scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
		
		attn = F.softmax(scores, dim=-1)
		attn = self.dropout(attn)

		out = torch.matmul(attn, v)  # [B, n_head, T, d_k]
		out = out.transpose(1, 2).contiguous().view(B, T, C)
		out = self.out_proj(out)

		return out, attn


class DecoderBlock(nn.Module):
	"""Transformer decoder block with masked self-attention."""
	def __init__(self, n_embd, n_head, dropout=0.0):
		super().__init__()
		self.attn = MaskedMultiHeadSelfAttention(n_embd, n_head, dropout)
		self.ln1 = nn.LayerNorm(n_embd)
		self.ff = nn.Sequential(
			nn.Linear(n_embd, 100),  # feedforward hidden dim = 100
			nn.ReLU(),
			nn.Linear(100, n_embd),
			nn.Dropout(dropout)
		)
		self.ln2 = nn.LayerNorm(n_embd)

	def forward(self, x):
		attn_out, attn_map = self.attn(self.ln1(x))
		x = x + attn_out
		ff_out = self.ff(self.ln2(x))
		x = x + ff_out
		return x, attn_map


class TransformerDecoder(nn.Module):
	"""Transformer decoder for language modeling."""
	def __init__(self, vocab_size, n_embd=64, n_head=2, n_layer=4, max_len=512, dropout=0.0):
		super().__init__()
		self.token_emb = nn.Embedding(vocab_size, n_embd)
		self.pos_emb = nn.Embedding(max_len, n_embd)
		self.layers = nn.ModuleList([DecoderBlock(n_embd, n_head, dropout) for _ in range(n_layer)])
		self.ln_f = nn.LayerNorm(n_embd)
		self.lm_head = nn.Linear(n_embd, vocab_size)  # output probabilities over vocab
		self.n_layer = n_layer

	def forward(self, idx):
		# idx: [B, T]
		device = idx.device
		B, T = idx.size()
		pos = torch.arange(T, device=device).unsqueeze(0)
		x = self.token_emb(idx) + self.pos_emb(pos)

		attn_maps = []
		for layer in self.layers:
			x, attn = layer(x)
			attn_maps.append(attn)

		x = self.ln_f(x)
		logits = self.lm_head(x)  # [B, T, vocab_size]
		return logits, attn_maps