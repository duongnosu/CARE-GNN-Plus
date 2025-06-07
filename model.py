# =========================================================================
# CARE-GNN Model Architecture
# =========================================================================
# Purpose: Define the complete CARE-GNN model for fraud detection
# Paper: "Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters"
# Source: https://github.com/YingtongDou/CARE-GNN
#
# Model Architecture:
# 1. Multi-relation graph neural network with adaptive neighbor filtering
# 2. Label-aware similarity learning for camouflage-resistant detection
# 3. Combined loss function balancing GNN and similarity objectives
# 4. One-layer design for efficient training on large-scale fraud graphs
# =========================================================================

import torch
import torch.nn as nn
from torch.nn import init


"""
	CARE-GNN Models
	Paper: Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters
	Source: https://github.com/YingtongDou/CARE-GNN

	Key Innovation: Combines graph neural network with label-aware similarity learning
	to detect camouflaged fraudsters who try to mimic legitimate user behavior.
	
	Architecture Design:
	- Single layer for efficiency (fraud graphs are often large-scale)
	- Multi-relation aggregation (UPU, USU, UVU for Amazon)
	- Reinforcement learning for adaptive neighbor selection
	- Dual-objective training (GNN + similarity learning)
"""


class OneLayerCARE(nn.Module):
	"""
	Single-Layer CARE-GNN Model
	
	Purpose: Complete fraud detection model combining graph neural networks
	         with label-aware similarity learning
	
	Architecture Overview:
	┌─────────────────────────────────────────────────────────────┐
	│                    OneLayerCARE Model                       │
	├─────────────────────────────────────────────────────────────┤
	│  Input: Node features + Multi-relation adjacency lists     │
	│  ↓                                                          │
	│  InterAgg Layer (Multi-relation GNN)                       │
	│  ├── UPU Relation Processing                               │
	│  ├── USU Relation Processing                               │
	│  ├── UVU Relation Processing                               │
	│  └── Inter-relation Aggregation                            │
	│  ↓                                                          │
	│  Final Classification Layer                                 │
	│  ↓                                                          │
	│  Output: Fraud probability + Similarity scores             │
	└─────────────────────────────────────────────────────────────┘
	
	Key Components:
	1. Inter-relation aggregator (handles multiple graph relations)
	2. Label-aware similarity predictor (detects camouflaged fraud)
	3. Combined loss function (balances both objectives)
	"""

	def __init__(self, num_classes, inter1, lambda_1):
		"""
		Initialize the CARE-GNN model
		:param num_classes: number of classes (2 in our paper)
		:param inter1: the inter-relation aggregator that output the final embedding
		"""
		super(OneLayerCARE, self).__init__()
		self.inter1 = inter1
		self.xent = nn.CrossEntropyLoss()

		# the parameter to transform the final embedding
		self.weight = nn.Parameter(torch.FloatTensor(inter1.embed_dim, num_classes))
		init.xavier_uniform_(self.weight)
		self.lambda_1 = lambda_1

	def forward(self, nodes, labels, train_flag=True):
		"""
		Forward pass through the CARE-GNN model
		
		Args:
			nodes: Batch of node IDs to process
			labels: Node labels (used by RL module and similarity learning)
			train_flag: Whether in training mode (affects RL and dropout)
			
		Returns:
			scores: Final classification scores (logits) for fraud detection
			label_scores: Label-aware similarity scores for camouflage detection
			
		Process Flow:
		1. Multi-relation aggregation → Rich node embeddings
		2. Final classification layer → Fraud probability scores
		3. Label-aware scoring → Similarity-based fraud detection
		"""
		embeds1, label_scores = self.inter1(nodes, labels, train_flag)
		scores = torch.mm(embeds1, self.weight)
		return scores, label_scores

	def to_prob(self, nodes, labels, train_flag=True):
		"""
		Convert model outputs to probability distributions
		
		Purpose: Get interpretable probability scores for both GNN and similarity modules
		Essential for evaluation and comparison of model components
		
		Args:
			nodes: Batch of node IDs
			labels: Node labels  
			train_flag: Training mode flag
			
		Returns:
			gnn_prob: Softmax probabilities from GNN module
			label_prob: Softmax probabilities from similarity module
			
		Usage: Primarily used during evaluation to get probability scores
		       for AUC, precision-recall, and other probability-based metrics
		"""
		gnn_scores, label_scores = self.forward(nodes, labels, train_flag)
		gnn_prob = nn.functional.softmax(gnn_scores, dim=1)
		label_prob = nn.functional.softmax(label_scores, dim=1)
		return gnn_prob, label_prob

	def loss(self, nodes, labels, train_flag=True):
		"""
		Compute the combined CARE-GNN loss function
		
		Purpose: Train both GNN and similarity modules simultaneously
		Strategy: Weighted combination of two complementary objectives
		
		Args:
			nodes: Batch of node IDs
			labels: True node labels for supervision
			train_flag: Training mode flag
			
		Returns:
			final_loss: Combined loss for gradient-based optimization
			
		Loss Function Breakdown:
		
		1. SIMILARITY LOSS (Equation 4 in paper):
		   - Trains label predictor to distinguish fraud vs benign based on features
		   - Enables label-aware neighbor similarity computation
		   - Helps identify camouflaged fraudsters with similar feature patterns
		
		2. GNN LOSS (Equation 10 in paper):
		   - Trains multi-relation graph neural network for fraud classification
		   - Leverages graph structure and neighbor information
		   - Captures fraud patterns through relation-aware aggregation
		
		3. COMBINED LOSS (Equation 11 in paper):
		   final_loss = gnn_loss + λ₁ × similarity_loss
		   
		   Where λ₁ (lambda_1) balances the two objectives:
		   - λ₁ = 0: Pure GNN training (ignores similarity)
		   - λ₁ = 1: Equal weight to both components  
		   - λ₁ > 1: Emphasizes similarity learning
		"""
		gnn_scores, label_scores = self.forward(nodes, labels, train_flag)
		# Simi loss, Eq. (4) in the paper
		label_loss = self.xent(label_scores, labels.squeeze())
		# GNN loss, Eq. (10) in the paper
		gnn_loss = self.xent(gnn_scores, labels.squeeze())
		# the loss function of CARE-GNN, Eq. (11) in the paper
		final_loss = gnn_loss + self.lambda_1 * label_loss
		return final_loss