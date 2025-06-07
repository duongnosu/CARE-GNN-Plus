# =========================================================================
# Data Processing Module
# =========================================================================
# Purpose: Convert sparse adjacency matrices from .mat files to adjacency lists
# This preprocessing step is essential for efficient graph neural network operations
# 
# Key Functions:
# 1. Load graph data from MATLAB format (.mat files)
# 2. Convert sparse matrices to adjacency list format for faster neighbor lookup
# 3. Process both Yelp and Amazon fraud detection datasets
# =========================================================================


from utils import sparse_to_adjlist
from scipy.io import loadmat

"""
	Read data and save the adjacency matrices to adjacency lists
	
	Dataset Structure:
	- Amazon: User-based fraud detection with 3 relations (UPU, USU, UVU)
	
	Relations Explained:
	- UPU: User-Product-User (users reviewing same products)
	- USU: User-Star-User (users giving same star ratings)
	- UVU: User-View-User (users with similar viewing patterns)
"""


if __name__ == "__main__":

	prefix = 'data/'



	amz = loadmat('data/Amazon.mat')  # Load Amazon dataset
	net_upu = amz['net_upu']  # User-Product-User relation matrix
	net_usu = amz['net_usu'] # User-Star-User relation matrix
	net_uvu = amz['net_uvu'] # User-View-User relation matrix
	amz_homo = amz['homo'] # Homogeneous graph (all relations combined)


	  
	           
	              
	# Convert each relation's sparse matrix to adjacency list format
	# This conversion enables faster neighbor lookups during GNN training
	
	# UPU: Users connected through shared product reviews
	# Critical for detecting coordinated review campaigns
	sparse_to_adjlist(net_upu, prefix + 'amz_upu_adjlists.pickle')

	# USU: Users connected through similar rating patterns  
	# Helps identify users with suspiciously similar rating behaviors
	sparse_to_adjlist(net_usu, prefix + 'amz_usu_adjlists.pickle')

	# UVU: Users connected through similar viewing/interaction patterns
	# Captures behavioral similarities beyond just ratings
	sparse_to_adjlist(net_uvu, prefix + 'amz_uvu_adjlists.pickle')
	
	# Homogeneous graph: Combines all relations into single adjacency list
	# Used for baseline comparisons and single-relation GNN models
	sparse_to_adjlist(amz_homo, prefix + 'amz_homo_adjlists.pickle')