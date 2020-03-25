import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mplc
import matplotlib
matplotlib.rc('font', family='Arial')

full_visualize_path = '/home/atanwar/mlp-project/Visualize/visualization_heatmaps/'
def plot_visualization_self(sen_no, src_raw, head, i, batch_count, matrix):
	matrix = matrix.data.numpy()
	plt.clf()
	plt.figure(figsize=(15,15))
	plt.imshow(matrix, cmap='afmhot',  interpolation="None")
	my_xticks = src_raw
	my_yticks = src_raw

	ax = plt.gca()
	ax.set_xticks(np.arange(0, len(my_xticks), 1))
	ax.set_yticks(np.arange(0, len(my_yticks), 1))
	ax.xaxis.tick_top()

	plt.tick_params(axis='both', which='major', labelsize=15)
	plt.tick_params(axis='both', which='minor', labelsize=1)

	plt.xticks(np.arange(0, len(my_xticks), 1), my_xticks, rotation='vertical')
	plt.yticks(np.arange(0, len(my_yticks), 1), my_yticks)
	file_name = full_visualize_path + '/self/' + 'batch_no_'+ str(batch_count) + '_sen_no_' + str(sen_no) + '_layer_'+ str(i) +'_head_' + str(head)
	plt.tight_layout()     
	plt.savefig(file_name)   

def plot_visualization_global(srcs, preds, attns, batch_no, sen_no):
	matrix = attns.data.numpy()
	print('matrix:',matrix.shape)
	plt.clf()
	plt.figure(figsize=(15,15))	
	plt.imshow(matrix, cmap='afmhot',  interpolation="None")
	my_xticks = srcs
	my_yticks = preds

	ax = plt.gca()
	ax.set_xticks(np.arange(0, len(my_xticks), 1))
	ax.set_yticks(np.arange(0, len(my_yticks), 1))
	ax.xaxis.tick_top()

	plt.tick_params(axis='both', which='major', labelsize=15)
	plt.tick_params(axis='both', which='minor', labelsize=1)

	plt.xticks(np.arange(0, len(my_xticks), 1), my_xticks, rotation='vertical')
	plt.yticks(np.arange(0, len(my_yticks), 1), my_yticks)
	file_name = full_visualize_path + '/global/' + 'batch_no_'+ str(batch_no) + '_sen_no_' + str(sen_no)
	plt.tight_layout()      
	plt.savefig(file_name) 
