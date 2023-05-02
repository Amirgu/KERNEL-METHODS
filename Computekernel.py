import networkx as nx
from Reconstruct import *
import pickle

from Kernel import weisfeilerlehmankernel
from algorithms import load_graphs
from grakel import *
import time

### Loading :

with open('/home/amine/Graphs1200.pkl','rb') as f:    ###LOAD RECONSTRUCTED SET 
   Gtrain=pickle.load(f)

with open('/home/amine/projet/data/test_data.pkl','rb') as f:         ###LOAD TEST SET
   Gtest=pickle.load(f)

### Concatenating Reconstructed Training set and test set:

G3=[]
for i in range(len(Gtrain)):
   G3.append(Gtrain[i])
for i in range(len(Gtest)):
   G3.append(Gtest[i])

G = graph_from_networkx(G3, node_labels_tag="labels",edge_labels_tag="labels")

start_time = time.time()

###Computing Kernel Matrix: 


tk=WeisfeilerLehmanOptimalAssignment(n_jobs=2, verbose=False, normalize=False, n_iter=3, sparse=False)



A=tk.fit_transform(G)

end_time = time.time()
total_time = end_time - start_time

print(f"Time taken: {total_time} seconds")



with open('KERNELOARECONSTRUCTEDTRAINORMALTEST.pkl','wb') as f:
   ###SAVE KERNEL MATRIX
   pickle.dump(A,f)
