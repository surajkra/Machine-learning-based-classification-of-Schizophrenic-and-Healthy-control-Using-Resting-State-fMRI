import nibabel as nib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score 
from scipy.sparse import csc_matrix
from scipy.sparse import spdiags 
import scipy.sparse.linalg as spl
from scipy import signal
import matplotlib.pyplot as plt

img= nib.load('./melodic_IC.nii')

img = img.get_data()
img = np.array(img)
nc  = img.shape[3]			#No of Independent Components

mix = np.genfromtxt('./melodic_mix') 	#Mix Matrix - Time vs No. of components
t = mix.shape[0]			#Number of time repetitions

S = np.zeros((nc,147456))
P = np.zeros((nc,1))


def pear_skew(arr):
	m_ = np.mean((arr))
	me = np.median((arr))
	s = np.std((arr))
	if(s==0):
		return 0
	else:
		return 3*(m_ - me)/s
print 'Flatten Spatial map and calculate Pearson Skew'	

for i in range (0,nc):
	arr = img[:,:,:,i]
	arr = np.ndarray.flatten((arr))
	S[i,:] = arr
	P[i] = pear_skew(arr)
#print P
P_med = np.median((P))
#print P_med

j = 0

for i in range(0,nc):
	if (P[i]>=P_med):
		j = j + 1

S_new = np.zeros((j,147456))
mix_new = np.zeros((180,nc))
k = 0

indx = []

for i in range(0,nc):
	if(P[i]>=P_med):
		S_new[k,:] = S[i,:]
		mix_new[:,k] = mix[:,i]
		indx.append(i)
		k = k + 1

#print indx
S_new = np.append(S_new, np.zeros((nc-k,147456)),axis=0)
#mix_new = np.append(mix_new,np.zeros((180,nc-k)),axis=1)
#print S_new.shape,mix_new.shape

print 'K-Means CLustering and Voxel Elimination'
for i in range(0,k):
	art = S_new[i,:]
	y = range(len(art))
	art = np.matrix([art,y]).transpose()
	#print art[40500:40550,:]

	clusterer = KMeans(n_clusters=5,random_state = 10)
	cluster_labels = np.array((clusterer.fit_predict(art)))
	
	min_clust = np.argmin((np.absolute((clusterer.cluster_centers_[:,0]))))
	low_indices = np.where((cluster_labels == min_clust))
	#print low_indices,min_clust
	S_new[i,low_indices] = 0
	#print (np.where((S_new[i,:] != 0)))

np.savetxt('S_new.csv',S_new,delimiter=',')
np.savetxt('mix_new.csv',mix_new,delimiter=',')


#S_new = np.genfromtxt('./S_new.csv')
#mix_new = np.genfromtxt('./mix_new.csv')
#print np.where((S_new!=0))

#k = 50

X_nm = np.zeros((180,145476))
temp1 = np.zeros((t,1))
temp2 = np.zeros((1,147456))

print 'Spectral Analysis'

P1_m = np.zeros((k,1))
P2_m = np.zeros((k,1))

for i in range(0,k):
	temp1 = mix_new[:,i]
	temp1 = temp1.reshape(temp1.shape[0],1)

	temp2 = S_new[i,:]
	temp2 = temp2.reshape(temp2.shape[0],1)
	temp2 = temp2.transpose()

	X_nm = np.dot(temp1,temp2)
	#print '\t Signal Detrending'
	X_nm = signal.detrend(X_nm)
	X_avg = np.mean(X_nm,axis=0)
	
	#print '\t Periodogram'
	f,X_per = signal.periodogram(X_avg)
	X_per = X_per.reshape(X_per.shape[0],1)
	a = f[-1]
	f1 = a/X_per.shape[0]
	f2 = np.arange(0.01,0.1,a/X_per.shape[0])
	
	den = np.sum(X_per)
	P1 = np.sum(X_per[:1475])/den
	P2 = np.sum(X_per[1475:14755])/den
	
	P1_m[i] = P1
	P2_m[i] = P2

	#print X_per.shape
	#plt.figure()
	#plt.plot(f,X_per)
	#plt.semilogy(f,X_per)

#plt.show()
print P2_m,P1_m+P2_m
P2_max = np.amax(P2_m)
print P2_max
P12_max = np.amax(P1_m+P2_m)
print P12_max
print 'component selection'

comp = []

count_comp = 0
inc = 0
power_components=np.zeros((len(comp)))
for i in range(0,k):
	if(  (P2_m[i]<0.5) and (P1_m[i]+P2_m[i])<0.7*P12_max):
		count_comp = count_comp + 1
	else:
		comp.append(indx[i])
		power_components[inc]=P1_m[i]+P2_m[i]
		inc = inc + 1
var=[inc[0] for inc in sorted(enumerate(power_components), key=lambda x:x[1])]
print 'Var'
print var
var=var[len(var)-15:len(var)]
print 'Modified Var'
print var
#mix_auto_reduced=mix_auto_reduced[:,len(comp):len(comp)-15]
#print count_comp
#print comp
#print len(comp)
mix_final = np.zeros((180,len(var)))
for i in range(0,len(var)):
	mix_final[:,i] = mix[:,comp[var[i]]]
np.savetxt('mix_auto.csv',mix_final,delimiter=',')
print 'fiin'

