#  Machine learning based classification of Schizophrenic and Healthy control Using Resting State fMRI features


#### Objective:
To create an Automated classifier capable of classifying subjects with Schizophrenia from Healthy control using Resting state fMRI data and correlation of relevant brain networks.

#### Existing Techniques:
Most of the techniques involve classification using Task based fMRI data wherein the patient response is recorded with respect to the task given. The use of Resting state fMRI is relatively new and a hot topic in the research community. However existing methods require a visual inspection for the selection of brain networks which is tedious.

#### Motivation:
To devise a workaround to visual inspection while maintaining/improving the efficiency of existing techniques and Utilize Resting state fMRI, so that the results are independent of the patient's task responses and hence would just require the patient to relax.

#### Pre-Processing:
Raw fMRI data were pre-processed using the Statistical Parametric Mapping toolbox (SPM-12) in MATLAB. The following operations were performed in sequence, to obtain the denoised fMRI data. Raw fMRI data in DICOM format was formatted to obtain the standard neuro imaging, .nii (Nifti) format. Then, the fMRI data are slice time realigned to ensure that the data is acquired simultaneously and is normalized with the acquisition time. In order to correct the translational and rotational motion, realign and estimate operation is performed to generate the mean image. Upon this mean image, the origin is set based on the standard configurations of brain anatomy. fMRI data have high temporal resolution and low spatial resolution and in order to compensate, the structural and functional brain images are co-registered.

#### Feature Extraction:
In order to successfully classify the schizophrenic and healthy controls, the most important step is to identify the suitable feature vectors for the classification tasks. It is highly significant to understand the fact that fMRI data have 4 dimensions (4D), hence “the curse of dimensionality” and problems due to over-fitting arise. The literature on schizophrenia research on resting-state data suggests that the Default mode network (DMN), a task-positive network (TPN), and the salience network (SN) are involved. Patients with schizophrenia experienced excess activity in the substantia nigra, decreased activity in the prefrontal cortex, and diminished functional connectivity between these regions, suggesting that communication among these regions was out of sync.
The first step of feature extraction is to perform ICA and generate the time and space matrix. In this work, we considered two cases in which, one case we used 41 Independent Components and the other, 100 Independent Components. Melodic ICA was performed using the FSL toolbox. The corresponding space and time matrices are obtained.

<img align="right" src="https://github.com/surajkra/Automatic-Classification-of-Schizophrenic-Healthy-controls-using-Resting-state-Network-fMRI/blob/master/Images/FSL_Toolbox.png" width="400">


#### Automated Network Identification:
Visual Inspection is very unreliable and needs medical practitioners for further identification and hence analysis. So in order to develop a computer aided automatic schizophrenia classifier, we need an automated network identifier. This topic is still a very hot topic amongst the researchers.
##### Step 1:
The first step involves the computation of Pearson's skewness coefficient on the spatial map. We want our distribution along space to be skewed as this indicates more activity in certain voxels. So we find the median skewness of all rows and eliminate those rows which have less than the median. So about half of the rows are made 0.
##### Step 2:
From Step 1, now for the remaining non-zero rows, each row is segregated into clusters using K-means clustering. From trial-and-error 5 clusters were found to be optimal. Once a row is clustered, the centroids of each cluster were found. The cluster with centroid closest to 0 was identified, its elements were made 0. So in steps 1 and 2, we have effectively eliminated the spatial snaps with low activity in time, and voxels with low activity in each row.
##### Step 3:
The SM from step 2 is obtained. And the corresponding time map is also obtained. Multiplying the 2 maps will give us a Space x Time matrix. So for each Independent component, we multiply the corresponding time series from the Time Map and the corresponding Spatial snapshot from the Spatial Map to get a Space x Time matrix.

##### Step 4:
Now each column can be considered the time series of a voxel corresponding to a particular IC. The time series are detrended and then averaged to get an average time series for that IC. We find the power spectrum using a periodogram. Since Resting state networks are characterized by slow fluctuations of functional imaging signals, the power spectrum of a Network must have most of its power in the low-frequency region and rapidly decrease as the frequency increases.


The networks which contribute significantly (15 components) are the ones (left side pic) whose power spectrum is dominant in the lower frequencies. Those components (right side pic) for which the power spectrum is spread out, and present even in higher freq. contribute very less and hence are rejected.


| Selected Component Periodogram   | Rejected Component Periodogram|
| ------------- |:-------------:|
|  <img src="https://github.com/surajkra/Automatic-Classification-of-Schizophrenic-Healthy-controls-using-Resting-state-Network-fMRI/blob/master/Images/Automatic_Periodogram.png" width="400">   | <img src="https://github.com/surajkra/Automatic-Classification-of-Schizophrenic-Healthy-controls-using-Resting-state-Network-fMRI/blob/master/Images/Rejected_Periodogram.png" width="400"> |


##### Feature Selection:
Once the Independent components are computed and the time and space matrix are obtained, we select features from the identified networks. For case 1, we consider all the 41 independent components while for case 2 from 100 Independent components, automatically 15 components are selected using the above steps.

Functional Network Connectivity and Autoconnectivity features are computed. Functional Network connectivity is calculated by computing the pairwise pearson correlation coefficient and Autoconnectivity is computed by modeling the time course as a regression parameter with its lagged version. A combination of FNC and AC feature forms the third set of features.

The figure to the left indicates the Functional Network Connectivity for a Healthy subject. The figure on the right indicates the Functional Network Connectivity for a Schizophrenic Subject. The amount of activation can be clearly seen in the heat maps and this intuition is exploited for classification purposes.

| Healthy Control Connectivity   | Schizophrenic Patient Connectivity|
| ------------- |:-------------:|
|  <img src="https://github.com/surajkra/Automatic-Classification-of-Schizophrenic-Healthy-controls-using-Resting-state-Network-fMRI/blob/master/Images/Healthy_Subject.png" width="400">   | <img src="https://github.com/surajkra/Automatic-Classification-of-Schizophrenic-Healthy-controls-using-Resting-state-Network-fMRI/blob/master/Images/Patient_Subject.png" width="400"> |
<img align="center" src="https://github.com/surajkra/Automatic-Classification-of-Schizophrenic-Healthy-controls-using-Resting-state-Network-fMRI/blob/master/Images/Feature_Set.png" width="400">

### Results and Inferences
We have evaluated data set based on the 3 feature sets, using
Support Vector Machine
Neural Networks,
-> 10 hidden neurons
-> 50 hidden neurons
-> 100 hidden neurons
-> 500 hidden neurons

#### Dataset for Evaluation:
Our dataset comprises of 40 Schizophrenic and 41 Normal subject patients. All the patients remained at rest during the acquisition of the fMRI data.

#### SVM Results
| Complete Set - Independent Components | 15 Automated Independent Components |
| ------------- |:-------------:|
|  <img src="https://github.com/surajkra/Automatic-Classification-of-Schizophrenic-Healthy-controls-using-Resting-state-Network-fMRI/blob/master/Images/SVM_41_Components.png" width="400">   | <img src="https://github.com/surajkra/Automatic-Classification-of-Schizophrenic-Healthy-controls-using-Resting-state-Network-fMRI/blob/master/Images/SVM_15_Components.png" width="400"> |
<img src="https://github.com/surajkra/Automatic-Classification-of-Schizophrenic-Healthy-controls-using-Resting-state-Network-fMRI/blob/master/Images/SVM_Histogram_Plot.png" width="800">

#### Neural Network Results
<img src="https://github.com/surajkra/Automatic-Classification-of-Schizophrenic-Healthy-controls-using-Resting-state-Network-fMRI/blob/master/Images/NN_Results.png" width="800">
<img  src="https://github.com/surajkra/Automatic-Classification-of-Schizophrenic-Healthy-controls-using-Resting-state-Network-fMRI/blob/master/Images/NN_Histograms.png" width="800">

Comparing the above tables and histograms,  we find that the performance of the automated network identification and feature extraction is better than that of using all 41 components for feature extraction.

<img align="right" src="https://github.com/surajkra/Automatic-Classification-of-Schizophrenic-Healthy-controls-using-Resting-state-Network-fMRI/blob/master/Images/Boxplot.png" width="500">

#### Inferences:

-> Automatic resting state network identification and feature extraction perform better than taking all components into consideration. This step has two advantages. First, it reduces the computational cost and the Second, it makes the entire process automatic eliminating strenuous visual inspection and human intervention.

-> The combined feature set of FNC and AC works better than the individual cases. This is visible in all the test cases(accuracy, specificity, and sensitivity). This is because the feature set here is wholesome which includes the correlations with respect to other networks as well as with itself(time-lagged version).

-> Neural Networks, tend to provide better results when compared to SVM. This is mainly because of the limited samples and high feature size. Neural Networks can fit nonlinear curves for classification but SVMs are restricted to linear fitting curves. It indicates that feature space in its current form is not linearly separable by itself. But the interesting problem here is to identify features that are linearly separable.

-> Upon observing the above parameters, we find that Neural Networks with 100 hidden neurons, serve our purpose. It has the highest median. It also has the highest low value and highest high value indicating high-performance efficiency and less variability when compared to the other two.
