model_ft.eval()
pred_list = []
label_list = []
softmax_list = []
image_list = []
ct12 = 0 
for inputs1, labels1 in dataloaders['val']:
                ct12 += 1
                inputs1 = inputs1.to(device)
                labels1 = labels1.to(device)
                labels1 = labels1.type(torch.cuda.LongTensor)
                # zero the parameter gradients
                # optimizer.zero_grad()
                # forward
                # track history if only in train
                # with torch.set_grad_enabled(phase == 'train'):
                outputs1 = model_ft(inputs1)
                  
                    #print(outputs.shape)
                _, preds1 = torch.max(outputs1, 1)
                pred_list.extend(preds1.cpu().numpy())
                label_list.extend(labels1.data)
                #softmax_list.extend(softmax1(outputs1.cpu().detach().numpy(),axis =1))
                image_list.extend(inputs1)
                if ct12 == 50:
                  break
                  
def softmax1(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats. 
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the 
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter, 
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    
    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p
    
y_pred_prob = [i[1] for i in softmax_list]

!pip install pycm
from pycm import *
cm = ConfusionMatrix(actual_vector=np.asarray(label_list), predict_vector=np.asarray(pred_list))
#print(cm.classes)
#print(cm.table)
print(cm)

# calculate accuracy
from sklearn import metrics
print(metrics.roc_auc_score(label_list, y_pred_prob))
fpr, tpr, thresholds = metrics.roc_curve(label_list, y_pred_prob)

plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)

hpd_cond	Mem_with_Frac_hpd_cond	Total no.of mem with hpd	% of Fractured People in the hpd Condition
PARKINSONS_DISEASE	1734	8671	20.00%
ALCOHOLISM	1652	8491	19.46%
PSYCHIATRIC_DISORDERS_RELATED_TO_MED_CONDITIONS	3253	17983	18.09%
NEUROSIS	218	1207	18.06%
EPILEPSY	2131	11930	17.86%
PSYCHOSES	1152	6491	17.75%
OSTEOPOROSIS	32780	195143	16.80%
BRAIN_CANCER	155	939	16.51%
DEMENTIA	8173	49655	16.46%
MULTIPLE_SCLEROSIS	575	3660	15.71%
SUBSTANCES_RELATED_DISORDERS	3019	19397	15.56%
BIPOLAR	1327	8607	15.42%
IRON_DEFICIENCY_ANEMIA	8586	56363	15.23%
LEUKEMIA_OR_MYELOMA	1036	6885	15.05%
HEMOPHILIA_OR_CONGENITAL_COAGULOPATHIES	152	1033	14.71%
LUNG_CANCER	1226	8361	14.66%
RHEUMATOID_ARTHRITIS	4468	31086	14.37%
HYPERCOAGUABLE_SYNDROME	777	5433	14.30%
PEPTIC_ULCER_DISEASE	2534	17798	14.24%
SYSTEMIC_LUPUS_ERYTHEMATOSUS	677	4765	14.21%
FIBROMYALGIA	3312	23518	14.08%
HEART_FAILURE	1589	72758	13.93%
DEPRESSION	22697	163852	13.85%
ATRIAL_FIBRILLATION	10640	77014	13.82%
LOW_BACK_PAIN	38393	291303	13.18%
CEREBROVASCULAR_DISEASE	15251	118104	12.91%
PANCREATITIS	1232	9627	12.80%
CONGENTIAL_HEART_DISEASE	 	6421	12.79%
ISCHEMIC_HEART_DISEASE	10454	82216	12.72%
HODGKINSDISEASE_OR_LYMPHOMA	871	6896	12.63%
INFLAMMATORY_BOWEL_DISEASE	1358	10788	12.59%
ATTENTION_DEFICIT_DISORDER	310	2487	12.46%
ANXIETY	21033	168746	12.46%
CHRONIC_OBSTRUCTIVE_PULMONARY_DISEASE	16313	131030	12.45%
EATING_DISORDERS	148	1195	12.38%
PERIPHERAL_ARTERY_DISEASE	13621	112059	12.16%
OSTEOARTHRITIS	37565	315249	11.92%
OTHER_CANCER	3611	30368	11.89%
VENTRICULAR_ARRHYTHMIA	2861	24443	11.70%
HEPATITIS	1121	9596	11.68%
CHOLELITHIASIS_OR_CHOLECYSTITIS	1589	13689	11.61%
CHRONIC_RENAL_FAILURE	13317	116281	11.45%
MIGRAINE_AND_OTHER_HEADACHES	11904	104036	11.44%
CHRONIC_FATIGUE_SYNDROME	1906	16730	11.39%
PANCREATIC_CANCER	125	1104	11.32%
NONSPECIFIC_GASTRITIS_OR_DYSPEPSIA	39029	346212	11.27%
LOW_VISION_AND_BLINDNESS	830	7508	11.05%
HEAD_OR_NECKCANCER	176	1601	10.99%
STOMACH_CANCER	100	912	10.96%
ASTHMA	9201	83996	10.95%
COLORECTAL_CANCER	1291	11914	10.84%
KIDNEY_STONES	2773	25735	10.78%
SKIN_CANCER	10249	96787	10.59%
OVARIAN_CANCER	373	3548	10.51%
BREAST_CANCER	6877	66241	10.38%
BLADDER_CANCER	435	4240	10.26%
CHRONIC_THYROID_DISORDERS	28977	285810	10.14%
ALLERGY	19663	193959	10.14%
MALIGNANT_MELANOMA	1422	14266	9.97%
HYPERTENSION	59656	603540	9.88%
DIVERTICULAR_DISEASE	11434	116402	9.82%
DIABETES_MELLITUS	21943	228794	9.59%
CERVICAL_CANCER	222	2320	9.57%
MENOPAUSE	12309	129188	9.53%
HYPERLIPIDEMIA	60959	645911	9.44%
ORAL_CANCER	158	1685	9.38%
CATARACT	36183	389812	9.28%
OTITIS_MEDIA	3420	37687	9.07%
GLAUCOMA	15605	173843	8.98%
OBESITY	20297	233412	8.70%
ENDOMETRIAL_CANCER	811	9506	8.53%
LYME_DISEASE	290	3406	8.51%
METABOLIC_SYNDROME	3873	46053	8.41%
HIV_OR_AIDS	51	660	7.73%
FEMALE_INFERTILITY	55	806	6.82%
PERIODONTAL_DISEASE	1886	29867	6.31%
ENDOMETRIOSIS	67	1086	6.17%
SICKLE_CELL_ANEMIA	35	584	5.99%


Sum of Days Supply of Osteo causing Drug	Bin 	200	200 - 300	400	500	600	700	800	900	1000	1100	1200	1300	1300+
	No. of members	130130	3979	2996	2843	2811	1878	1978	2888	1722	1666	2549	2621	1543
Osteo - Diagnosed - members	No. of members	41214	1646	1301	1185	1204	828	918	1310	817	816	1160	1162	781
Fractured -  members	No. of members	15663	559	474	406	327	266	295	390	259	268	378	388	258
		32%	41%	43%	42%	43%	44%	46%	45%	47%	49%	46%	44%	51%
  
  	Ones	Zeros	Population	TargetRate	CumulativeTargetRate	TargetsCaptured	Non_Event_Captured	Cumulative_population	Lift	KS
Decile										
0.(0.57,0.91]	3382.0	36622.0	40004	0.084542	0.084542	0.312252	0.094088	0.099995	3.122683	0.218164
1.(0.49,0.57]	1864.0	38138.0	40002	0.046598	0.065570	0.484350	0.192072	0.199985	2.421940	0.292279
2.(0.41,0.49]	1391.0	38606.0	39997	0.034778	0.055307	0.612778	0.291257	0.299962	2.042854	0.321521
3.(0.37,0.41]	988.0	39022.0	40010	0.024694	0.047652	0.703998	0.391511	0.399972	1.760120	0.312486
4.(0.32,0.37]	912.0	38596.0	39508	0.023084	0.042787	0.788201	0.490671	0.498726	1.580427	0.297529
5.(0.29,0.32]	685.0	39437.0	40122	0.017073	0.038482	0.851445	0.591992	0.599016	1.421406	0.259453
6.(0.23,0.29]	600.0	39157.0	39757	0.015092	0.035154	0.906841	0.692593	0.698393	1.298468	0.214248
7.(0.18,0.23]	410.0	38957.0	39367	0.010415	0.032099	0.944696	0.792680	0.796796	1.185618	0.152015
8.(0.17,0.18]	277.0	40538.0	40815	0.006787	0.029226	0.970271	0.896830	0.898818	1.079496	0.073441
9.[0.15,0.17]	322.0	40157.0	40479	0.007955	0.027073	1.000000	1.000000	1.000000	1.000000	0.000000


