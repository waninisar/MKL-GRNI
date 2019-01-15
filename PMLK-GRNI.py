from shogun import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import memory_profiler
from collections import defaultdict
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_curve, auc , precision_score
from sklearn.metrics import confusion_matrix , f1_score 
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score, recall_score
from multiprocessing import Process
from multiprocessing.queues import Queue
import time


"""		Takes Test labels and discriminant scores as input.
        Return Precision, recall and F1 scores.
"""
def calculate_model_metrics(y_test,y_pred):
    print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
    print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
    print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))
   
"""		
        Return thresholds fro different precision levels.
""" 
def generate_thresholds(y_test,y_score):
    precision, recall, thr = precision_recall_curve(y_test.flatten(), y_score.flatten())     
    idx_50=np.where(precision>0.50)
    idx_60=np.where(precision>0.60)
    idx_70=np.where(precision>0.70)
    thr_50=thr[idx_50[0][0]]
    thr_60=thr[idx_60[0][0]]
    thr_70=thr[idx_70[0][0]]
    print("\tThreshold at 50 precision: %1.3f\n" % thr_50)
    print("\tThreshold at 60 precision: %1.3f\n" % thr_60)
    print("\tThreshold at 70 precision: %1.3f\n" % thr_70)
    return (thr_50,thr_60,thr_70)
    
"""		
        Returns AUROC and AUPR for given testlables and confidence scores.
""" 

def calculate_auc_roc_pr(y_test, y_score):
    
    precision, recall, thr = precision_recall_curve(y_test.flatten(), y_score.flatten())
    aupr_auc = auc(recall,precision)
    fpr, tpr, _ = roc_curve(y_test.flatten(), y_score.flatten())
    roc_auc = auc(fpr, tpr)  
    print("\tAUC-PR: %1.3f" % aupr_auc)
    print("\tAUC-ROC: %1.3f" % roc_auc)
    
"""		
        Maps gene names with their indices and returns gene2id and id2gene hashes.
""" 
    
    
def get_gene_id_mapping (gene_list):
    id2gene=defaultdict()
    gene2id =defaultdict()
     
    for idx, gene  in enumerate(gene_list):        
        id2gene[idx]=str(gene)
        gene2id[str(gene)]=idx
    return id2gene, gene2id

"""
Reads the TF-target regulation file and filters missing genes

"""
def genRegulationFile(gene2id):
    tf_reg=np.genfromtxt('datasets/BRCA/hs_brca_regulation.tsv',delimiter='\t', dtype='object')
    f = open("datasets/BRCA/regulation.tsv", "w")
    for i in range(0,len(tf_reg)):
        if tf_reg[i,1] in gene2id:  
            if tf_reg[i,0] in gene2id:  
                f.write(str(gene2id[tf_reg[i,0]])+"\t"+ str(gene2id[tf_reg[i,1]]) + "\n")
#             print str(gene2id[tf_reg[i,0]])+"\t"+ str(gene2id[tf_reg[i,1]])
    regulation=np.genfromtxt("datasets/BRCA/regulation.tsv",delimiter='\t', dtype='int')
    
    return regulation
    
  """
  Returns Shogun machine learning toolbox complaint feature format 
  """
def convert_to_shogun_features(sample_features):
     return (np.transpose(sample_features))

"""
Returns combined kernels in a map
"""
def BuildCombinedKernelMatrix(expr_data, meth_expr):
    
    mexp_matrix = meth_expr
    exp_matrix = expr_data
    kernels= {  'MEXPR':  meth_expr,                
                'EXPR'  : exp_matrix}
    return kernels 

"""
Combine all the datasets using combinedkernels and combined features for MKL training/ MKL testing
"""
def FuseGenomicKernels (indices,kernels):
    
    #print "In FusedGenomicKernels"
  
    expr_shogun_features=convert_to_shogun_features(kernels['EXPR']) 
    feats_train_expr=RealFeatures(expr_shogun_features[:,indices])
    expr_kernel = GaussianKernel(feats_train_expr,feats_train_expr,10)
    mexp_matrix= kernels['MEXPR']
    mexp_shogun_features=convert_to_shogun_features(mexp_matrix)
    feats_train_mexp  = RealFeatures(mexp_shogun_features.astype(np.double)[:,indices])
    mexpr_kernel= GaussianKernel(feats_train_mexp,feats_train_mexp,.2573)
    comb_kernel = CombinedKernel()
    comb_kernel.append_kernel(expr_kernel)
    comb_kernel.append_kernel(mexpr_kernel)    
    comb_features=CombinedFeatures()
    comb_features.append_feature_obj(feats_train_expr)
    comb_features.append_feature_obj(feats_train_mexp)
    return {'kernel' : comb_kernel, 'features' : comb_features}

"""
Returns train/test indice and corresponding combined kernels
"""
def MKLTrainTestData(MKLData,kernels):
    i,j,sourceIndices = np.unique(MKLData['edges'][:,0], return_index=True, return_inverse=True)
    row_count,col_count=[MKLData['nvertices'], MKLData['nsources']]
    CGold=-np.ones([row_count,col_count])
    edges=MKLData['edges']
    for i in range(0,MKLData['nedges']):
        CGold[edges[i,1],sourceIndices[i]] = 1 
     	
    
    all_train=MKLData['knownTargets']
    objects_mkl_train=FuseGenomicKernels (all_train,kernels) 
    combined_kernel_train=objects_mkl_train['kernel']
    combined_features_train=objects_mkl_train['features']
    combined_kernel_train.init(combined_features_train,combined_features_train)
    all_test=MKLData['toPredict']
    objects_mkl_test=FuseGenomicKernels(all_test,kernels) 
    combined_kernel_test=objects_mkl_test['kernel']
    combined_features_test=objects_mkl_test['features']
    combined_kernel_test.init(combined_features_train,combined_features_test)
    return {'TR_IDX': all_train, 'TST_IDX': all_test, 'KTR' : combined_kernel_train , 'KTST' : combined_kernel_test , 'GOLD' : CGold }
    
    
"""
Main function to implement MKL-GRNI approach
INPUT: MKLData structure, TF indices and queue
Returns: Test labels and confidence scores for each classification problem written to a queue object
"""    
def  TrainMKLClassifier (mkldata,index_row,nsources,queues,queueNo):

    
    Object=MKLTrainTestData(mkldata,kernels)
    print("In MKL Training")
    train_indices=mkldata['knownTargets']#Object['TR_IDX']
    test_indices=mkldata["toPredict"] #Object['TST_IDX']
    print (train_indices.shape[0],test_indices.shape[0])
    combined_kernel_train=Object['KTR']
    combined_kernel_test=Object['KTST']
    CGold=Object['GOLD']
    print CGold.shape
    DecisionScores=np.zeros([(nsources-index_row),test_indices.shape[0]],dtype=np.double)
    TestLabels=-np.ones([(nsources-index_row),test_indices.shape[0]],dtype=np.double)	
    k=0 
    score_list=list()

    for i in range(index_row,nsources):
        print ("in MKL Learning section.........")
        mkl = MKLClassification()
        mkl.set_kernel(combined_kernel_train)
        mkl.set_epsilon(1e-4)
        mkl.set_mkl_epsilon(0.001)
        mkl.set_mkl_norm(1)
        mkl.set_C(10,10)
        mkl.set_C_mkl(10)
        trainlabels=CGold[train_indices,i]
        testlabels=CGold[test_indices,i]
        npos=np.sum(trainlabels > 0.5)
        if (npos==0):
            DecisionScores[k]=-np.ones(test_indices.shape[0])
            #print "In if statement"	
        else:
            train_labels=BinaryLabels(trainlabels)
            mkl.set_labels(train_labels)
            mkl.train()    
            mkl.set_kernel(combined_kernel_test)
            output=mkl.apply()
            DecisionScores[k]=output.get_values()
            TestLabels[k]=testlabels
        k=k+1
        del mkl
        mkl=None 
    #print("Storing result in Queue No = +" +`queueNo`  + "(" + `index_row` + "," + `nsources` + ")")
    score_list=[(queueNo,DecisionScores,TestLabels)]
    queues.put(score_list)
    del combined_kernel_train
    del combined_kernel_test
    combined_kernel_train=None
    combined_kernel_test=None
    return (queues)


"""
Function that facilitates parallel execution of MKL-GRNI.
INPUT:  MKLData structure, No. of Tfs and No. of Cores
Returns: A pandas DataFrame with testlabels and Decision scores for all the TF specific 
classification problems retrieved from the shared queue object.
"""

def MKLMultiProcessing(MKLData,CORES,NSOURCES,toPredict):

    Scores=np.zeros([(CORES*(NSOURCES/CORES)),toPredict],dtype=np.object)
    Labels=np.zeros([(CORES*(NSOURCES/CORES)),toPredict],dtype=np.object)
    queues = [Queue() for i in range(CORES)]
    args = [(MKLData,(i*int(NSOURCES/CORES)), int(NSOURCES/CORES)*(i+1),queues[i],i) for i in range(CORES)]
    #print args
    jobs = [Process(target=TrainMKLClassifier, args=(a)) for a in args]
    for j in jobs: j.start()
    i=0
    k=0
    for q in queues: 
        item=q.get()
        l= item[0]
#        print l
        val = l[1]
        lab= l[2]
        for j in range (0,val.shape[0]):
            Scores[k,:]=val[j]
            Labels[k,:]=lab[j]
            k=k+1
        i=i+1
    for j in jobs: j.join()
    df_scores=pd.DataFrame(Scores.T)
    df_testlabels=pd.DataFrame(Labels.T)
    y_score=df_scores.as_matrix()
    y_test=df_testlabels.as_matrix()    
    return y_score,y_test

# This function takes input an gene expression file  , regulation file (TF-Gene pair) and an operon group file to which genes belong
# Return a dictionary with multiple keys and values
def  buildRegulationData (nFeatures,nSamples):
    gene_list_file='/datasets/BRCA/gene_names_GE_methylation.txt'
    df_meth_expr=pd.read_csv('/datasets/BRCA/BRCA.meth.by_mean.txt', sep='\t', header=None)
    df_gene_expr=pd.read_csv('/datasets/BRCA/BRCA.transcriptome_normalised.csv', sep=',', header=None)
    gene_names_ids=np.genfromtxt(gene_list_file,  usecols= (0,1,2), dtype='object')
    if nFeatures > 0:
        gene_names_ids=gene_names_ids[0:nFeatures]
        
    gene_names=gene_names_ids[:,0]
    gene_exp_ids=gene_names_ids[:,1].astype(int)
    meth_exp_ids=gene_names_ids[:,2].astype(int)
    expr_matrix=df_gene_expr.as_matrix()[gene_exp_ids,0:nSamples]       
    meth_expr=df_meth_expr.as_matrix()[meth_exp_ids,1:8]
    genes=gene_names
    _,gene2id=get_gene_id_mapping (gene_names)
    regulation=genRegulationFile(gene2id)
    edges=regulation
    knownTargets= np.unique(edges[:,1])
    nrows, ncols = expr_matrix.shape
    toPredict = np.setdiff1d(np.arange(nrows),knownTargets)
    sources = np.unique(edges[:,0])
    nsource = sources.size
    nvertices , nfeatures = expr_matrix.shape
    MKLData = { 'features'  :    expr_matrix , 
                'edges'     :    edges,
                 'genes'    :    genes,
                'nvertices' :    nvertices,
                'nfeatures' :    nfeatures,
                 'meth_expr':    meth_expr,
                'nedges' :       edges.shape[0],
                'knownTargets' : knownTargets,
                'toPredict' :    toPredict,
                'sources'   :    sources,
                'nsources' :     nsource,                  
                 'nsplitCV':     5 
          }
    
    return  MKLData
    
"""
Function to perform cross validation
"""
def crossValidateMKLinference (cvData):
    mkldata=cvData
    indexList=np.array(range(0,cvData['nedges']),np.int32)    
    iFold=0
    cvFolds=KFold(n_splits=mkldata['nsplitCV'])
#     cvFolds = StratifiedKFold(n_splits=3)
    edges=cvData['edges']
    all_edges=mkldata['edges']
    x, indices= np.unique(all_edges,return_inverse=True)
#     print Dec.T, Agold[k][0]['label']
    edge=np.zeros([mkldata['nedges'],2],dtype=np.int32)
    k=0
    for i in range(0,mkldata['nedges']):
        k=2*i
        for j in range(0,1):
            edge[i,j], edge[i,j+1] = indices[k] , indices[k+1]
    mkldata['knownTargets'] = np.unique(edge[:,1])
    mkldata['sources'] = np.unique(edge[:,0])
    mkldata['features'] = mkldata['features'][x,:]
    mkldata['nvertices'] = x.shape[0]    
    mkldata['edges']=edge
    mkldata['nedges']=edge.shape[0]

    #Gold matrix reduced to non-orphan edges
    Agold= -np.ones([mkldata['nvertices'],mkldata['nvertices']],dtype=np.int32)
    for i in  range(0,mkldata['nedges']):
        row_index, col_index= edge[i,1],edge[i,0]
        Agold[row_index,col_index]=1
    Agold= Agold[:,mkldata['sources']]
    cv_dict=defaultdict(list)
    queues=[Queue() for i in range(cvData['nsplitCV'])]
    for train_idx,test_idx in  cvFolds.split(mkldata['knownTargets']) :
        results={}                
        mkldata['knownTargets']=np.unique(mkldata['edges'][train_idx,1])
        mkldata['toPredict']= np.unique(edge[test_idx,1])
        queues[iFold].put(mkldata)
        #print (len(train_indices),len(test_indices))
        results={ 'label' : Agold[mkldata['toPredict'],:]  }
        cv_dict[iFold].append(results)
        iFold=iFold + 1
    return (queues , cv_dict)

"""
Main Function
"""
if __name__ == '__main__':
    nFeatures=5000
    nSamples=100
    MKLData=buildRegulationData(nFeatures,nSamples)  
    CORES=16
    #NSOURCES=MKLData['nsources']
    NSOURCES=32
    queues = Queue()
    kernels=BuildCombinedKernelMatrix(MKLData['features'], MKLData['meth_expr'])       
    queue , Agold=crossValidateMKLinference(MKLData)  
    scores=defaultdict()
    source=2
    k=0   
    start_time = time.time()
    for item in queue:
        mkl=item.get()       
        #Dec,labels=TrainMKLClassifier(mkl,0,source,queues,1)
        toPredict=mkl['toPredict'].shape[0]
        y_score,y_test = MKLMultiProcessing(mkl,CORES,NSOURCES,toPredict)  

        #y_score=Dec.T
	#y_test=labels
        #y_test=Agold[k][0]['label'][:,0:source]
        thr_50,thr_60,thr_70=generate_thresholds(y_test,y_score)
        calculate_auc_roc_pr(y_test,y_score)
        write_prec_rec_score(y_test,y_score)
        k=k+1      
    	print("--- %s seconds ---" % (time.time() - start_time))
