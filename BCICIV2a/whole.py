import numpy as np
import mne
mne.set_log_level('WARNING')
from sklearn.base import BaseEstimator,TransformerMixin,ClassifierMixin
from scipy.signal import cheby2,sosfiltfilt
from scipy.stats import norm
from sklearn.feature_selection import mutual_info_classif
from mne.decoding import CSP

def load_data_moabb(subject_id=1):
    try:
        from moabb.datasets import BNCI2014_001
        from moabb.paradigms import MotorImagery
    except ImportError as e:
        print(f"导入 MOABB失败：{e}")
        return
    try:
        dataset=BNCI2014_001()
        dataset.subject_list[subject_id]
        paradigm = MotorImagery(n_class=4,tmin=0.5,tmax=2.5,channels=None)

        X,y,metadata=paradigm.get_data(dataset=dataset,subjects=[subject_id])
    except Exception as e:
        print(f"提取数据失败：{e}")
        return
    label_dict={'left_hand':1,'right_hand':2,'feet':3,'tongue':4}
    y_int=np.array([label_dict[label] for label in y])

    is_train=metadata['session'].str.contains('train').values
    is_test=metadata['session'].str.contains('test').values
    X_train,y_train=X[is_train],y_int[is_train]
    X_test,y_test=X[is_test],y_int[is_test]
    return X_train,X_test,y_train,y_test,250

class FilterBank(BaseEstimator,TransformerMixin):
    def __init__(self, sfreq=250,bands=[(4,8),(8,12),(12,16),(16,20),(20,24),(24,28),(28,32),(32,36),(36,40)]):
        self.sfreq=sfreq
        self.bands=bands
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        X_flitered=[]
        for low,high in self.bands:
            sos=cheby2(N=4,rs=40,Wn=[low,high],btype='bandpass',fs=self.sfreq,output='sos')
            filtered_band=sosfiltfilt(sos,X,axis=-1)
            X_flitered.append(filtered_band)
        return np.array(X_flitered)
class NBPWClassifier(BaseEstimator,ClassifierMixin):
    def __init__(self):
        self.classes_=None
        self.prior_={}
        self.train_data_={}
        self.h_opt_={}
    def fit(self,X,y):
        selfclass_=np.unique(y)
        n_samples=X.shape[0]
        for c in self.classes_:
            X_c=X[y==c]
            n_c=X_c.shape[0]
            sigma=np.std(X_c,axis=0)
            sigma[sigma==0]=1e-6
            self.h_opt_[c]= ((4.0/(3.0*n_c))**0.2)*sigma
        
        return self
    def predict(self,X):
        n_samples,n_features=X.shape
        probas=np.zeros((n_samples,len(self.classes_)))
        for i,c in enumerate(self.classes_):
            X_c=self.train_data_[c]
            h_c=self.h_opt_[c]
            n_c=len(X_c)
            p_x_given_w=np.ones(n_samples)
            for j in range(n_features):
                x_j=X[:,j][:,np.newaxis]
                x_train_j=X_c[:,j][np.newaxis,:]
                diff=x_j-x_train_j
                kernel_vals=norm.pdf(diff,loc=0,scale=h_c[j])
                p_xj_given_w=np.sum(kernel_vals,axis=1)/n_c
                p_x_given_w*=p_xj_given_w
            probas[:,i]=p_x_given_w*self.prior_[c]
        row_sums=probas.sum(axis=1)[:,np.newaxis]
        row_sums[row_sums==0]=1e-6
        probas_normalized=probas/row_sums
        final_probas=self.classes_[np.argmax(probas_normalized,axis=1)]
        return final_probas
class PairedMIBIF(BaseEstimator, TransformerMixin):
    def __init__(self,k=4,m=2,n_bands=9):
        self.k=k
        self.m=m
        self.n_bands=n_bands
        self.selected_indices_=[]

    def fit(self, X,y):
        mi_scores=mutual_info_classif(X,y)
        top_k_indices=np.argsort(mi_scores)[::-1][:self.k]
        final_indices=set(top_k_indices)
        for idx in top_k_indices:
            band_idx=idx//(2*self.m)
            intra_band_idx=idx%(2*self.m)
            pair_intra_idx=(2*self.m-1)-intra_band_idx
            pair_idx=band_idx*(2*self.m)+pair_intra_idx
            final_indices.add(pair_idx)
        self.selected_indices_=sorted(list(final_indices))
        return self

    def transform(self,x):
        return x[:, self.selected_indices_]

class BinaryFBCSP_Pipline:
    def __init__(self,m=2,k=4):
        self.m=m
        self.csps=[]
        self.selector=PairedMIBIF(k=k,m=m)
        self.clf=NBPWClassifier()
    def fit(self,X_fb,y_binary):
        n_bands=X_fb.shape[0]
        self.csps=[]
        features=[]
        for i in range(n_bands):
            csp=CSP(n_components=2*self.m,reg=None,log=True,norm_trace=False)
            csp.fit(X_fb[i],y_binary)
            self.csps.append(csp)
            features.append(csp.transform(X_fb[i]))
        X_csp=np.concatenate(features,axis=1)
        X_selected=self.selector.fit_transform(X_csp,y_binary)
        self.cls.fit(X_selected,y_binary)
        return self
    def predit_proba(self,X_fb):
        n_bands=X_fb.shape[0]
        features=[]
        for i in range(n_bands):
            features.append(self.csps[i])
        X_csp=np.concatenate(features,axis=1)
        X_selected=self.selector.transform(X_csp)
        idx_positive=np.where(self.cls.classes_==1)[0][0]
        return self.clf.predict_proba(X_selected)[:,idx_positive]
class OVR_FBCSP:
    def __init__(self,classes=[1,2,3,4],m=2,k=4):
        self.classes.classes=classes
        self.models={}
        for c in self.classes:
            self.models[c]=BinaryFBCSP_Pipline(m=m,k=k)
    def fit(self,X,y):
        for c in self.classes:
            y_binary=(y==c).astype(int)
            self.models[c].fit(X,y_binary)
        return self
    def predict(self,X):
        n_samples=X.shape[1]
        probas=np.zeros(n_samples,len(self.classes))
        for i,c in enumerate(self.classes):
            probas[:,i]=self.models[c].predict_proba(X)
        best_class_indices=np.argmax(probas,axis=1)
        return np.array(self.classes[idx] for idx in best_class_indices)
if __name__=="__main__":
    print("\n=== 使用 moabb 官方开源库载入数据 ===")
    X_tr,X_te,y_tr,y_te,sfreq_moabb=load_data_moabb(subject_id=1)
    if X_tr is not None:
        print(f"MOABB 提取 (Subject 1) -> 训练集 Session_T: X={X_tr.shape}, y={y_tr.shape}")
        print(f"MOABB 提取 (Subject 1) -> 测试集 Session_E: X={X_te.shape}, y={y_te.shape}")
        fb=FilterBank()