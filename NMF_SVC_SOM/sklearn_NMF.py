from sklearn.decomposition import NMF
import process_data as proc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def scale(data):
    scaler = StandardScaler().fit(data)
    scaled_data= scaler.transform(data)
    return scaler, scaled_data

def scale_transform(scaler, data):
    scaled_data= scaler.transform(data)
    return scaled_data

def NMFfit_transform(V, k = 3):
    nmf = NMF(n_components=k, max_iter = 400, alpha = 0.1)
    nmf.fit(V)
    H = nmf.components_
    W = nmf.transform(V)
    return nmf, W

def NMF_transform(nmf, X_test):
    W_xtest = nmf.transform(X_test)
    return W_xtest

if __name__ == '__main__':
    data = proc.read_data()
    f1, yfill = proc.f1_yfill(data)
    scaler, f1_scaled = scale(f1)
    f1X_train, f1X_test, f1y_train, f1y_test =train_test_split(f1_scaled, yfill, test_size=0.20, random_state=42, stratify =yfill)
    nmf_fit, W_train = NMFfit_transform(f1X_train, k=3)
    W_test = NMF_transfrom(nmf_fit, f1X_test)
    #doesnt work with negative input values. Scaling doesn't push everything up to postive values. I could do this (scale all data up to postitive) I think I will try SVM first.
