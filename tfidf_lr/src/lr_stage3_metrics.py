"""Independent confusion-count metrics and sklearn comparison."""
from fractions import Fraction
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

def from_counts(labels,predictions):
    tn=sum(y==0 and p==0 for y,p in zip(labels,predictions)); fp=sum(y==0 and p==1 for y,p in zip(labels,predictions))
    fn=sum(y==1 and p==0 for y,p in zip(labels,predictions)); tp=sum(y==1 and p==1 for y,p in zip(labels,predictions)); n=len(labels)
    acc=Fraction(tn+tp,n)
    f1_1=Fraction(2*tp,2*tp+fp+fn) if 2*tp+fp+fn else Fraction(0)
    f1_0=Fraction(2*tn,2*tn+fp+fn) if 2*tn+fp+fn else Fraction(0)
    precision=Fraction(tp,tp+fp) if tp+fp else Fraction(0)
    recall=Fraction(tp,tp+fn) if tp+fn else Fraction(0)
    macro=(f1_0+f1_1)/2
    return {'TN':tn,'FP':fp,'FN':fn,'TP':tp,'n':n,'accuracy':float(acc),'precision_1':float(precision),
      'recall_1':float(recall),'f1_1':float(f1_1),'f1_0':float(f1_0),'macro_f1':float(macro),
      'exact':{'accuracy':str(acc),'precision_1':str(precision),'recall_1':str(recall),'f1_1':str(f1_1),'f1_0':str(f1_0),'macro_f1':str(macro)},
      'confusion_matrix':[[tn,fp],[fn,tp]],'prediction_counts':{'0':tn+fn,'1':fp+tp},'error_count':fp+fn}

def compare_sklearn(labels,predictions):
    metrics=from_counts(labels,predictions)
    sk={'accuracy':accuracy_score(labels,predictions),'precision_1':precision_score(labels,predictions,pos_label=1,zero_division=0),
      'recall_1':recall_score(labels,predictions,pos_label=1,zero_division=0),'f1_1':f1_score(labels,predictions,pos_label=1,zero_division=0),
      'macro_f1':f1_score(labels,predictions,labels=[0,1],average='macro',zero_division=0),
      'confusion_matrix':confusion_matrix(labels,predictions,labels=[0,1]).tolist()}
    for key in ('accuracy','precision_1','recall_1','f1_1','macro_f1'):
        if abs(metrics[key]-sk[key])>1e-14: raise AssertionError(f'{key} differs from sklearn')
    if metrics['confusion_matrix']!=sk['confusion_matrix']: raise AssertionError('confusion matrix differs from sklearn')
    return metrics,sk
