"""Exact threshold scoring and deterministic six-level selection."""
from fractions import Fraction

REP_ORDER = {"char_1": 0, "char_1_2": 1, "word_1": 2}
WEIGHT_ORDER = {None: 0, "balanced": 1}

def predict(probabilities, threshold):
    return [int(float(p) >= threshold) for p in probabilities]

def exact_scores(labels, predictions):
    tn=sum(y==0 and p==0 for y,p in zip(labels,predictions))
    fp=sum(y==0 and p==1 for y,p in zip(labels,predictions))
    fn=sum(y==1 and p==0 for y,p in zip(labels,predictions))
    tp=sum(y==1 and p==1 for y,p in zip(labels,predictions))
    n=len(labels)
    acc=Fraction(tp+tn,n) if n else Fraction(0)
    d1=2*tp+fp+fn; f1=Fraction(2*tp,d1) if d1 else Fraction(0)
    d0=2*tn+fp+fn; f0=Fraction(2*tn,d0) if d0 else Fraction(0)
    macro=(f0+f1)/2
    precision=Fraction(tp,tp+fp) if tp+fp else Fraction(0)
    recall=Fraction(tp,tp+fn) if tp+fn else Fraction(0)
    return {"TN":tn,"FP":fp,"FN":fn,"TP":tp,"accuracy_exact":acc,
      "precision_exact":precision,"recall_exact":recall,"f1_exact":f1,
      "f1_0_exact":f0,"macro_f1_exact":macro,
      "Accuracy":float(acc),"Precision":float(precision),"Recall":float(recall),
      "F1":float(f1),"Macro-F1":float(macro),"confusion_matrix":[[tn,fp],[fn,tp]]}

def threshold_records(config, labels, probabilities):
    records=[]
    for index in range(102):
        threshold=index/100
        prediction=predict(probabilities,threshold)
        score=exact_scores(labels,prediction)
        records.append({"config_id":config["config_id"],"representation":config["scheme"],
          "C":config["C"],"class_weight":config["class_weight"],"threshold_index":index,
          "threshold":threshold,**score})
    return records

def selection_key(row):
    return (-row["macro_f1_exact"],-row["accuracy_exact"],-row["threshold"],
      row["C"],WEIGHT_ORDER[row["class_weight"]],REP_ORDER[row["representation"]])

def select_best(rows):
    return min(rows,key=selection_key)

def displayed(row):
    return {k:v for k,v in row.items() if not k.endswith("_exact")}

