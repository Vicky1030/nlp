"""Synthetic contract tests for scoring, tie-breaking and record formatting."""
from fractions import Fraction
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
from lr_stage2_scoring import exact_scores,predict,selection_key

def run_contract_tests():
    checks={}
    assert predict([0.5],0.5)==[1]; checks['threshold_equal_is_positive']=True
    assert predict([0.0,1.0],0.0)==[1,1]
    assert predict([0.0,1.0],1.0)==[0,1]
    assert predict([0.0,1.0],1.01)==[0,0]
    checks['threshold_edges_0_1_1p01']=True
    all0=exact_scores([0,1],[0,0]); all1=exact_scores([0,1],[1,1])
    assert all0['TN']==1 and all0['FN']==1 and all1['FP']==1 and all1['TP']==1
    assert exact_scores([0,0],[0,0])['f1_exact']==0
    assert exact_scores([1,1],[1,1])['f1_0_exact']==0
    checks['all_one_class_and_zero_denominators']=True
    labels=[0,1]; scores=exact_scores(labels,[0,0])
    independent=float((f1_score(labels,[0,0],labels=[0,1],average=None,zero_division=0).sum())/2)
    assert scores['Macro-F1']==independent
    checks['macro_f1_fixed_labels_0_1']=True
    base={'macro_f1_exact':Fraction(1,2),'accuracy_exact':Fraction(1,2),'threshold':0.5,'C':1.0,
          'class_weight':None,'representation':'char_1'}
    # Each tie-break is exercised while later keys deliberately favor the loser.
    macro=dict(base,macro_f1_exact=Fraction(3,5),accuracy_exact=Fraction(1,4),threshold=0.1,C=10.0,class_weight='balanced',representation='word_1')
    assert selection_key(macro)<selection_key(base)
    accuracy=dict(base,accuracy_exact=Fraction(3,5),threshold=0.1,C=10.0,class_weight='balanced',representation='word_1')
    assert selection_key(accuracy)<selection_key(base)
    threshold=dict(base,threshold=0.6,C=10.0,class_weight='balanced',representation='word_1')
    assert selection_key(threshold)<selection_key(base)
    low_c=dict(base,C=0.1,class_weight='balanced',representation='word_1')
    assert selection_key(low_c)<selection_key(base)
    none=dict(base,class_weight=None); balanced=dict(base,class_weight='balanced')
    assert selection_key(none)<selection_key(balanced)
    assert selection_key(dict(base,representation='char_1'))<selection_key(dict(base,representation='char_1_2'))
    checks['six_lexicographic_levels_direction_and_ties']=True
    # Two exact scores may format identically at 3 decimals; exact ordering remains strict.
    a=dict(base,macro_f1_exact=Fraction(5000001,10000000)); b=dict(base,macro_f1_exact=Fraction(5000000,10000000))
    assert f'{float(a["macro_f1_exact"]):.3f}'==f'{float(b["macro_f1_exact"]):.3f}' and selection_key(a)<selection_key(b)
    checks['rounded_display_does_not_drive_selection']=True
    # Metric/matrix orientation and counts are checked against sklearn.
    y=[0,0,1,1]; pred=[0,1,0,1]; m=exact_scores(y,pred)
    assert m['confusion_matrix']==confusion_matrix(y,pred,labels=[0,1]).tolist()
    assert m['TN']+m['FP']+m['FN']+m['TP']==len(y)
    assert m['Accuracy']==accuracy_score(y,pred)
    assert m['Precision']==precision_score(y,pred,zero_division=0)
    assert m['Recall']==recall_score(y,pred,zero_division=0)
    assert m['F1']==f1_score(y,pred,zero_division=0)
    checks['metric_orientation_and_denominators']=True
    return checks
