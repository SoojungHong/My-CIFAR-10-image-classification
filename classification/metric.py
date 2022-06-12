import numpy as np


def accuracy(y_true, y_pred):
    correct_predictions = 0

    for yt, yp in zip(y_true, y_pred):

        if yt == yp:
            correct_predictions += 1

    return correct_predictions / len(y_true)


def true_positive(y_true, y_pred):
    tp = 0

    for yt, yp in zip(y_true, y_pred):

        if yt == 1 and yp == 1:
            tp += 1

    return tp


def true_negative(y_true, y_pred):
    tn = 0

    for yt, yp in zip(y_true, y_pred):

        if yt == 0 and yp == 0:
            tn += 1

    return tn


def false_positive(y_true, y_pred):
    fp = 0

    for yt, yp in zip(y_true, y_pred):

        if yt == 0 and yp == 1:
            fp += 1

    return fp


def false_negative(y_true, y_pred):
    fn = 0

    for yt, yp in zip(y_true, y_pred):

        if yt == 1 and yp == 0:
            fn += 1

    return fn


def macro_precision(y_true, y_pred):
    num_classes = len(np.unique(y_true))

    precision = 0

    for class_ in list(y_true.unique()):
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        tp = true_positive(temp_true, temp_pred)
        fp = false_positive(temp_true, temp_pred)

        temp_precision = tp / (tp + fp + 1e-6)
        precision += temp_precision

    precision /= num_classes

    return precision


def precision_per_class(y_true, y_pred):
    num_classes = len(np.unique(y_true))
    precision = 0
    precisions = []

    for class_ in list(np.unique(y_true)) : 
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        tp = true_positive(temp_true, temp_pred)
        fp = false_positive(temp_true, temp_pred)

        temp_precision = tp / (tp + fp + 1e-6)
        precision += temp_precision
        precisions.append(round(temp_precision,2))

    precision /= num_classes

    return precisions 


def macro_recall(y_true, y_pred):
    num_classes = len(np.unique(y_true))

    recall = 0

    for class_ in list(np.unique(y_true)) : 
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        tp = true_positive(temp_true, temp_pred)
        fn = false_negative(temp_true, temp_pred)

        temp_recall = tp / (tp + fn + 1e-6)
        recall += temp_recall

    recall /= num_classes

    return recall


def recall_per_class(y_true, y_pred):
    num_classes = len(np.unique(y_true))

    recall = 0
    recalls = []

    for class_ in list(np.unique(y_true)) :      
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        tp = true_positive(temp_true, temp_pred)
        fn = false_negative(temp_true, temp_pred)

        temp_recall = tp / (tp + fn + 1e-6)
        recall += temp_recall
        recalls.append(round(temp_recall, 2))

    recall /= num_classes

    return recalls 


def average_precision(precisions, recalls):
    recalls = np.array(recalls)
    precisions = np.array(precisions) 

    average_precisions = []

    for i in range(len(recalls)): 
      average_precisions.append(precisions[i] * recalls[i]) 
    
    return average_precisions


def meanAveragePrecision(average_precisions, numClass):
    print('average_precisions:', average_precisions) 
    print('num class:', numClass) 
    total_ap = sum(average_precisions)
    mAP = total_ap/numClass 
    print('mAP:', mAP) 
    return mAP











