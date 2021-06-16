from keras import backend as K
def recall_m(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) + K.epsilon()
    true_positives = K.sum(y_true * y_pred) + K.epsilon()
    possible_positives = K.sum(y_true) + K.epsilon()
    recall = true_positives / possible_positives
    return recall

def precision_m(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) + K.epsilon()
    true_positives = K.sum(y_true * y_pred) + K.epsilon()
    predicted_positives = K.sum(y_pred) + K.epsilon()
    precision = true_positives / predicted_positives
    return precision

def f1_metric(y_true, y_pred, average="macro"):
    precision_1 = precision_m(y_true, y_pred)
    recall_1 = recall_m(y_true, y_pred)
#     print("p/r 1", K.eval(precision_1), K.eval(recall_1))
    f1_1 = 2.0*precision_1*recall_1 / (precision_1+recall_1)
#     print("f1_1", K.eval(f1_1))
    if average=="macro":
        precision_0 = precision_m((1-y_true), (1-y_pred))
        recall_0 = recall_m((1-y_true), (1-y_pred))
#         print("p/r 0", K.eval(precision_0), K.eval(recall_0))
        f1_0 = 2.0*precision_0*recall_0 / (precision_0+recall_0)
#         print("f1_0", K.eval(f1_0))
        f1 = (f1_1+f1_0)/2.0
#         print("f1  ", K.eval(f1))
        return f1
    else:
        return f1_1

def f1_loss(y_true, y_pred, average="macro"):
    return (1.0 - f1_metric(y_true, y_pred, average=average))