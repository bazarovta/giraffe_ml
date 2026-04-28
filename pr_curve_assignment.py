import numpy as np


def build_precision_recall_curve(
    true_labels: np.ndarray, predicted_probas: np.ndarray
) -> np.ndarray:
    """
    Данная функция строит PR-кривую для задачи бинарной классификации.
    В случае, когда нет ни одного объекта положительного класса функция должна вызывать ValueError().

    Args:
        true_labels (np.ndarray): Массив истинных меток класса. Состоит из 0 и 1.
            1 считается меткой положительного класса.
        predicted_probas (np.ndarray): Массив предсказанных вероятностей принадлежности объекта
            к положительному классу.

    Returns:
        np.ndarray: Массив размерами (len(true_labels)+1, 2), где в каждой строчке стоит пара (recall, precision), первым элементом всегда идет (0, 1)
    """
    d = {}
    for i in range(0, len(true_labels)):
        d[i] = predicted_probas[i]
    index = sorted(d.items(), key=lambda x: x[1], reverse=True)
    t_l = []
    p_p = []
    for i in index:
        p_p.append(i[1])
        t_l.append(true_labels[i[0]])     
    pr_curve = [(0,1)]

    
    for trashhold in p_p:
        tp = 0
        fp = 0
        fn = 0
        for i in range(0, len(p_p)):
            if p_p[i] >= trashhold and t_l[i] == 1:
                tp += 1
            elif p_p[i] >= trashhold and t_l[i] == 0:
                fp += 1
            elif p_p[i] < trashhold and t_l[i] == 1:
                fn += 1
        if tp == 0 and fp == 0:
            pr_curve.append((0, 1))
        elif tp == 0 and fn == 0:
            raise ValueError()
        else:
            pr_curve.append( ( tp/(tp + fn), tp/(tp + fp) ))
    
    
    return pr_curve
    