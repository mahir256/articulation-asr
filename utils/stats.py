import editdistance

def compute_cer(results):
    """Returns the character error rate for the pairs of results
    in the list provided as input. Each pair must consist of a tuple
    or list whose first element is a ground truth and whose second
    element is a prediction.

    TODO: get individual insertion/deletion/substitution counts; see https://gist.github.com/kylebgorman/8034009 for something to adapt
    """
    dist = sum(editdistance.eval(label, pred) for label, pred in results)
    total = sum(len(label) for label, _ in results)
    return dist / total

