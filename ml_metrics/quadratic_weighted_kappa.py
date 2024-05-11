import numpy
from functools import reduce

def confusion_matrix(rater_a, rater_b, min_rating = None, max_rating = None):
    "Returns the consfusion matrix between rater's rating"
    assert(len(rater_a) == len(rater_b))
    if min_rating is None: min_rating = min(reduce(min, rater_a), reduce(min, rater_b))
    if max_rating is None: max_rating = max(reduce(max, rater_a), reduce(max, rater_b))
    num_ratings = max_rating - min_rating + 1
    conf_mat = [[0 for i in range(num_ratings)]    for j in range(num_ratings)]
    for a,b in zip(rater_a, rater_b):
        conf_mat[a-min_rating][b-min_rating] += 1
    return conf_mat

def histogram(ratings, min_rating = None, max_rating = None):
    "Returns the count of each type of rating that a rater made"
    if min_rating is None: min_rating = reduce(min, ratings)
    if max_rating is None: max_rating = reduce(max, ratings)
    num_ratings = max_rating - min_rating + 1
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r-min_rating] += 1
    return hist_ratings

def quadratic_weighted_kappa(rater_a, rater_b, min_rating = None, max_rating = None):
    "Calculates the quadratic weighted kappa"
    assert(len(rater_a) == len(rater_b))
    if min_rating is None: min_rating = min(reduce(min, rater_a), reduce(min, rater_b))

    if max_rating is None: max_rating = max(reduce(max, rater_a), reduce(max, rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b, min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j] / num_scored_items)
            d = pow(i-j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return 1.0 - numerator/denominator

def linear_weighted_kappa(rater_a, rater_b, min_rating = None, max_rating = None):
    "Calculates linear weighted kappa"
    assert(len(rater_a) == len(rater_b))
    if min_rating is None: min_rating = min(reduce(min, rater_a), reduce(min, rater_b))
    if max_rating is None: max_rating = max(reduce(max, rater_a), reduce(max, rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b, min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0 
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j] / num_scored_items)
            d = abs(i-j) / float(num_ratings - 1)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items
    return 1.0 - numerator / denominator

def kappa(rater_a, rater_b, min_rating = None, max_rating = None):
    "Calculates Kappa"
    assert(len(rater_a) == len(rater_b))
    if min_rating is None: min_rating = min(reduce(min, rater_a), reduce(min, rater_b))
    if max_rating is None: max_rating = max(reduce(max, rater_a), reduce(max, rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b, min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0 
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j] / num_scored_items)
            if i == j:
                d = 0.0
            else:
                d = 1.0
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items
    return 1.0 - numerator / denominator

def mean_quadratic_weighted_kappa(kappas, weights= None):
    "Calculates mean of quadratic weighted kappas after applying Fisher's r-to-z transform"
    kappas = numpy.array(kappas, dtype=float)
    if weights is None:
        weights = numpy.ones(numpy.shape(kappas))
    else:
        weights /= numpy.mean(weights)

    kappas = numpy.array([min(x, .999) for x in kappas])
    kappas = numpy.array([max(x, -0.999) for x in kappas])

    z = 0.5 * numpy.log((1+kappas)/(1-kappas)) * weights
    z = numpy.mean(z)
    kappa = (numpy.exp(2*z) - 1) / (numpy.exp(2*z) + 1)
    return kappa 
