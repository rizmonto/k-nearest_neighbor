import math
import collections

def dot(v, w):
    """v_1 * w_1 + ... + v_n * w_n"""
    return sum(v_i * w_i for v_i, w_i in zip(v, w))

def sum_of_squares(v):
    """v_1 * v_1 + ... + v_n * v_n"""
    return dot(v, v)

def vector_subtract(v, w):
    """subtracts two vectors componentwise"""
    return [v_i - w_i for v_i, w_i in zip(v,w)]

def squared_distance(v, w):
    return sum_of_squares(vector_subtract(v, w))

def distance(v, w):
    return math.sqrt(squared_distance(v, w))

def majority_vote(labels):
    '''used to have nearest points cast a vote to classify the point in question'''

    # init max number of winning majority vote
    num_winners = 1

    """assumes that labels are ordered from nearest to farthest"""
    vote_counts = collections.Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]

    #convert to a regular list of elements
    win_count = [x for _, x in vote_counts.items()]

    # if there's more than 1 label, then 
    if len(win_count) > 1:
        # check the number of equal highest votes
        for count in win_count:
            if count == winner_count:
                num_winners += 1

    if num_winners == 1:
        return winner                     # unique winner, so return it
    else:
        return majority_vote(labels[:-1]) # try again without the farthest

def knn_classify(number_of_neighbors, all_labeled_points, point_to_classify):
    # all_labeled_points should be ([xy coordinate], label)

    # order points from nearest to farthest
    all_sorted_points = sorted(all_labeled_points, key = lambda (point, _): distance(point, point_to_classify))

    # find the labels of the nearest points up the number of neighbors
    nearest_points = [label for _, label in all_sorted_points[:number_of_neighbors]]
    
    # then have them vote
    return majority_vote(nearest_points)