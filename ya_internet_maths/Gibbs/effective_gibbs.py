import copy
from math import exp
import numpy
import random

__author__ = 'Alexandra Mikhaylova mikhaylova@yandex-team.ru'

TRAINING_SET = 'train'
TESTING_SET = 'test'
OUTPUT = 'out'
OUTPUT_BEST = 'out_best'

STEPS = 5000
C_EXP = 0.001
LEARN_SAMPLES_COUNT = 97290
FEATURES_COUNT = 245

def num(ls) :
    return [int(float(ls[0])), int(ls[1]), float(ls[2])]

def newVector() :
    v = []
    for x in xrange(FEATURES_COUNT) :
        v.append(0)
    return v

def newRandVector() :
    v = []
    r = random.Random()
    for x in xrange(FEATURES_COUNT) :
        v.append(r.random())
    return v

def multiplyVectors(x, y) :
    res = 0.0
    for i in xrange(FEATURES_COUNT) :
        res += x[i] * y[i]
    return res

def newSample() : # [vector 0, y = 0]
    return [newVector(), 0]

def readData(filename) : # sample = [X, y] where y is relevance
    f = open(filename, 'r')
    samples = []
    new_sample = newSample() # start
    sample_number = 1 # start
    for line in f :
        line_split = line.split()
        l = num(line_split)
        if l[0] > sample_number :
            samples.append(new_sample) # [X, y]
            new_sample = newSample()
            sample_number += 1
            print sample_number
        if l[1] == FEATURES_COUNT + 1 :
            new_sample[1] = l[1]
        else :
            new_sample[0][l[1] - 1] = l[2]
    samples.append(new_sample) # the last
    f.close()
    return samples

def mse(coeff, samples) : # sample = [X, y] where y is relevance
    result = 0.0
    for sample in samples :
        (x, y) = sample
        result += (y - multiplyVectors(x, coeff))**2
    result /= LEARN_SAMPLES_COUNT
    print result
    return result

def target(coeff, samples) : # exp(-MSE)
    x = C_EXP * mse(coeff, samples)
    return exp((-1) * x)


### utils for effective work
def getDotList(coeff, samples) : # on start
    dot_list = []
    for sample in samples :
        (x, y) = sample
        dot_list.append(multiplyVectors(coeff, x))
    return dot_list

def modifyDotList(dot_list, samples, step, index) :
    for i in xrange(len(samples)) :
        (x, y) = samples[i]
        dot_list[i] += x[index] * step
    return dot_list

def getCurrentMSE(dot_list, samples) :
    mse = 0
    for i in xrange(len(samples)) :
        (x, y) = samples[i]
        mse += (y - dot_list[i]) ** 2
    return mse / LEARN_SAMPLES_COUNT

def effectiveTarget(mse) :
    return exp((-1) * C_EXP * mse)
###


def learnCoefficients(samples) :
    coeff = newVector()
#    coeff = newRandVector()
    best_coeff = copy.deepcopy(coeff)
    dot_list = getDotList(coeff, samples)
    mse = getCurrentMSE(dot_list, samples)
    target_current = effectiveTarget(mse)
    distribution = numpy.random.normal(0.005, 0.1, STEPS)
    psi_distribution = numpy.random.uniform(0.0, 1.0, STEPS)
    for i in xrange(STEPS) :
        print 'Iteration ', i
        direction = random.randint(0, FEATURES_COUNT - 1)
        step = distribution[i]
        coeff[direction] += step
        dot_list = modifyDotList(dot_list, samples, step, direction)
        mse = getCurrentMSE(dot_list, samples)
        target_new = effectiveTarget(mse)
        psi = psi_distribution[i]
        print 'PSI ', psi
        if target_current == 0.0 or target_new / target_current > psi :
            print 'CHANGED with step ', distribution[i]
            target_current = target_new
            print 'current target: ', target_current
            if target_current == 0.0 or target_new / target_current > 1.0 :
                best_coeff = copy.deepcopy(coeff)
        else :
            print 'DON\'T CHANGE'
            coeff[direction] -= distribution[i]
    return (coeff, best_coeff)

def main() :
    samples = readData(TRAINING_SET)
    (coeff, best_coeff) = learnCoefficients(samples)

    test_samples = readData(TESTING_SET)
    f = open(OUTPUT, 'w')
    f_best = open(OUTPUT_BEST, 'w')
    for test_sample in test_samples :
        (x, y) = test_sample
        dot = multiplyVectors(x, coeff)
        dot_best = multiplyVectors(x, best_coeff)
        f.write('%s\n' % (str(dot)))
        f_best.write('%s\n' % (str(dot_best)))
    f.close()
    f_best.close()

if __name__ == "__main__" :
    main()
