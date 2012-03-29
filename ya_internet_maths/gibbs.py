from math import exp
import numpy
import random

__author__ = 'Alexandra Mikhaylova mikhaylova@yandex-team.ru'

TRAINING_SET = 'train'
TESTING_SET = 'test'
OUTPUT = 'out'

STEPS = 100
C_EXP = 0.001
PSI = 0.8 # TODO: try normal distribution if needed
LEARN_SAMPLES_COUNT = 97290
FEATURES_COUNT = 245

def num(ls) :
    return [int(float(ls[0])), int(ls[1]), float(ls[2])]

def newVector() :
    v = []
    for x in xrange(FEATURES_COUNT) :
        v.append(0)
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
#            print sample_number
        if l[1] == FEATURES_COUNT + 1 :
            new_sample[1] = l[1]
        else :
            new_sample[0][l[1] - 1] = l[2]
    f.close()
    return samples

def mse(coeff, samples) : # sample = [X, y] where y is relevance
    result = 0.0
    for sample in samples :
        (x, y) = sample
        result += (y - multiplyVectors(x, coeff))**2
    result /= LEARN_SAMPLES_COUNT
#    print result
    return result

def target(coeff, samples) : # exp(-MSE)
    x = C_EXP * mse(coeff, samples)
    return exp((-1) * x)

def learnCoefficients(samples) :
    coeff = newVector()
    target_current = target(coeff, samples)
    distribution = numpy.random.normal(0.0, 1.0, STEPS) # here it's standard normal yet
    for i in xrange(STEPS) :
#        print 'Iteration ', i
        direction = random.randint(0, FEATURES_COUNT - 1)
        coeff[direction] += distribution[i]
        target_new = target(coeff, samples)
        if target_current == 0.0 or target_new / target_current > PSI :
#            print 'CHANGED with step ', distribution[i]
            target_current = target_new
#            print 'current target: ', target_current
        else :
#            print 'DON\'T CHANGE'
            coeff[direction] -= distribution[i]
    return coeff

def main() :
    samples = readData(TRAINING_SET)
    coeff = learnCoefficients(samples)

    test_samples = readData(TESTING_SET)
    f = open(OUTPUT, 'w')
    for test_sample in test_samples :
        (x, y) = test_sample
        dot = multiplyVectors(x, coeff)
        f.write('%s\n' % (str(dot)))
    f.close()

if __name__ == "__main__" :
    main()
