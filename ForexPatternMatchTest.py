__author__ = 'Shashank'

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import numpy as np
import time

total_start = time.time()

date, bid, ask = np.loadtxt('./data/GBPUSD1d.txt', unpack=True, delimiter=',',
                            converters={0: mdates.strpdate2num('%Y%m%d%H%M%S')})


def percent_change(start_point, current_point):
    try:
        x = (current_point - start_point) * 100.00 / float(start_point)
        if x == 0:
            return 0.0000000001
        return x
    except:
        return 0.0000000001


def pattern_storage():
    pat_start_time = time.time()
    x = len(avg_line) - 2*pattern_length

    y = 31

    while y < x:

        p = list(avg_line[y - pattern_length: y + 1])

        for i in range(1, pattern_length + 1):
            p[i] = percent_change(p[0], p[i])

        outcome_range = avg_line[y + 20: y + 30]
        current_point = avg_line[y]

        try:
            avg_outcome = sum(outcome_range) / float(len(outcome_range))
        except Exception, e:
            print str(e)
            avg_outcome = 0

        future_outcome = percent_change(current_point, avg_outcome)

        patternAr.append(list(p[1:]))
        performanceAr.append(future_outcome)

        y += 1

    pat_end_time = time.time()

    print len(patternAr)
    print len(performanceAr)
    print 'Pattern storage took:', pat_end_time - pat_start_time, 'seconds'


def current_pattern():
    for i in range(0, pattern_length):
        patternInRec.append(percent_change(avg_line[-pattern_length-1], avg_line[i - pattern_length]))

    print patternInRec
    print '#########################################'


def pattern_similarity(pattern_1, pattern_2):
    if (len(pattern_1) != len(pattern_2)):
        return 0

    sim = []
    for i in range(0, len(pattern_1)):
        sim.append(100.0 - abs(percent_change(pattern_1[i], pattern_2[i])))

    return sum(sim) / float(len(sim))


def pattern_recognition():

    patfound = 0
    plotpatAr = []
    predictedoutcomesAr = []

    for each_pattern in patternAr:
        similarity = pattern_similarity(each_pattern, patternInRec)

        if similarity > 70:
            patfound = 1
            plotpatAr.append(each_pattern)

    if patfound == 1:
        #fig = plt.figure(figsize=(10, 6))
        #xp = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
        #          29, 30]

        predAr = []

        for eachpat in plotpatAr:
            future_points = patternAr.index(eachpat)

            if performanceAr[future_points] > eachpat[pattern_length-1]:
            #    pcolor = '#24bc00'
                predAr.append(1.00)
            else:
            #    pcolor = '#d40000'
                predAr.append(-1.00)

            #plt.plot(xp, eachpat)
            #plt.scatter(35, performanceAr[future_points], c=pcolor, alpha=.3)
            predictedoutcomesAr.append(performanceAr[future_points])

        real_outcome_range = all_data[towhat+20: towhat+30]
        real_outcome_avg = reduce(lambda x, y: x+y, real_outcome_range)/len(real_outcome_range)
        real_movement = percent_change(all_data[towhat], real_outcome_avg)
        predictedAvgOutcome = sum(predictedoutcomesAr) / float(len(predictedoutcomesAr))

        print predAr
        predictionAvg = sum(predAr) / float(len(predAr))

        print predictionAvg
        if predictionAvg < 0:
            print 'drop predicted'
            print patternInRec[pattern_length-1]
            print real_movement
            if real_movement < patternInRec[pattern_length-1]:
                accuracyAr.append(100)
            else:
                accuracyAr.append(0)

        if predictionAvg > 0:
            print 'rise predicted'
            print patternInRec[pattern_length-1]
            print real_movement
            if real_movement > patternInRec[pattern_length-1]:
                accuracyAr.append(100)
            else:
                accuracyAr.append(0)

        #plt.scatter(40, real_movement, c='#54fff7', s=25)
        #plt.scatter(40, predictedAvgOutcome, c='b', s=25)

        #plt.plot(xp, patternInRec, c='#54fff7', linewidth=3)
        #plt.grid = True
        #plt.title('Pattern Recognition')
        #plt.show()


def graphRawFX():
    fig = plt.figure(figsize=(10, 7))

    ax1 = plt.subplot2grid((40, 40), (0, 0), rowspan=40, colspan=40)

    ax1.plot(date, bid)
    ax1.plot(date, ask)
    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))

    ax_2 = ax1.twinx()

    ax_2.fill_between(date, 0, ask - bid, facecolor='g', alpha=.3)

    for label in ax1.xaxis.get_ticklabels():
        label.set_rotation(45)

    plt.subplots_adjust(bottom=.23)

    plt.grid(True)
    plt.show()


datalength = int(bid.shape[0])
print 'Data length : ', datalength

towhat = 37000
all_data = ((bid + ask)/2)
pattern_length = 30

accuracyAr = []
samps = 0

while towhat < datalength:
    avg_line = all_data[:towhat]

    patternAr = []
    performanceAr = []
    patternInRec = []

    pattern_storage()
    current_pattern()
    pattern_recognition()
    print 'Total time taken : ', time.time() - total_start
    #moveOn = raw_input('Press Enter to continue.........')
    samps += 1
    towhat += 1
    accuracyAverage = sum(accuracyAr) / float(len(accuracyAr))
    print 'Backtested accuracy is ',str(accuracyAverage),'% after ',samps,' samples'