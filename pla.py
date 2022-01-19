import numpy as np
import pandas as pd
import matplotlib.lines as mlines
from matplotlib import pyplot as plt
import sys, csv


def visualize_scatter(df, weights, feat1=0, feat2=1, labels=2,
                      title=''):
    # Draw color-coded scatter plot
    colors = pd.Series(['r' if label > 0 else 'b' for label in df[labels]])
    ax = df.plot(x=feat1, y=feat2, kind='scatter', c=colors)

    # Get scatter plot boundaries to define line boundaries
    xmin, xmax = ax.get_xlim()

    # Compute and draw line. ax + by + c = 0  =>  y = -a/b*x - c/b
    a = weights[0]
    b = weights[1]
    c = weights[2]

    def y(x):
        return (-a / b) * x - c / b

    line_start = (xmin, xmax)
    line_end = (y(xmin), y(xmax))
    line = mlines.Line2D(line_start, line_end, color='red')
    ax.add_line(line)

    # if title == '':
    #     title = 'Scatter of feature %s vs %s' % (str(feat1), str(feat2))
    # ax.set_title(title)

    plt.show()


###
#   pla function
###
def pla(data):
    weight = np.zeros(len(data[0]))
    new_data = [np.copy(weight)]

    # adding affine dimension
    data = np.insert(data, 2, 1, axis=1)
    #     print(in_data)
    while True:
        converged = True
        for sample in data:
            feature = sample[:-1]
            label = sample[-1]
            if weight.dot(feature) * label <= 0:
                converged = False
                weight += label * feature
        if converged:
            break
        new_data.append(np.copy(weight))
    new_data.pop(0)
    return new_data


def main():

    # infile, outfile = 'data1.csv', 'result1.csv'
    infile, outfile = sys.argv[1], sys.argv[2]
    with open(infile, 'r') as in_file, open(outfile, 'w') as out_file:
        res = np.array(list(csv.reader(in_file)), dtype=float)
        weights = pla(res)

        data = pd.read_csv('data5.csv', header=None)
        w = weights[-1][:]
        visualize_scatter(data, weights=w)
        out_result = csv.writer(out_file, delimiter=',')
        for row in weights:
            out_result.writerow(row)

        #
        # fig = plt.figure()
        # ax = plt.axes()
        # # ax.plot(weight)
        # plt.show()
    # data = pd.read_csv('data5.csv', header=None)
    # visualize_scatter(data, weights=[1,1,1,1,-1,-1,-1])
if __name__ == '__main__':
    main()
