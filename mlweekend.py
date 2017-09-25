# RESOURCES
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
# https://docs.python.org/2/library/random.html
# https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.poly1d.html#numpy.poly1d
# https://matplotlib.org/users/pyplot_tutorial.html
# https://docs.python.org/3/howto/urllib2.html
# http://docs.python-guide.org/en/latest/scenarios/json/
# https://shapeofdata.wordpress.com/2013/03/26/general-regression-and-over-fitting/
# http://enhancedatascience.com/2017/06/29/machine-learning-explained-overfitting/
# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

# ASSIGNMENT
# Since the summer was kind of boring, you applied for a temporary contract on the
# dark webIt turned out that you’re the only data scientist in the team; the
# others are mostly hackers doing industrial espionage. Long story short, after
# some work on their side, they provided you with VPN access to a factory
# in Kazakhstan. They’ve enabled you to call the endpoint of one of the machines
# where a secret mathematical formula is hidden. Your goal is to get the formula.
# Unfortunately, the machine works like a black box — you can just send an x-value
# to it
#
# /api/do_measurement?x=4 (the IP will be specified in the e-mail)
#
# to measure the y-value within the resulted JSON
#
# {"data": {"y": 188.58938971580736, "x": 4.0}}
#
# It looks like that the machine is just a working prototype, so the readings
# are a bit fuzzy and it’s probably a good idea to do more measurements to get
# some reasonable data. Also for some intervals of x, the y-values are missing;
# feel free to interpolate.
#
# Your task is do the measurements, get the secret formula f where y = f(x) and
# send it to the organisers (with a graph included). Please use Numpy to attach
# all the code you’ve used.
#
# Send your solution to mlweekend@kiwi.com no later than midnight on Sunday,
# October 22. We'll send you our evaluation by Tuesday, October 24. The sooner
# you send your solution, the sooner we'll let you know.

import urllib.request
import json
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

class Blackbox:
    '''
    class intended to handle blackbox function over the web api and to
    find best approximation of the measured data_x
    '''
    def __init__(self):
        '''
        initialisation
        url - string, url address of the api
        key - string, measured value name
        data_x, data_y - list, measured dataset
        train_x, train_y - numpy array, training dataset
        test_x, test_y - numpy array, testing dataset
        '''
        self.url = 'http://165.227.157.145:8080/api/do_measurement'
        self.key = 'x'
        self.data_x = []
        self.data_y = []
        self.train_x = np.array([])
        self.train_y = np.array([])
        self.test_x = np.array([])
        self.test_y = np.array([])
        self.best_order = 0

    def measure_data(self, count, limit):
        '''
        retrieve data from server, x values are randomly, uniformly
        chosen from the range -limit, +limit
        count - int, number of measured samples
        limit - float, max and min value for data_x
        '''
        for i in range(count):
            val = random.uniform(-limit, limit)
            url_values = {}
            url_values[self.key] = val
            url_values = urllib.parse.urlencode(url_values)
            full_url = self.url + '?' + url_values

            response = urllib.request.urlopen(full_url)
            raw_data = response.read()
            parsed_data = json.loads(raw_data)
            self.data_x.append(parsed_data['data']['x'])
            self.data_y.append(parsed_data['data']['y'])

    def clean_data(self):
        '''
        removes None values from the dataset
        '''
        none = []
        for i in range(len(self.data_x)):
            if self.data_x[i] is None:
                none.append(i)
        for i in range(len(none)):
            self.data_x.pop(none[i] - i)
            self.data_y.pop(none[i] - i)

    def split_data(self):
        '''
        splits measured data to training and testing dataset
        '''
        #self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.data_x, self.data_y, test_size=0.33, random_state=42)
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.data_x, self.data_y, test_size=0.3)

    def fitting_poly(self, max_poly):
        '''
        looking for best polynomial fit, target is maximum r2
        and minimum order
        max_poly - int, highest order potentialy tested
        poly - numpy poly1d, the polynom
        '''
        fits = []
        for i in range(1, max_poly + 1):
            poly = np.poly1d(np.polyfit(self.train_x, self.train_y, i))
            train_r2 = r2_score(self.train_y, poly(self.train_x))
            test_r2 = r2_score(self.test_y, poly(self.test_x))
            fits.append([i, train_r2, test_r2, 0, poly])
        fits = self.score_poly(fits)
        self.poly = (sorted(fits,key=lambda x: x[3], reverse=True)[0])[4]

    def score_poly(self, fits):
        '''
        compare fittings and appends score
        fits - list of lists, order, train_r2, test_r2, score
        '''
        # order score
        fits = sorted(fits,key=lambda x: x[0], reverse=True)
        for i in range(len(fits)):
            fits[i][3] += i
        # train_r2 score
        #fits = sorted(fits,key=lambda x: x[1])
        #for i in range(len(fits)):
        #    fits[i][3] += i
        # test_r2 score
        fits = sorted(fits,key=lambda x: x[2])
        for i in range(len(fits)):
            fits[i][3] += i
        #fits = sorted(fits,key=lambda x: x[0])
        #for i in range(len(fits)):
        #    print('{0:2d}, {1:5.2f}, {2:5.2f}, {3:3d}'.format(fits[i][0], fits[i][1], fits[i][2], fits[i][3]))
        return(fits)

    def plot_poly(self):
        '''
        prints final fitting with chart
        '''
        xp_max = max(map(abs, self.data_x))
        xp = np.linspace(-xp_max, xp_max, 100)
        plt.figure(figsize=(10,10))
        plt.title('Blackbox hides this equation' + '\n' +
                  self.poly_text(self.poly) + '\n' +
                  'r2 = ' + str('{0:.2f}'.format(r2_score(self.data_y, self.poly(self.data_x)))))
        plt.plot(self.data_x, self.data_y, '.', xp, self.poly(xp), '-')
        plt.legend(('data', 'model'), loc='upper center')
        plt.savefig('blackbox.png')
        plt.show()

    def poly_text(self, poly):
        '''
        returns string represents polynom in human readable format
        poly - numpy poly1d, polynom
        '''
        string = 'y = '
        for i in range(len(poly.c)):
            string += str('{0:+.2e}'.format(poly.c[i]))
            if i < len(poly.c) - 1:
                string += ' * x'
                if i < len(poly.c) - 2:
                    string += '^'
                    string += str(len(poly.c) - i - 1)
                string += ' '
        return(string)


# test case
limit = 1000000
count = 20
max_poly = 10

black_box = Blackbox()
black_box.measure_data(count, limit)
black_box.clean_data()
black_box.split_data()
black_box.fitting_poly(max_poly)
black_box.plot_poly()
